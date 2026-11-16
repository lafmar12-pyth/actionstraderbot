# dual_advisor.py
import os
import time
import functools
import google.generativeai as genai
from openai import OpenAI
from telegram_notifier import send_ai_error_alert

def safe_api_call(max_retries=3, delay=3):
    """Décorateur pour retenter automatiquement un appel en cas d'erreur réseau."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ Erreur réseau dans {func.__name__} (Tentative {attempt}/{max_retries}): {e}", flush=True)
                    if attempt == max_retries:
                        raise e
                    time.sleep(delay)
        return wrapper
    return decorator

def _query_gemini_with_retry(prompt, gemini_key, max_retries=3):
    """Interroge Gemini avec gestion spécifique des timeouts 504/503."""
    genai.configure(api_key=gemini_key)
    g_model = genai.GenerativeModel('gemini-3.8-flash')
    
    for attempt in range(1, max_retries + 1):
        try:
            g_resp = g_model.generate_content(
                prompt,
                request_options={"timeout": 25}
            )
            text = g_resp.text.strip()
            decision = "VALIDÉ" if "VALIDÉ" in text.upper() else "VETO"
            return decision, text, False
        except Exception as e:
            err_str = str(e)
            is_transient = any(code in err_str for code in ["504", "503", "Deadline", "Timeout", "UNAVAILABLE"])
            if is_transient and attempt < max_retries:
                wait_time = 3 * attempt
                print(f"⏳ Latence Gemini ({err_str[:40]}...). Nouvelle tentative {attempt}/{max_retries} dans {wait_time}s...", flush=True)
                time.sleep(wait_time)
                continue
            return "VETO", f"Erreur Gemini: {err_str}", True

def ask_dual_ai_decision(symbol, price, score, reason, trend, recent_history, memory_summary):
    """
    Interroge Gemini et OpenAI avec protection réseau, secours quantitatif et alerte Telegram sur erreur.
    """
    prompt = f"""
    En tant que comité des risques senior d'un fonds quantitatif actions, analyse cette décision d'investissement à court terme :
    - Actif candidat : {symbol} à {price}$
    - Score technique (Calculs) : {score}/100 ({reason})
    - Tendance globale (Macro) : {trend}
    - Règle de positionnement : Le capital engagé est strictement limité à 2% du portefeuille total.
    - Horizon temporel : La prochaine fenêtre d'analyse et de réévaluation du bot se fera lors du prochain cycle. L'IA doit donc évaluer si le risque et le momentum permettent de tenir ou de capter un mouvement de swing trading.
    - Historique récent : {recent_history}
    - Mémoire des erreurs passées : {memory_summary}
    
    Règles strictes :
    Si la configuration présente un risque de faux signal ou contredit la mémoire des pertes passées, réponds VETO. 
    Sinon, réponds VALIDÉ.
    
    Format de réponse strict :
    DECISION: [VALIDÉ ou VETO]
    RAISON: [Ton explication en une phrase]
    """

    # 1. Avis de Gemini (avec retry dédié anti-504)
    gemini_decision, gemini_reason, gemini_api_failed = "VETO", "Clé API GEMINI_API_KEY absente", True
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        gemini_decision, gemini_reason, gemini_api_failed = _query_gemini_with_retry(prompt, gemini_key)
    
    if gemini_api_failed:
        send_ai_error_alert("Gemini", f"[{symbol}] {gemini_reason}")

    # 2. Avis d'OpenAI (ChatGPT)
    openai_decision, openai_reason, openai_api_failed = "VETO", "Clé API OPENAI_API_KEY absente", True
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        for attempt in range(1, 3):
            try:
                client = OpenAI(api_key=openai_key)
                o_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                text = o_resp.choices[0].message.content.strip()
                openai_decision = "VALIDÉ" if "VALIDÉ" in text.upper() else "VETO"
                openai_reason = text
                openai_api_failed = False
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                    continue
                openai_reason = f"Erreur OpenAI: {e}"
                openai_api_failed = True

    if openai_api_failed:
        send_ai_error_alert("OpenAI", f"[{symbol}] {openai_reason}")

    # 3. Affichage des rapports et Arbitrage du Double Check
    print(f"\n💬 [Comité Gemini] : {gemini_reason}", flush=True)
    print(f"💬 [Comité OpenAI] : {openai_reason}", flush=True)

    # Cas idéal : Double validation unanime
    if gemini_decision == "VALIDÉ" and openai_decision == "VALIDÉ":
        return "VALIDÉ", "Double consensus IA obtenu (Gemini + OpenAI)."

    # Cas de secours : Une des deux IA subit une panne réseau/quota, l'autre valide ET score >= 80
    if gemini_api_failed and not openai_api_failed and openai_decision == "VALIDÉ" and score >= 80:
        return "VALIDÉ", f"Validation hybride (OpenAI + Secours Quantitatif Score {score}/100 suite à indisponibilité Gemini)."

    if openai_api_failed and not gemini_api_failed and gemini_decision == "VALIDÉ" and score >= 80:
        return "VALIDÉ", f"Validation hybride (Gemini + Secours Quantitatif Score {score}/100 suite à indisponibilité OpenAI)."

    # Cas par défaut : Veto ou désaccord réel
    return "VETO", "Désaccord des IA, absence de crédits ou Veto prononcé."
