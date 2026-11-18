# dual_advisor.py
import os
import time
import functools
from google import genai
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

def _parse_ai_decision(text: str) -> str:
    """Extrait proprement VALIDÉ ou VETO depuis la ligne DECISION de l'IA."""
    for line in text.splitlines():
        clean_line = line.strip().upper()
        if clean_line.startswith("DECISION"):
            return "VALIDÉ" if "VALIDÉ" in clean_line else "VETO"
    return "VALIDÉ" if "VALIDÉ" in text.upper() else "VETO"

def _query_gemini_with_retry(prompt, gemini_key, max_retries=3):
    """Interroge Gemini (via le nouveau SDK google.genai) avec gestion des retries."""
    client = genai.Client(api_key=gemini_key)
    
    for attempt in range(1, max_retries + 1):
        try:
            g_resp = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt
            )
            text = g_resp.text.strip()
            decision = _parse_ai_decision(text)
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
    Interroge Gemini et OpenAI avec audit macro/sectoriel, secours quantitatif et alerte Telegram sur erreur.
    """
    prompt = f"""
    En tant que comité des risques senior d'un fonds quantitatif actions (US & Canada), analyse cette décision de swing trading :
    - Actif candidat : {symbol} à {price}$
    - Signaux techniques & Volume : {score}/100 ({reason})
    - Tendance générale du marché (S&P 500) : {trend}
    - Règle de gestion : 2% du portefeuille par tranche (pyramiding progressif autorisé jusqu'à 8% max par action en 4 tranches, plafond global 40%).
    - Horizon temporel : Swing trading multi-jours (holding minimum 1 jour pour éviter le Pattern Day Trading).
    - Positions déjà ouvertes en portefeuille : {recent_history}
    - Consignes de vigilance : {memory_summary}
    
    Checklist d'audit obligatoire avant validation :
    1. CORRÉLATION SECTORIELLE : L'ajout de {symbol} crée-t-il une concentration excessive sur un même secteur par rapport aux positions déjà ouvertes ({recent_history}) ? (Ex: éviter d'accumuler trop de valeurs Tech/Semi-conducteurs ou trop de Banques canadiennes simultanément).
    2. CALENDRIER ENTREPRISE & MACRO : Y a-t-il une publication de résultats financiers (Earnings) sur {symbol} ou une annonce majeure (Fed, Banque du Canada, CPI) dans les prochaines 48h ?
    3. CONFIRMATION PAR LE VOLUME & LE MARCHÉ : Si le marché global ({trend}) est baissier ou si le volume relatif est faible (inférieur à 0.8x la moyenne 20j), le signal est-il suffisamment robuste ?
    4. QUALITÉ DU RENFORCEMENT : S'il s'agit d'un pyramiding, le momentum justifie-t-il réellement d'augmenter la taille de la position ?

    Règle de décision : Si l'un des 4 points présente un risque élevé, réponds VETO. Sinon, réponds VALIDÉ.
    
    Format de réponse strict :
    DECISION: [VALIDÉ ou VETO]
    RAISON: [Ton explication concise en une phrase citant le critère déterminant]
    """

    # 1. Avis de Gemini (via le nouveau client google.genai)
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
                openai_decision = _parse_ai_decision(text)
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
