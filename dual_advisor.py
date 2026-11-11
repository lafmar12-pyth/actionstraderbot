# dual_advisor.py
import os
import time
import functools
import google.generativeai as genai
from openai import OpenAI

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

@safe_api_call(max_retries=3, delay=2)
def ask_dual_ai_decision(symbol, price, score, reason, trend, recent_history, memory_summary):
    """
    Interroge Gemini et OpenAI avec protection réseau. 
    Les deux doivent valider pour donner un feu vert. En cas d'erreur, VETO par sécurité.
    """
    prompt = f"""
    En tant que comité des risques senior d'un fonds quantitatif crypto, analyse cette décision d'investissement à court terme :
    - Actif candidat : {symbol} à {price}$
    - Score technique (Calculs) : {score}/100 ({reason})
    - Tendance globale (Macro BTC) : {trend}
    - Règle de positionnement : Le capital engagé est strictement limité à 1.6% du portefeuille total.
    - Horizon temporel : La prochaine fenêtre d'analyse et de réévaluation du bot se fera exactement dans 1h30. L'IA doit donc évaluer si le risque et le momentum permettent de tenir ou de capter un mouvement sur cette fenêtre de 1h30.
    - Historique récent : {recent_history}
    - Mémoire des erreurs passées : {memory_summary}
    
    Règles strictes :
    Si la configuration présente un risque de faux signal ou contredit la mémoire des pertes passées, réponds VETO. 
    Sinon, réponds VALIDÉ.
    
    Format de réponse strict :
    DECISION: [VALIDÉ ou VETO]
    RAISON: [Ton explication en une phrase]
    """

    # 1. Avis de Gemini (Par défaut sur VETO par sécurité)
    gemini_decision, gemini_reason = "VETO", "Non évalué / Erreur"
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            # Utilisation d'un modèle flash récent et stable
            g_model = genai.GenerativeModel('gemini-3.8-flash')
            g_resp = g_model.generate_content(prompt)
            text = g_resp.text.strip()
            gemini_decision = "VALIDÉ" if "VALIDÉ" in text.upper() else "VETO"
            gemini_reason = text
    except Exception as e:
        gemini_reason = f"Erreur Gemini: {e}"

    # 2. Avis d'OpenAI (ChatGPT) (Par défaut sur VETO par sécurité)
    openai_decision, openai_reason = "VETO", "Non évalué / Erreur"
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            client = OpenAI(api_key=openai_key)
            o_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            text = o_resp.choices[0].message.content.strip()
            openai_decision = "VALIDÉ" if "VALIDÉ" in text.upper() else "VETO"
            openai_reason = text
    except Exception as e:
        openai_reason = f"Erreur OpenAI: {e}"

    # 3. Arbitrage du Double Check
    print(f"\n💬 [Comité Gemini] : {gemini_reason}", flush=True)
    print(f"💬 [Comité OpenAI] : {openai_reason}", flush=True)

    if gemini_decision == "VALIDÉ" and openai_decision == "VALIDÉ":
        return "VALIDÉ", "Double consensus IA obtenu (Gemini + OpenAI)."
    else:
        return "VETO", "Désaccord des IA, absence de crédits ou Veto prononcé."
