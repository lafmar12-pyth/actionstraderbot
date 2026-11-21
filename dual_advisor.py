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


def _parse_ai_decision(text: str) -> str:
    """Extrait proprement VALIDÉ ou VETO depuis la réponse de l'IA."""
    for line in text.splitlines():
        clean_line = line.strip().upper()
        if clean_line.startswith("DECISION"):
            if "VETO" in clean_line:
                return "VETO"
            if "VALIDÉ" in clean_line or "VALIDE" in clean_line:
                return "VALIDÉ"
    upper = text.upper()
    if "VETO" in upper and "VALIDÉ" not in upper:
        return "VETO"
    return "VALIDÉ" if "VALIDÉ" in upper else "VETO"


def _query_gemini_with_retry(prompt, gemini_key, max_retries=3):
    """Interroge Gemini avec gestion spécifique des timeouts 504/503."""
    genai.configure(api_key=gemini_key)
    g_model = genai.GenerativeModel("gemini-3.8-flash")

    for attempt in range(1, max_retries + 1):
        try:
            g_resp = g_model.generate_content(
                prompt,
                request_options={"timeout": 25},
            )
            text = (g_resp.text or "").strip()
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
    Interroge Gemini et OpenAI avec audit sectoriel/macro actions (US & Canada)
    et repli quantitatif local si les clés/API échouent.
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
1. CORRÉLATION SECTORIELLE : L'ajout de {symbol} crée-t-il une concentration excessive sur un même secteur par rapport aux positions déjà ouvertes ({recent_history}) ?
2. CALENDRIER ENTREPRISE & MACRO : Y a-t-il une publication de résultats financiers (Earnings) sur {symbol} ou une annonce majeure (Fed, Banque du Canada, CPI) dans les prochaines 48h ?
3. CONFIRMATION PAR LE VOLUME & LE MARCHÉ : Si le marché global ({trend}) est baissier, un achat est autorisé **uniquement si le score technique est très élevé (>= 78/100)** et que le titre montre une force évidente. Si le volume relatif est faible (< 0.8x), réponds VETO.
4. QUALITÉ DU RENFORCEMENT : S'il s'agit d'un pyramiding, le momentum justifie-t-il réellement d'augmenter la taille de la position ?

    Règle de décision : Si l'un des 4 points présente un risque majeur, réponds VETO. Sinon, réponds VALIDÉ.

    Format de réponse strict :
    DECISION: [VALIDÉ ou VETO]
    RAISON: [Ton explication concise en une phrase citant le critère déterminant]
    """

    gemini_decision, gemini_reason, gemini_api_failed = "VETO", "Clé API GEMINI_API_KEY absente", True
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        gemini_decision, gemini_reason, gemini_api_failed = _query_gemini_with_retry(prompt, gemini_key)
    if gemini_api_failed and gemini_key:
        send_ai_error_alert("Gemini", f"[{symbol}] {gemini_reason}")

    openai_decision, openai_reason, openai_api_failed = "VETO", "Clé API OPENAI_API_KEY absente", True
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        for attempt in range(1, 3):
            try:
                client = OpenAI(api_key=openai_key)
                o_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                text = (o_resp.choices[0].message.content or "").strip()
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
    if openai_api_failed and openai_key:
        send_ai_error_alert("OpenAI", f"[{symbol}] {openai_reason}")

    print(f"\n💬 [Comité Gemini] : {gemini_reason}", flush=True)
    print(f"💬 [Comité OpenAI] : {openai_reason}", flush=True)

    # Si les deux IA ont échoué techniquement, repli quantitatif local prudent (sans bloquer sur un bug réseau)
    if gemini_api_failed and openai_api_failed:
        if score >= 80 and "BAISSIER" not in str(trend).upper():
            return "VALIDÉ", "Repli quantitatif local : score >= 80 hors marché baissier (IA indisponibles)."
        return "VETO", "Repli quantitatif local : IA indisponibles et score/contexte insuffisant."

    # Règle du comité : tout VETO explicite d'une IA qui a répondu bloque l'entrée
    if (not gemini_api_failed and gemini_decision == "VETO") or (not openai_api_failed and openai_decision == "VETO"):
        return "VETO", "Veto prononcé par au moins un membre du comité."

    if gemini_decision == "VALIDÉ" or openai_decision == "VALIDÉ":
        return "VALIDÉ", "Configuration validée par le comité."

    return "VETO", "Aucun consensus de validation."
