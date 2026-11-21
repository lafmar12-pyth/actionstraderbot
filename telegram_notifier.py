# telegram_notifier.py
import os
import requests

_LAST_ERROR_CACHE: set[str] = set()


def send_telegram_alert(message_body: str):
    """
    Envoie une alerte instantanée via un Bot Telegram, avec repli texte brut
    si le formatage Markdown provoque une erreur de syntaxe côté Telegram.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ℹ️ [Telegram] Identifiants non configurés, alerte ignorée.", flush=True)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    formatted_text = f"🤖 *[BOT TRADING]*\n{message_body}"
    payload = {
        "chat_id": chat_id,
        "text": formatted_text,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("📱 [Telegram] Alerte envoyée avec succès sur votre téléphone !", flush=True)
            return

        # Repli en texte brut si Markdown est rejeté par l'API Telegram
        if response.status_code == 400 and "can't parse entities" in response.text.lower():
            fallback_payload = {
                "chat_id": chat_id,
                "text": f"🤖 [BOT TRADING]\n{message_body}",
            }
            fallback_resp = requests.post(url, json=fallback_payload, timeout=10)
            if fallback_resp.status_code == 200:
                print("📱 [Telegram] Alerte envoyée (texte brut après repli Markdown).", flush=True)
                return

        print(f"⚠️ [Telegram] Erreur d'envoi : {response.text}", flush=True)
    except Exception as e:
        print(f"⚠️ [Telegram] Exception réseau : {e}", flush=True)


def send_ai_error_alert(ai_name: str, error_details: str):
    """
    Alerte déclenchée lorsqu'une IA (Gemini ou OpenAI) rencontre une erreur réelle,
    en évitant de spammer la même erreur sur un même processus.
    """
    details_str = str(error_details)
    if "clé api" in details_str.lower() and "absente" in details_str.lower():
        return

    cache_key = f"{ai_name.upper()}:{details_str[:120]}"
    if cache_key in _LAST_ERROR_CACHE:
        return
    _LAST_ERROR_CACHE.add(cache_key)

    clean_details = details_str.replace("`", "'")
    message = (
        f"🧠❌ *ERREUR DE L'IA : {ai_name.upper()}*\n\n"
        f"Le comité a échoué à obtenir une réponse valide :\n"
        f"`{clean_details[:500]}`"
    )
    send_telegram_alert(message)
