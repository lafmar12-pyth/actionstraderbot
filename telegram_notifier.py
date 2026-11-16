# telegram_notifier.py
import os
import requests

def send_telegram_alert(message_body):
    """
    Envoie une alerte instantanée via un Bot Telegram.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ℹ️ [Telegram] Identifiants non configurés, alerte ignorée.", flush=True)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"🤖 *[BOT TRADING]*\n{message_body}",
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("📱 [Telegram] Alerte envoyée avec succès sur votre téléphone !", flush=True)
        else:
            print(f"⚠️ [Telegram] Erreur d'envoi : {response.text}", flush=True)
    except Exception as e:
        print(f"⚠️ [Telegram] Exception réseau : {e}", flush=True)


def send_ai_error_alert(ai_name: str, error_details: str):
    """
    Alerte spécifique déclenchée lorsqu'une IA (Gemini ou OpenAI) rencontre une erreur.
    """
    message = (
        f"🧠❌ *ERREUR DE L'IA : {ai_name.upper()}*\n\n"
        f"Le comité a échoué à obtenir une réponse valide :\n"
        f"`{error_details}`"
    )
    send_telegram_alert(message)
