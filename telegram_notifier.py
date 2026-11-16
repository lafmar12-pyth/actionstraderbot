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


def send_ai_error_alert(ai_name, symbol, error_details):
    """
    Envoie une alerte technique spécifique lorsqu'une IA (Gemini ou OpenAI) échoue.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return

    # Nettoyage des caractères spéciaux pour éviter tout rejet Markdown de Telegram
    clean_error = str(error_details).replace("_", "-").replace("*", "").replace("`", "")
    # Limite la longueur du message d'erreur pour garder une notification lisible
    if len(clean_error) > 350:
        clean_error = clean_error[:350] + "..."

    message = (
        f"⚠️ *ERREUR API IA ({ai_name})*\n"
        f"• *Actif analysé :* {symbol}\n"
        f"• *Détail :* {clean_error}"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        requests.post(url, json=payload, timeout=10)
        print(f"📱 [Telegram] Alerte d'erreur IA ({ai_name}) transmise.", flush=True)
    except Exception as e:
        print(f"⚠️ [Telegram] Impossible d'envoyer l'alerte d'erreur IA : {e}", flush=True)
