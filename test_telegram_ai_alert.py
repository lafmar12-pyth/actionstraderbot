# test_telegram_ai_alert.py
import os
import sys
from datetime import datetime, timezone
from telegram_notifier import send_ai_error_alert


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("❌ Variables TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquantes.", flush=True)
        return 1

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sample_error = f"Simulated 503 UNAVAILABLE timeout during test ({now_str})"

    print("1/2 Envoi d'une alerte d'erreur IA simulée...", flush=True)
    send_ai_error_alert("Gemini", sample_error)

    print("2/2 Vérification de l'anti-doublon sur la même erreur...", flush=True)
    send_ai_error_alert("Gemini", sample_error)

    print("✅ Script test_telegram_ai_alert.py terminé.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
