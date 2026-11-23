# test_telegram.py
import os
import sys
from datetime import datetime, timezone
from telegram_notifier import send_telegram_alert


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("❌ Variables TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquantes.", flush=True)
        return 1

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("1/2 Envoi d'une alerte Telegram standard...", flush=True)
    send_telegram_alert(
        f"✅ *Test Telegram réussi* (`Stock Swing Trading Bot`)\n"
        f"• Horodatage : `{now_str}`\n"
        f"• Statut : Notification opérationnelle"
    )

    print("2/2 Envoi d'une alerte avec caractères spéciaux (test repli Markdown)...", flush=True)
    send_telegram_alert(
        f"Test repli Markdown _ * ` [ prix=125.50$ _test ({now_str})"
    )

    print("✅ Script test_telegram.py terminé.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
