# stock_main_runner.py
import traceback
from core_stock_engine import StockSwingEngine
from telegram_notifier import send_telegram_alert


def main():
    print("🚀 Démarrage du runner pour le Swing Trading Actions...", flush=True)
    engine = None
    try:
        engine = StockSwingEngine(initial_capital=1000.0)
        engine.run_cycle()
        print("✅ Cycle Swing Trading Actions terminé proprement.", flush=True)
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        print(f"❌ [ERREUR CRITIQUE RUNNER] {err_msg}", flush=True)
        print(traceback.format_exc(), flush=True)

        if engine is not None:
            try:
                engine.save_state()
            except Exception as save_err:
                print(f"⚠️ Impossible de sauvegarder l'état après erreur : {save_err}", flush=True)

        send_telegram_alert(f"🚨 *ERREUR RUNNER ACTIONS*\n`{err_msg[:400]}`")
        raise


if __name__ == "__main__":
    main()
