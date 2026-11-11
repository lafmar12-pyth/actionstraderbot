# stock_main_runner.py
from core_stock_engine import StockSwingEngine

if __name__ == "__main__":
    print("🚀 Démarrage du runner pour le Swing Trading Actions...", flush=True)
    engine = StockSwingEngine(initial_capital=1000.0)
    engine.run_cycle()
