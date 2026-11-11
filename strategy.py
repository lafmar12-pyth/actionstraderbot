import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dual_advisor import ask_dual_ai_decision
from market_memory import MarketMemoryManager

class StockSwingEngine:
    def __init__(self, initial_capital=1000.0):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        # paper=True pour utiliser le mode simulation Alpaca
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        
        # Watchlist d'actions US et canadiennes (.TO)
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
            'SHOP.TO', 'TD.TO', 'RY.TO', 'CNR.TO'
        ]
        self.memory = MarketMemoryManager()

    def is_market_open(self):
        """Vérifie si la bourse est ouverte."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"⚠️ Impossible de vérifier l'état du marché: {e}")
            return True

    def evaluate_stock_opportunity(self, df):
        """Analyse technique journalière (1D) : SMA20, SMA50 et RSI(14)."""
        if len(df) < 50:
            return 0, "Données insuffisantes (<50 bougies)"

        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        score = 0
        reasons = []

        if last_row['SMA20'] > last_row['SMA50']:
            score += 35
            reasons.append("Tendance haussière (SMA20 > SMA50)")

        if prev_row['Low'] <= prev_row['SMA20'] and last_row['Close'] > last_row['SMA20']:
            score += 35
            reasons.append("Rebond technique sur la SMA20")

        if 40 <= last_row['RSI'] <= 65:
            score += 30
            reasons.append(f"RSI optimal ({last_row['RSI']:.1f})")

        return score, " | ".join(reasons) if reasons else "Pas de signal pertinent"

    def run_cycle(self):
        print(f"[{datetime.utcnow().isoformat()}] Lancement du cycle Swing Trading Actions...", flush=True)

        if not self.is_market_open():
            print("💤 Le marché boursier est actuellement FERMÉ. Cycle ignoré.", flush=True)
            return

        account = self.trading_client.get_account()
        total_portfolio = float(account.equity)
        allocated_capital = total_portfolio * 0.02  # 2% max par position

        print(f"💰 Capital total portefeuille : {total_portfolio:.2f} $ | Allocation max (2%) : {allocated_capital:.2f} $")

        candidates = []

        print(f"\n🔍 Scan des {len(self.symbols)} actions de la watchlist...", flush=True)
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="60d", interval="1d")

                if df.empty:
                    continue

                current_price = float(df['Close'].iloc[-1])
                score, reason = self.evaluate_stock_opportunity(df)

                print(f"   • {symbol:10} | Prix: {current_price:8.2f}$ | Score: {score:3}/100 | {reason}")

                if score >= 65:
                    candidates.append({
                        'symbol': symbol,
                        'price': current_price,
                        'score': score,
                        'reason': reason
                    })
            except Exception as e:
                print(f"   ⚠️ Erreur d'analyse sur {symbol}: {e}")

        if candidates and not self.memory.current_position:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_choice = candidates[0]

            print(f"\n🧠 Soumission du dossier Swing {best_choice['symbol']} au comité multi-IA...")
            
            clean_symbol = best_choice['symbol'].replace('.TO', '')

            decision, feedback = ask_dual_ai_decision(
                clean_symbol,
                best_choice['price'],
                best_choice['score'],
                best_choice['reason'],
                "HAUSSIER_MARCHE",
                self.memory.lessons,
                "Aucune leçon critique"
            )

            if decision == "VALIDÉ":
                qty_to_buy = round(allocated_capital / best_choice['price'], 2)
                if qty_to_buy < 0.01:
                    qty_to_buy = 1.0

                print(f"🔥 Exécution de l'ordre d'achat Alpaca : {qty_to_buy} actions de {clean_symbol}")
                
                market_order_data = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=qty_to_buy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order_data=market_order_data)
                self.memory.current_position = clean_symbol
                self.memory.entry_price = best_choice['price']
                self.memory.save_state()
                self.memory.add_lesson(f"Achat validé de {clean_symbol} à {best_choice['price']}$ (Score: {best_choice['score']}/100).")
            else:
                print(f"🛡️ [VETO IA] Ordre annulé : {feedback}")
                self.memory.add_lesson(f"Achat rejeté pour {clean_symbol} : {feedback}")
        else:
            print("📌 Aucun candidat éligible ou position déjà active.")
