# core_stock_engine.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dual_advisor import ask_dual_ai_decision
from telegram_notifier import send_telegram_alert

STATE_FILE = "stock_bot_state.json"

class StockSwingEngine:
    def __init__(self, initial_capital=1000.0):
        # Initialisation Alpaca (Paper Trading)
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Les clés ALPACA_API_KEY ou ALPACA_SECRET_KEY sont manquantes.")

        # Client de trading (paper=True pour l'environnement de test)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        
        # Client de données historiques Alpaca
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Watchlist élargie (25 actions liquides US & Canadiennes cotées aux US)
        # Nécessaire pour pouvoir atteindre jusqu'à 20 positions (20 x 2% = 40% max du capital)
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
            'SHOP', 'TD', 'RY', 'CNI', 'BNS',
            'JPM', 'V', 'MA', 'UNH', 'HD', 'PG', 'COST', 'CRM', 'AVGO', 'NFLX', 'DIS', 'PEP'
        ]
        
        # Paramètres de gestion du risque Swing Trading
        self.risk_per_trade = 0.02      # 2% par position
        self.max_capital_usage = 0.40   # 40% maximum du portefeuille engagé simultanément
        self.take_profit_pct = 0.06     # Prise de profit à +6%
        self.stop_loss_pct = -0.03      # Stop-loss à -3%
        
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    self.trade_history = json.load(f)
                    return
            except Exception:
                pass
        self.trade_history = {"last_run": None, "trades": []}

    def save_state(self):
        self.trade_history["last_run"] = datetime.now(timezone.utc).isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(self.trade_history, f, indent=4)

    def is_market_open(self):
        """Vérifie auprès de l'API si la bourse est ouverte aujourd'hui."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"⚠️ Impossible de vérifier l'état du marché: {e}")
            return False

    def get_historical_bars(self, symbol: str, calendar_days: int = 90) -> pd.DataFrame:
        """Récupère les bougies journalières via l'API Alpaca."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=calendar_days)

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX
        )

        bars = self.data_client.get_stock_bars(request_params)
        df = bars.df

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        return df

    def evaluate_stock_opportunity(self, df):
        """Logique de Swing Trading sur bougies Journalières (1D) : SMA20 / SMA50 et RSI(14)."""
        if len(df) < 50:
            return 0, f"Données insuffisantes ({len(df)} bougies)"

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
            reasons.append("Rebond sur la SMA20")

        if 40 <= last_row['RSI'] <= 65:
            score += 30
            reasons.append(f"RSI optimal ({last_row['RSI']:.1f})")

        return score, " | ".join(reasons) if reasons else "Pas de signal"

    def manage_open_positions(self, open_positions):
        """Vérifie le Take-Profit (+6%) et le Stop-Loss (-3%) sur les positions ouvertes."""
        if not open_positions:
            return 0

        print(f"\n📋 Suivi des {len(open_positions)} position(s) ouverte(s) :", flush=True)
        closed_count = 0

        for pos in open_positions:
            symbol = pos.symbol
            unrealized_plpc = float(pos.unrealized_plpc)  # Rendement en décimal (ex: 0.06 = +6%)
            unrealized_pl = float(pos.unrealized_pl)
            market_value = float(pos.market_value)

            print(f"   • {symbol:6} | Valeur: {market_value:8.2f}$ | P&L: {unrealized_plpc*100:+6.2f}% ({unrealized_pl:+.2f}$)")

            if unrealized_plpc >= self.take_profit_pct:
                print(f"   🎯 TAKE-PROFIT atteint sur {symbol} ({unrealized_plpc*100:+.2f}%) -> Clôture !")
                self.trading_client.close_position(symbol)
                send_telegram_alert(f"🎯 *TAKE-PROFIT* sur `{symbol}` : `{unrealized_plpc*100:+.2f}%` (`{unrealized_pl:+.2f}$`)")
                closed_count += 1

            elif unrealized_plpc <= self.stop_loss_pct:
                print(f"   🛑 STOP-LOSS atteint sur {symbol} ({unrealized_plpc*100:+.2f}%) -> Clôture !")
                self.trading_client.close_position(symbol)
                send_telegram_alert(f"🛑 *STOP-LOSS* sur `{symbol}` : `{unrealized_plpc*100:+.2f}%` (`{unrealized_pl:+.2f}$`)")
                closed_count += 1

        return closed_count

    def run_cycle(self):
        print(f"[{datetime.now(timezone.utc).isoformat()}] Lancement du cycle Swing Trading Actions...", flush=True)

        # 1. Vérification si le marché boursier est ouvert
        if not self.is_market_open():
            print("💤 Le marché boursier est actuellement FERMÉ. Cycle ignoré.", flush=True)
            return

        # 2. Récupération du capital et calcul de l'exposition actuelle sur Alpaca
        account = self.trading_client.get_account()
        total_portfolio = float(account.equity)
        allocated_capital = total_portfolio * self.risk_per_trade       # 2% par trade
        max_exposure_capital = total_portfolio * self.max_capital_usage # 40% maximum du portefeuille

        open_positions = self.trading_client.get_all_positions()
        
        # 3. Gestion des sorties (Take-Profit / Stop-Loss) avant d'évaluer le capital disponible
        if self.manage_open_positions(open_positions) > 0:
            # Rafraîchit la liste des positions si une clôture a eu lieu
            open_positions = self.trading_client.get_all_positions()

        held_symbols = [pos.symbol for pos in open_positions]
        current_invested = sum(float(pos.market_value) for pos in open_positions)
        current_exposure_pct = (current_invested / total_portfolio) * 100 if total_portfolio > 0 else 0

        print(f"\n💰 Portefeuille total : {total_portfolio:.2f} $ | Allocation/trade (2%) : {allocated_capital:.2f} $")
        print(f"📊 Exposition actuelle : {current_invested:.2f} $ ({current_exposure_pct:.1f}% / 40.0% max) | Positions actives : {len(held_symbols)}")

        # 4. Vérification du plafond de 40% d'utilisation du capital
        if (current_invested + allocated_capital) > max_exposure_capital:
            print("🛑 Plafond de 40% d'utilisation du capital atteint. Aucun nouvel achat sur ce cycle.", flush=True)
            self.save_state()
            return

        candidates = []

        # 5. Scan des actions (en ignorant celles déjà détenues)
        print(f"\n🔍 Scan des {len(self.symbols)} actions de la watchlist...", flush=True)
        for symbol in self.symbols:
            if symbol in held_symbols:
                print(f"   • {symbol:10} | Déjà en portefeuille (ignoré)")
                continue

            try:
                df = self.get_historical_bars(symbol, calendar_days=90)
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

        # 6. Comité d'IA et Exécution de l'Ordre
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_choice = candidates[0]
            clean_symbol = best_choice['symbol']

            print(f"\n🧠 Soumission du dossier Swing {clean_symbol} au comité multi-IA...")

            decision, feedback = ask_dual_ai_decision(
                clean_symbol,
                best_choice['price'],
                best_choice['score'],
                best_choice['reason'],
                "NEUTRE_BOURSE",
                held_symbols,
                "Pas de leçon récente"
            )

            if decision == "VALIDÉ":
                qty_to_buy = round(allocated_capital / best_choice['price'], 2)
                print(f"🔥 Exécution de l'ordre d'achat Alpaca : {qty_to_buy} actions de {clean_symbol} (~{allocated_capital:.2f}$)")
                
                market_order_data = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=qty_to_buy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order_data=market_order_data)
                
                self.trade_history["trades"].append({
                    "date": datetime.now(timezone.utc).isoformat(),
                    "symbol": clean_symbol,
                    "qty": qty_to_buy,
                    "price": best_choice['price'],
                    "allocated": round(allocated_capital, 2)
                })
                
                send_telegram_alert(
                    f"📈 *ACHAT EXÉCUTÉ* : `{qty_to_buy}` actions de `{clean_symbol}` à `{best_choice['price']:.2f}$`\n"
                    f"• Montant : `{allocated_capital:.2f}$` (2%)\n"
                    f"• Exposition globale : `{(current_invested + allocated_capital)/total_portfolio*100:.1f}% / 40%`"
                )
            else:
                print(f"🛡️ [VETO IA] Ordre annulé : {feedback}")

        self.save_state()
