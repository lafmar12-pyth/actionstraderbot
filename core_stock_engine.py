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
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Les clés ALPACA_API_KEY ou ALPACA_SECRET_KEY sont manquantes.")

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
            'SHOP', 'TD', 'RY', 'CNI', 'BNS',
            'JPM', 'V', 'MA', 'UNH', 'HD', 'PG', 'COST', 'CRM', 'AVGO', 'NFLX', 'DIS', 'PEP'
        ]
        
        # Paramètres de gestion du risque & Pyramiding
        self.tranche_pct = 0.02          # 2% du capital par tranche d'achat
        self.max_per_stock_pct = 0.08    # 8% max sur une même action (soit 4 tranches de 2%)
        self.max_capital_usage = 0.40    # 40% maximum du portefeuille engagé au total
        
        self.take_profit_pct = 0.06      # Prise de profit à +6%
        self.stop_loss_pct = -0.03       # Stop-loss à -3%
        self.min_gain_to_pyramid = 0.01  # +1% de gain latent minimum requis pour renforcer
        
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

    def manage_open_positions(self, open_positions, total_portfolio):
        """Vérifie le Take-Profit (+6%) et le Stop-Loss (-3%) sur les positions ouvertes."""
        if not open_positions:
            return 0

        print(f"\n📋 Suivi des {len(open_positions)} position(s) ouverte(s) :", flush=True)
        closed_count = 0

        for pos in open_positions:
            symbol = pos.symbol
            unrealized_plpc = float(pos.unrealized_plpc)
            unrealized_pl = float(pos.unrealized_pl)
            market_value = float(pos.market_value)
            weight_pct = (market_value / total_portfolio) * 100 if total_portfolio > 0 else 0
            tranches_est = max(1, round(weight_pct / (self.tranche_pct * 100)))

            print(f"   • {symbol:6} | Poids: {weight_pct:4.1f}% (~{tranches_est}/4) | Valeur: {market_value:8.2f}$ | P&L: {unrealized_plpc*100:+6.2f}% ({unrealized_pl:+.2f}$)")

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

        # 2. Récupération du capital et calcul des limites
        account = self.trading_client.get_account()
        total_portfolio = float(account.equity)
        tranche_capital = total_portfolio * self.tranche_pct         # 2% par ordre
        max_per_stock_capital = total_portfolio * self.max_per_stock_pct # 8% max par action
        max_exposure_capital = total_portfolio * self.max_capital_usage  # 40% max global

        open_positions = self.trading_client.get_all_positions()
        
        # 3. Gestion des sorties (Take-Profit / Stop-Loss)
        if self.manage_open_positions(open_positions, total_portfolio) > 0:
            open_positions = self.trading_client.get_all_positions()

        # Indexation des positions détenues par symbole
        positions_map = {pos.symbol: pos for pos in open_positions}
        current_invested = sum(float(pos.market_value) for pos in open_positions)
        current_exposure_pct = (current_invested / total_portfolio) * 100 if total_portfolio > 0 else 0

        print(f"\n💰 Portefeuille : {total_portfolio:.2f} $ | Tranche (2%) : {tranche_capital:.2f} $ | Max/Action (8%) : {max_per_stock_capital:.2f} $")
        print(f"📊 Exposition globale : {current_invested:.2f} $ ({current_exposure_pct:.1f}% / 40.0% max)")

        # 4. Vérification du plafond global de 40%
        if (current_invested + tranche_capital) > max_exposure_capital:
            print("🛑 Plafond global de 40% d'utilisation du capital atteint. Aucun nouvel achat sur ce cycle.", flush=True)
            self.save_state()
            return

        candidates = []

        # 5. Scan des actions (Nouvelles entrées + Renforcements jusqu'à 8%)
        print(f"\n🔍 Scan des {len(self.symbols)} actions de la watchlist...", flush=True)
        for symbol in self.symbols:
            existing_pos = positions_map.get(symbol)
            current_stock_value = float(existing_pos.market_value) if existing_pos else 0.0
            current_stock_pct = (current_stock_value / total_portfolio) * 100 if total_portfolio > 0 else 0.0

            # Vérifie si l'action a déjà atteint le plafond de 8% (4 tranches de 2%)
            # Tolérance de 0.5% pour absorber la hausse du cours
            if (current_stock_value + tranche_capital) > (max_per_stock_capital * 1.05):
                print(f"   • {symbol:10} | Plafond 8% atteint ({current_stock_pct:.1f}%) - Ignoré")
                continue

            try:
                df = self.get_historical_bars(symbol, calendar_days=90)
                if df.empty:
                    continue

                current_price = float(df['Close'].iloc[-1])
                score, reason = self.evaluate_stock_opportunity(df)

                if not existing_pos:
                    # --- CAS A : NOUVELLE POSITION (Palier 1/4 - 2%) ---
                    print(f"   • {symbol:10} | Prix: {current_price:8.2f}$ | Score: {score:3}/100 | {reason}")
                    if score >= 65:
                        candidates.append({
                            'symbol': symbol,
                            'price': current_price,
                            'score': score,
                            'reason': f"[Entrée 1/4 - 2%] {reason}",
                            'is_pyramid': False,
                            'tranche_num': 1
                        })
                else:
                    # --- CAS B : RENFORCEMENT (Paliers 2/4, 3/4 ou 4/4 - jusqu'à 8%) ---
                    unrealized_plpc = float(existing_pos.unrealized_plpc)
                    next_tranche = min(4, round(current_stock_pct / 2.0) + 1)
                    print(f"   • {symbol:10} | Prix: {current_price:8.2f}$ | Score: {score:3}/100 | En cours ({current_stock_pct:.1f}%, P&L: {unrealized_plpc*100:+.1f}%)")

                    # Conditions strictes pour renforcer : Score >= 80 (signal fort) ET position déjà gagnante (>= +1%)
                    if score >= 80 and unrealized_plpc >= self.min_gain_to_pyramid:
                        candidates.append({
                            'symbol': symbol,
                            'price': current_price,
                            'score': score + 5,  # Légère priorité aux gagnants confirmés
                            'reason': f"[Renforcement {next_tranche}/4 -> {next_tranche*2}%] P&L actuel {unrealized_plpc*100:+.1f}% | {reason}",
                            'is_pyramid': True,
                            'tranche_num': next_tranche
                        })

            except Exception as e:
                print(f"   ⚠️ Erreur d'analyse sur {symbol}: {e}")

        # 6. Comité d'IA et Exécution de l'Ordre
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_choice = candidates[0]
            clean_symbol = best_choice['symbol']
            action_label = f"RENFORCEMENT ({best_choice['tranche_num']}/4)" if best_choice['is_pyramid'] else "NOUVELLE ENTRÉE (1/4)"

            print(f"\n🧠 Soumission du dossier Swing {clean_symbol} [{action_label}] au comité multi-IA...")

            decision, feedback = ask_dual_ai_decision(
                clean_symbol,
                best_choice['price'],
                min(100, best_choice['score']),
                best_choice['reason'],
                "NEUTRE_BOURSE",
                list(positions_map.keys()),
                "Pyramiding actif : renforcement autorisé uniquement sur positions gagnantes"
            )

            if decision == "VALIDÉ":
                qty_to_buy = round(tranche_capital / best_choice['price'], 2)
                print(f"🔥 Exécution {action_label} Alpaca : +{qty_to_buy} actions de {clean_symbol} (~{tranche_capital:.2f}$)")
                
                market_order_data = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=qty_to_buy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order_data=market_order_data)
                
                self.trade_history["trades"].append({
                    "date": datetime.now(timezone.utc).isoformat(),
                    "type": action_label,
                    "symbol": clean_symbol,
                    "qty": qty_to_buy,
                    "price": best_choice['price'],
                    "allocated": round(tranche_capital, 2)
                })
                
                new_total_stock_pct = best_choice['tranche_num'] * 2
                send_telegram_alert(
                    f"📈 *{action_label}* : `+{qty_to_buy}` actions de `{clean_symbol}` à `{best_choice['price']:.2f}$`\n"
                    f"• Tranche ajoutée : `{tranche_capital:.2f}$` (+2%)\n"
                    f"• Exposition sur `{clean_symbol}` : `{new_total_stock_pct}% / 8% max`\n"
                    f"• Exposition globale portefeuille : `{(current_invested + tranche_capital)/total_portfolio*100:.1f}% / 40%`"
                )
            else:
                print(f"🛡️ [VETO IA] Ordre annulé : {feedback}")

        self.save_state()
