# core_stock_engine.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
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
        
        # Gestion du risque & Pyramiding
        self.tranche_pct = 0.02          # 2% par tranche
        self.max_per_stock_pct = 0.08    # 8% max par action (4 tranches)
        self.max_capital_usage = 0.40    # 40% max global
        
        self.take_profit_pct = 0.06      # Take-profit à +6%
        self.base_stop_loss_pct = -0.03  # Stop-loss initial à -3% (Palier 1)
        self.min_gain_to_pyramid = 0.01  # +1% de gain minimum pour renforcer
        
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    self.trade_history = json.load(f)
                    if "last_buy_dates" not in self.trade_history:
                        self.trade_history["last_buy_dates"] = {}
                    return
            except Exception:
                pass
        self.trade_history = {"last_run": None, "trades": [], "last_buy_dates": {}}

    def save_state(self):
        self.trade_history["last_run"] = datetime.now(timezone.utc).isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(self.trade_history, f, indent=4)

    def is_market_open(self):
        try:
            return self.trading_client.get_clock().is_open
        except Exception as e:
            print(f"⚠️ Impossible de vérifier l'état du marché: {e}")
            return False

    def get_historical_bars(self, symbol: str, calendar_days: int = 90) -> pd.DataFrame:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=calendar_days)

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX
        )

        df = self.data_client.get_stock_bars(request_params).df
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index().rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        return df

    def evaluate_stock_opportunity(self, df):
        if len(df) < 50:
            return 0, f"Données insuffisantes ({len(df)} bougies)"

        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

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

    def _cancel_open_orders_for_symbol(self, symbol: str):
        """Annule les ordres Stop/Trailing liés avant de clôturer manuellement une position."""
        orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        for order in orders:
            try:
                self.trading_client.cancel_order_by_id(order.id)
            except Exception:
                pass

    def manage_open_positions(self, open_positions, total_portfolio):
        """Gère les sorties avec Stop-Loss dynamique (Break-Even) et protection Anti-PDT."""
        if not open_positions:
            return 0

        print(f"\n📋 Suivi des {len(open_positions)} position(s) ouverte(s) :", flush=True)
        closed_count = 0
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for pos in open_positions:
            symbol = pos.symbol
            unrealized_plpc = float(pos.unrealized_plpc)
            unrealized_pl = float(pos.unrealized_pl)
            market_value = float(pos.market_value)
            weight_pct = (market_value / total_portfolio) * 100 if total_portfolio > 0 else 0
            tranches_est = max(1, round(weight_pct / (self.tranche_pct * 100)))

            # --- CORRECTION #1 : STOP-LOSS DYNAMIQUE (BREAK-EVEN) ---
            if tranches_est >= 3:
                dynamic_sl = 0.00   # Break-Even (0%) dès 6% ou 8% d'exposition
            elif tranches_est == 2:
                dynamic_sl = -0.01  # Resserré à -1% dès 4% d'exposition
            else:
                dynamic_sl = self.base_stop_loss_pct  # -3% sur la 1ère tranche (2%)

            print(f"   • {symbol:6} | Palier {tranches_est}/4 ({weight_pct:4.1f}%) | P&L: {unrealized_plpc*100:+6.2f}% | SL actif: {dynamic_sl*100:+.1f}%")

            # --- CORRECTION #3 : PROTECTION ANTI-DAY TRADING (PDT) ---
            last_buy_date = self.trade_history.get("last_buy_dates", {}).get(symbol)
            if last_buy_date == today_str and unrealized_plpc > -0.05:
                print(f"     ⏳ Position ouverte aujourd'hui ({today_str}) : maintien jusqu'à demain (Règle Swing / Anti-PDT).")
                continue

            if unrealized_plpc >= self.take_profit_pct:
                print(f"   🎯 TAKE-PROFIT atteint sur {symbol} ({unrealized_plpc*100:+.2f}%) -> Clôture !")
                self._cancel_open_orders_for_symbol(symbol)
                self.trading_client.close_position(symbol)
                send_telegram_alert(f"🎯 *TAKE-PROFIT* sur `{symbol}` : `{unrealized_plpc*100:+.2f}%` (`{unrealized_pl:+.2f}$`)")
                closed_count += 1

            elif unrealized_plpc <= dynamic_sl:
                sl_type = "BREAK-EVEN" if dynamic_sl == 0.0 else "STOP-LOSS"
                print(f"   🛑 {sl_type} atteint sur {symbol} ({unrealized_plpc*100:+.2f}%) -> Clôture !")
                self._cancel_open_orders_for_symbol(symbol)
                self.trading_client.close_position(symbol)
                send_telegram_alert(f"🛑 *{sl_type}* sur `{symbol}` (Palier {tranches_est}/4) : `{unrealized_plpc*100:+.2f}%` (`{unrealized_pl:+.2f}$`)")
                closed_count += 1

        return closed_count

    def run_cycle(self):
        print(f"[{datetime.now(timezone.utc).isoformat()}] Lancement du cycle Swing Trading Actions...", flush=True)

        if not self.is_market_open():
            print("💤 Le marché boursier est actuellement FERMÉ. Cycle ignoré.", flush=True)
            return

        account = self.trading_client.get_account()
        total_portfolio = float(account.equity)
        tranche_capital = total_portfolio * self.tranche_pct
        max_per_stock_capital = total_portfolio * self.max_per_stock_pct
        max_exposure_capital = total_portfolio * self.max_capital_usage

        open_positions = self.trading_client.get_all_positions()
        
        if self.manage_open_positions(open_positions, total_portfolio) > 0:
            open_positions = self.trading_client.get_all_positions()

        # --- CORRECTION #2 : VÉRIFICATION DES ORDRES D'ACHAT EN ATTENTE ---
        open_orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        pending_buy_symbols = {o.symbol for o in open_orders if o.side == OrderSide.BUY}

        positions_map = {pos.symbol: pos for pos in open_positions}
        current_invested = sum(float(pos.market_value) for pos in open_positions)
        current_exposure_pct = (current_invested / total_portfolio) * 100 if total_portfolio > 0 else 0

        print(f"\n💰 Portefeuille : {total_portfolio:.2f} $ | Tranche (2%) : {tranche_capital:.2f} $")
        print(f"📊 Exposition globale : {current_invested:.2f} $ ({current_exposure_pct:.1f}% / 40.0% max)")

        if (current_invested + tranche_capital) > max_exposure_capital:
            print("🛑 Plafond global de 40% atteint. Aucun nouvel achat.", flush=True)
            self.save_state()
            return

        candidates = []
        print(f"\n🔍 Scan des {len(self.symbols)} actions de la watchlist...", flush=True)
        for symbol in self.symbols:
            if symbol in pending_buy_symbols:
                print(f"   • {symbol:10} | Ordre d'achat déjà en attente chez Alpaca - Ignoré")
                continue

            existing_pos = positions_map.get(symbol)
            current_stock_value = float(existing_pos.market_value) if existing_pos else 0.0
            current_stock_pct = (current_stock_value / total_portfolio) * 100 if total_portfolio > 0 else 0.0

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
                    print(f"   • {symbol:10} | Prix: {current_price:8.2f}$ | Score: {score:3}/100 | {reason}")
                    if score >= 65:
                        candidates.append({
                            'symbol': symbol, 'price': current_price, 'score': score,
                            'reason': f"[Entrée 1/4 - 2%] {reason}", 'is_pyramid': False, 'tranche_num': 1
                        })
                else:
                    unrealized_plpc = float(existing_pos.unrealized_plpc)
                    next_tranche = min(4, round(current_stock_pct / 2.0) + 1)
                    print(f"   • {symbol:10} | Prix: {current_price:8.2f}$ | Score: {score:3}/100 | En cours ({current_stock_pct:.1f}%, P&L: {unrealized_plpc*100:+.1f}%)")

                    if score >= 80 and unrealized_plpc >= self.min_gain_to_pyramid:
                        candidates.append({
                            'symbol': symbol, 'price': current_price, 'score': score + 5,
                            'reason': f"[Renforcement {next_tranche}/4 -> {next_tranche*2}%] P&L {unrealized_plpc*100:+.1f}% | {reason}",
                            'is_pyramid': True, 'tranche_num': next_tranche
                        })
            except Exception as e:
                print(f"   ⚠️ Erreur d'analyse sur {symbol}: {e}")

        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_choice = candidates[0]
            clean_symbol = best_choice['symbol']
            action_label = f"RENFORCEMENT ({best_choice['tranche_num']}/4)" if best_choice['is_pyramid'] else "NOUVELLE ENTRÉE (1/4)"

            print(f"\n🧠 Soumission du dossier Swing {clean_symbol} [{action_label}] au comité multi-IA...")

            decision, feedback = ask_dual_ai_decision(
                clean_symbol, best_choice['price'], min(100, best_choice['score']),
                best_choice['reason'], "NEUTRE_BOURSE", list(positions_map.keys()),
                "Vérifier absence d'annonce de résultats (Earnings) imminente."
            )

            if decision == "VALIDÉ":
                qty_to_buy = round(tranche_capital / best_choice['price'], 2)
                print(f"🔥 Exécution {action_label} Alpaca : +{qty_to_buy} actions de {clean_symbol}")
                
                # 1. Ordre d'achat au marché
                market_order_data = MarketOrderRequest(
                    symbol=clean_symbol, qty=qty_to_buy, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order_data=market_order_data)

                # --- CORRECTION #4 : TRAILING STOP PERMANENT CHEZ ALPACA ---
                # Protège contre les krachs/gaps entre deux cycles du bot (sur actions entières)
                whole_qty = int(qty_to_buy)
                if whole_qty >= 1:
                    try:
                        trailing_stop = TrailingStopOrderRequest(
                            symbol=clean_symbol, qty=whole_qty, side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC, trail_percent=3.0
                        )
                        self.trading_client.submit_order(order_data=trailing_stop)
                    except Exception as e:
                        print(f"   ℹ️ Trailing Stop serveur non placé (géré par le bot) : {e}")
                
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                self.trade_history["last_buy_dates"][clean_symbol] = today_str
                self.trade_history["trades"].append({
                    "date": datetime.now(timezone.utc).isoformat(), "type": action_label,
                    "symbol": clean_symbol, "qty": qty_to_buy, "price": best_choice['price'],
                    "allocated": round(tranche_capital, 2)
                })
                
                send_telegram_alert(
                    f"📈 *{action_label}* : `+{qty_to_buy}` actions de `{clean_symbol}` à `{best_choice['price']:.2f}$`\n"
                    f"• Exposition sur `{clean_symbol}` : `{best_choice['tranche_num']*2}% / 8% max`\n"
                    f"• Exposition globale : `{(current_invested + tranche_capital)/total_portfolio*100:.1f}% / 40%`"
                )
            else:
                print(f"🛡️ [VETO IA] Ordre annulé : {feedback}")

        self.save_state()
