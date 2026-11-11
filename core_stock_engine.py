# core_stock_engine.py
import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dual_advisor import ask_dual_ai_decision  # Réutilise votre comité d'IA

STATE_FILE = "stock_bot_state.json"

class StockSwingEngine:
    def __init__(self, initial_capital=1000.0):
        # Initialisation Alpaca (Paper Trading)
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        # paper=True pour l'environnement de test
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        
        # Univers de recherche : Actions majeures US & Canadiennes (converties pour yfinance)
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
            'SHOP.TO', 'TD.TO', 'RY.TO', 'CNR.TO', 'BNS.TO'  # Actions TSX avec extension .TO
        ]
        self.load_state(initial_capital)

    def load_state(self, initial_capital):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.current_position = data.get('current_position', None)
                    self.entry_price = data.get('entry_price', None)
                    return
            except Exception:
                pass
        self.current_position = None
        self.entry_price = None

    def save_state(self):
        data = {
            'current_position': self.current_position,
            'entry_price': self.entry_price
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    def is_market_open(self):
        """Vérifie auprès de l'API si la bourse est ouverte aujourd'hui."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"⚠️ Impossible de vérifier l'état du marché: {e}")
            return False

    def evaluate_stock_opportunity(self, df):
        """
        Logique de Swing Trading sur bougies Journalières (1D).
        Exemple : Moyennes mobiles SMA20 / SMA50 et RSI(14).
        """
        if len(df) < 50:
            return 0, "Données insuffisantes"

        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # Calcul basique du RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        score = 0
        reasons = []

        # Tendance haussière (SMA20 > SMA50)
        if last_row['SMA20'] > last_row['SMA50']:
            score += 35
            reasons.append("Tendance haussière (SMA20 > SMA50)")

        # Rebond sur la SMA20
        if prev_row['Low'] <= prev_row['SMA20'] and last_row['Close'] > last_row['SMA20']:
            score += 35
            reasons.append("Rebond sur la SMA20")

        # RSI non suracheté (entre 40 et 65)
        if 40 <= last_row['RSI'] <= 65:
            score += 30
            reasons.append(f"RSI optimal ({last_row['RSI']:.1f})")

        return score, " | ".join(reasons) if reasons else "Pas de signal"

    def run_cycle(self):
        print(f"[{datetime.utcnow().isoformat()}] Lancement du cycle Swing Trading Actions...", flush=True)

        # 1. Vérification si le marché boursier est ouvert
        if not self.is_market_open():
            print("💤 Le marché boursier est actuellement FERMÉ. Cycle ignoré.", flush=True)
            return

        # 2. Récupération du capital disponible réels sur Alpaca
        account = self.trading_client.get_account()
        total_portfolio = float(account.equity)
        allocated_capital = total_portfolio * 0.02  # 2% max par position en Swing Trading

        print(f"💰 Capital total portefeuille : {total_portfolio:.2f} $ | Allocation max (2%) : {allocated_capital:.2f} $")

        # 3. Suivi de la position active (Gestion des Stop-Loss / Take-Profit de Swing)
        if self.current_position:
            # Code de suivi de la position boursière ouverte...
            pass

        candidates = []

        # 4. Scan des actions avec yfinance (Données 1D)
        print(f"\n🔍 Scan des {len(self.symbols)} actions de la watchlist...", flush=True)
        for symbol in self.symbols:
            try:
                # Récupère 60 jours de données journalières
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="60d", interval="1d")

                if df.empty:
                    continue

                current_price = df['Close'].iloc[-1]
                score, reason = self.evaluate_stock_opportunity(df)

                print(f"   • {symbol:10} | Prix: {current_price:8.2f}$ | Score: {score:3}/100 | {reason}")

                if score >= 65:  # Seuil plus exigeant pour le Swing
                    candidates.append({
                        'symbol': symbol,
                        'price': current_price,
                        'score': score,
                        'reason': reason
                    })
            except Exception as e:
                print(f"   ⚠️ Erreur d'analyse sur {symbol}: {e}")

        # 5. Comité d'IA et Exécution de l'Ordre
        if candidates and not self.current_position:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_choice = candidates[0]

            print(f"\n🧠 Soumission du dossier Swing {best_choice['symbol']} au comité multi-IA...")
            
            # Formate le symbole pour Alpaca si c'est une action US (ex: supprime .TO si besoin)
            clean_symbol = best_choice['symbol'].replace('.TO', '')

            decision, feedback = ask_dual_ai_decision(
                clean_symbol,
                best_choice['price'],
                best_choice['score'],
                best_choice['reason'],
                "NEUTRE_BOURSE",
                [],
                "Pas de leçon récente"
            )

            if decision == "VALIDÉ":
                # Calcul de la quantité d'actions (Fractional Shares supportées par Alpaca)
                qty_to_buy = round(allocated_capital / best_choice['price'], 2)

                print(f"🔥 Exécution de l'ordre d'achat Alpaca : {qty_to_buy} actions de {clean_symbol}")
                
                market_order_data = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=qty_to_buy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order_data=market_order_data)
                self.current_position = clean_symbol
                self.entry_price = best_choice['price']
                self.save_state()
            else:
                print(f"🛡️ [VETO IA] Ordre annulé : {feedback}")
