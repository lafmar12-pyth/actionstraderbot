# core_stock_engine.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dual_advisor import ask_dual_ai_decision
from market_memory import MarketMemoryManager
from stock_learning_library import StockLearningLibrary
from telegram_notifier import send_telegram_alert

STATE_FILE = "stock_bot_state.json"

COOLDOWN_DAYS_AFTER_LOSS = 3
MAX_LOSSES_IN_5_DAYS = 2
MIN_NEW_ENTRY_SCORE = 72
MIN_PYRAMID_SCORE = 82
MAX_POSITIONS_PER_SECTOR = 2
CUSHION_SPLIT_RATIO = 0.50  # 50% des bénéfices mis de côté dans le coussin 50/50

SECTOR_MAP = {
    # Tech & Croissance
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "AMZN": "TECH",
    "NVDA": "TECH", "META": "TECH", "TSLA": "TECH", "AMD": "TECH",
    "AVGO": "TECH", "NFLX": "TECH", "SHOP": "TECH", "CRM": "TECH",
    # Banques & Finance
    "TD": "FINANCE_CA", "RY": "FINANCE_CA", "BNS": "FINANCE_CA",
    "JPM": "FINANCE_US", "V": "FINANCE_US", "MA": "FINANCE_US",
    # Défensives, Santé, Consommation & Cycliques
    "UNH": "SANTE", "LLY": "SANTE",
    "PG": "CONSO_DEF", "KO": "CONSO_DEF", "PEP": "CONSO_DEF", "WMT": "CONSO_DEF", "COST": "CONSO_DEF",
    "HD": "CONSO_DISC", "DIS": "CONSO_DISC",
    "CNI": "INDUST_TRANS", "CAT": "INDUST_TRANS", "XOM": "ENERGIE",
}


class StockSwingEngine:
    def __init__(self, initial_capital=1000.0):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Les clés ALPACA_API_KEY ou ALPACA_SECRET_KEY sont manquantes.")

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.memory = MarketMemoryManager()
        self.library = StockLearningLibrary()

        # Univers de 30 actions (US & Canada) pour l'observation et le swing trading
        self.symbols = [
            # Tech & Croissance (10)
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "AVGO", "NFLX",
            # Canada / Finance / Rail / Commerce (6)
            "SHOP", "TD", "RY", "CNI", "BNS", "CRM",
            # Finance US, Santé, Consommation & Défensives (14)
            "JPM", "V", "MA", "UNH", "HD", "PG", "COST", "DIS", "PEP", "KO", "WMT", "LLY", "CAT", "XOM",
        ]

        # Gestion du risque & Pyramiding
        self.tranche_pct = 0.02          # 2% nominal par tranche
        self.max_per_stock_pct = 0.08    # 8% max par action (4 tranches)
        self.max_capital_usage = 0.40    # 40% max global

        self.take_profit_pct = 0.06      # Take-profit à +6%
        self.base_stop_loss_pct = -0.03  # Stop-loss initial à -3% (Palier 1)
        self.min_gain_to_pyramid = 0.01  # +1% de gain minimum pour renforcer

        self.load_state()

    def load_state(self):
        default_state = {
            "last_run": None,
            "trades": [],
            "last_buy_dates": {},
            "cushion_reserve": 0.0,
            "cushion_history": [],
            "market_journal": {
                "signals": [],
                "stats": {"total_5d": 0, "wins_5d": 0},
            },
        }
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.trade_history = loaded
                    self.trade_history.setdefault("trades", [])
                    self.trade_history.setdefault("last_buy_dates", {})
                    self.trade_history.setdefault("cushion_reserve", 0.0)
                    self.trade_history.setdefault("cushion_history", [])
                    journal = self.trade_history.setdefault(
                        "market_journal", {"signals": [], "stats": {"total_5d": 0, "wins_5d": 0}}
                    )
                    journal.setdefault("signals", [])
                    journal.setdefault("stats", {"total_5d": 0, "wins_5d": 0})
                    journal["stats"].setdefault("total_5d", 0)
                    journal["stats"].setdefault("wins_5d", 0)
                    return
            except Exception:
                pass
        self.trade_history = default_state

    def save_state(self):
        self.trade_history["last_run"] = datetime.now(timezone.utc).isoformat()
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.trade_history, f, indent=4, ensure_ascii=False)

    def get_dual_accounts_summary(self, total_portfolio: float, current_invested: float) -> dict:
        """Retourne les soldes du compte de trading actif et du coussin 50/50 mis à l'écart."""
        cushion_reserve = max(0.0, float(self.trade_history.get("cushion_reserve", 0.0)))
        active_capital = max(0.0, total_portfolio - cushion_reserve)
        available_to_invest = max(0.0, (active_capital * self.max_capital_usage) - current_invested)
        return {
            "total_portfolio": round(total_portfolio, 2),
            "cushion_reserve": round(cushion_reserve, 2),
            "active_capital": round(active_capital, 2),
            "current_invested": round(current_invested, 2),
            "available_to_invest": round(available_to_invest, 2),
        }

    def print_dual_accounts_summary(self, total_portfolio: float, current_invested: float):
        """Affiche en fin de cycle le compte de trading restant et le coussin 50/50."""
        summary = self.get_dual_accounts_summary(total_portfolio, current_invested)
        print("\n🏦 RÉCAPITULATIF DES 2 COMPTES (FIN DE CYCLE) :", flush=True)
        print(
            f"   • Compte Trading (capital actif) : {summary['active_capital']:.2f} $ "
            f"(Investi : {summary['current_invested']:.2f} $ | Marge restante sous 40% : {summary['available_to_invest']:.2f} $)",
            flush=True,
        )
        print(
            f"   • Coussin 50/50 (réserve à l'écart) : {summary['cushion_reserve']:.2f} $ "
            f"(Équité totale Alpaca : {summary['total_portfolio']:.2f} $)",
            flush=True,
        )

    def is_market_open(self):
        try:
            return bool(self.trading_client.get_clock().is_open)
        except Exception as e:
            print(f"⚠️ Impossible de vérifier l'état du marché: {e}", flush=True)
            return False

    def get_historical_bars(self, symbol: str, calendar_days: int = 120) -> pd.DataFrame:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=calendar_days)

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX,
        )

        df = self.data_client.get_stock_bars(request_params).df
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index().rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
        })
        return df

    def get_market_macro_trend(self) -> str:
        """Évalue la tendance générale du marché via l'ETF S&P 500 (SPY)."""
        try:
            spy_df = self.get_historical_bars("SPY", calendar_days=45)
            if len(spy_df) >= 20:
                spy_sma20 = float(spy_df["Close"].rolling(window=20).mean().iloc[-1])
                spy_close = float(spy_df["Close"].iloc[-1])
                diff_pct = ((spy_close - spy_sma20) / spy_sma20) * 100.0
                if spy_close > spy_sma20:
                    return f"HAUSSIER (SPY à {spy_close:.2f}$, {diff_pct:+.2f}% au-dessus de SMA20)"
                return f"BAISSIER / PRUDENCE (SPY à {spy_close:.2f}$, {diff_pct:+.2f}% sous SMA20)"
        except Exception as e:
            print(f"⚠️ Impossible de récupérer la tendance SPY : {e}", flush=True)
        return "NEUTRE_BOURSE"

    def _candle_quality_penalty(self, df: pd.DataFrame) -> tuple[int, str]:
        """Pénalise une forte mèche haute de rejet vendeur sur la dernière séance."""
        if df is None or len(df) < 1:
            return 0, ""
        last = df.iloc[-1]
        high = float(last["High"])
        low = float(last["Low"])
        close_p = float(last["Close"])
        open_p = float(last["Open"])
        total_range = high - low
        if total_range <= 0:
            return 0, ""

        upper_wick = high - max(open_p, close_p)
        upper_wick_ratio = upper_wick / total_range
        if upper_wick_ratio > 0.45 and close_p < open_p:
            return -10, f"Rejet en séance (mèche haute {upper_wick_ratio*100:.0f}%)"
        return 0, ""

    def evaluate_stock_opportunity(self, df):
        """Évalue la structure de tendance, le rebond/momentum, le volume relatif et la qualité de bougie."""
        if len(df) < 55:
            return 0, "Données insuffisantes (<55 séances)", {"above_sma20": False, "vol_ratio": 1.0}

        df = df.copy()
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["VOL20"] = df["Volume"].rolling(window=20).mean()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss.replace(0.0, pd.NA)
        df["RSI"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        sma50_prev = float(df["SMA50"].iloc[-6]) if len(df) >= 6 and pd.notna(df["SMA50"].iloc[-6]) else float(last_row["SMA50"])

        if pd.isna(last_row["SMA20"]) or pd.isna(last_row["SMA50"]) or float(last_row["SMA50"]) <= 0:
            return 0, "Indicateurs indisponibles", {"above_sma20": False, "vol_ratio": 1.0}

        close_now = float(last_row["Close"])
        close_prev = float(prev_row["Close"])
        sma20 = float(last_row["SMA20"])
        sma50 = float(last_row["SMA50"])
        rsi = float(last_row["RSI"])

        avg_vol = float(last_row["VOL20"]) if pd.notna(last_row["VOL20"]) else 0.0
        vol_ratio = float(last_row["Volume"] / avg_vol) if avg_vol > 0 else 1.0
        sma50_slope_pct = ((sma50 - sma50_prev) / sma50_prev) * 100.0 if sma50_prev > 0 else 0.0

        not_dumping = close_now >= close_prev * 0.985
        healthy_volume = vol_ratio >= 0.80

        score = 0
        reasons = []

        if sma20 > sma50 and sma50_slope_pct > -0.30:
            score += 35
            reasons.append(f"Structure haussière (SMA20 > SMA50, pente50 {sma50_slope_pct:+.2f}%)")

        if float(prev_row["Low"]) <= float(prev_row["SMA20"]) * 1.005 and close_now > sma20 and not_dumping:
            score += 35
            reasons.append("Rebond propre sur SMA20")
        elif close_now > sma20 and not_dumping:
            score += 20
            reasons.append("Cours soutenu au-dessus de SMA20")

        if 42.0 <= rsi <= 65.0:
            score += 20
            reasons.append(f"RSI sain ({rsi:.1f})")

        if healthy_volume:
            score += 10
            reasons.append(f"Volume confirmé ({vol_ratio:.1f}x)")
        else:
            reasons.append(f"Volume faible ({vol_ratio:.1f}x)")

        if not not_dumping:
            score = max(0, score - 25)
            reasons.append("Repli journalier marqué")

        wick_penalty, wick_reason = self._candle_quality_penalty(df)
        if wick_penalty != 0:
            score = max(0, score + wick_penalty)
            reasons.append(wick_reason)

        metrics = {
            "above_sma20": bool(close_now > sma20),
            "vol_ratio": vol_ratio,
        }

        return int(min(100, score)), " | ".join(reasons) if reasons else "Pas de signal", metrics

    def _check_early_risk_exit(self, symbol: str, unrealized_plpc: float, regime_tag: str) -> str | None:
        """Détecte une dégradation technique ou macro sur une position ouverte avant le stop-loss de base."""
        if unrealized_plpc >= 0:
            return None
        try:
            df = self.get_historical_bars(symbol, calendar_days=60)
            if len(df) < 22:
                return None
            sma20 = float(df["Close"].rolling(window=20).mean().iloc[-1])
            avg_vol20 = float(df["Volume"].rolling(window=20).mean().iloc[-1])
            last = df.iloc[-1]
            prev = df.iloc[-2]
            close_now = float(last["Close"])
            close_prev = float(prev["Close"])
            vol_ratio = float(last["Volume"] / avg_vol20) if avg_vol20 > 0 else 1.0
            _, wick_reason = self._candle_quality_penalty(df)

            # 1. Cassure franche de SMA20 sur volume vendeur marqué alors que la position est en perte
            if close_now < sma20 * 0.992 and vol_ratio >= 1.20 and unrealized_plpc <= -0.008:
                return f"Cassure SMA20 sous volume {vol_ratio:.1f}x"

            # 2. Repli marqué + rejet vendeur sur un titre déjà en perte sous régime baissier SPY
            if regime_tag == "BAISSIER" and close_now < close_prev * 0.985 and unrealized_plpc <= -0.015:
                extra = f" + {wick_reason}" if wick_reason else ""
                return f"Repli baissier SPY + baisse titre{extra}"
        except Exception:
            return None
        return None

    def compute_volatility_adjusted_tranche(self, df: pd.DataFrame, active_capital: float):
        """Module la tranche nominale de 2% (sur le capital actif) selon l'ADR% sur 14 jours."""
        base_pct = self.tranche_pct
        if df is None or len(df) < 14:
            return active_capital * base_pct, base_pct * 100, 0.0

        recent = df.tail(14)
        valid = recent[recent["Close"] > 0]
        if valid.empty:
            return active_capital * base_pct, base_pct * 100, 0.0

        adr_pct = float(((valid["High"] - valid["Low"]) / valid["Close"]).mean())
        if adr_pct > 0.035:
            multiplier = 0.75
        elif adr_pct < 0.015:
            multiplier = 1.25
        else:
            multiplier = 1.00

        adj_pct = base_pct * multiplier
        return active_capital * adj_pct, adj_pct * 100, adr_pct * 100

    def _parse_iso_utc(self, iso_str):
        if not iso_str:
            return None
        try:
            dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _record_closed_trade(self, symbol: str, reason_tag: str, plpc: float, pl_dollars: float) -> float:
        """Enregistre la clôture et prélève 50% des bénéfices nets dans le coussin 50/50."""
        now_iso = datetime.now(timezone.utc).isoformat()
        cushion_added = 0.0

        if pl_dollars > 0:
            cushion_added = round(pl_dollars * CUSHION_SPLIT_RATIO, 2)
            current_cushion = max(0.0, float(self.trade_history.get("cushion_reserve", 0.0)))
            new_cushion = round(current_cushion + cushion_added, 2)
            self.trade_history["cushion_reserve"] = new_cushion
            self.trade_history.setdefault("cushion_history", []).append({
                "date": now_iso,
                "symbol": symbol,
                "reason": reason_tag,
                "trade_profit": round(pl_dollars, 2),
                "cushion_added": cushion_added,
                "cushion_total": new_cushion,
            })

        self.trade_history.setdefault("trades", []).append({
            "date": now_iso,
            "type": "CLÔTURE",
            "reason": reason_tag,
            "symbol": symbol,
            "realized_plpc": round(plpc * 100, 2),
            "realized_pl": round(pl_dollars, 2),
            "cushion_added": cushion_added,
        })
        self.library.record_closed_trade(symbol, plpc, pl_dollars, reason_tag=reason_tag)
        tag = "gagnante" if plpc > 0 else "perdante"
        cushion_note = f" (Coussin 50/50 +{cushion_added:.2f}$)" if cushion_added > 0 else ""
        self.memory.add_lesson(
            f"Clôture {tag} ({reason_tag}) à {plpc*100:+.2f}% ({pl_dollars:+.2f}$){cushion_note}.",
            symbol=symbol,
            pnl_pct=plpc * 100.0,
        )
        return cushion_added

    def compute_expectancy_stats(self) -> dict:
        """Calcule le win rate, le gain/perte moyen et l'espérance (%) sur les clôtures enregistrées."""
        closed = [
            t for t in self.trade_history.get("trades", [])
            if t.get("type") == "CLÔTURE" and isinstance(t.get("realized_plpc"), (int, float))
        ]
        n = len(closed)
        if n == 0:
            return {"count": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0}

        wins = [t["realized_plpc"] for t in closed if t["realized_plpc"] > 0]
        losses = [abs(t["realized_plpc"]) for t in closed if t["realized_plpc"] <= 0]

        p_win = len(wins) / n
        p_loss = len(losses) / n
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = (p_win * avg_win) - (p_loss * avg_loss)

        return {
            "count": n,
            "win_rate": round(p_win * 100, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
        }

    def check_sector_concentration(self, symbol: str, positions_map: dict, is_pyramid: bool) -> tuple[bool, str]:
        """Limite l'exposition simultanée à un même secteur pour éviter les paris corrélés."""
        sector = SECTOR_MAP.get(symbol, "AUTRE")
        same_sector_count = 0
        for held_sym in positions_map.keys():
            if held_sym == symbol and is_pyramid:
                continue
            if SECTOR_MAP.get(held_sym, "AUTRE") == sector:
                same_sector_count += 1

        if same_sector_count >= MAX_POSITIONS_PER_SECTOR:
            return False, f"Concentration sectorielle sur {sector} ({same_sector_count}/{MAX_POSITIONS_PER_SECTOR} titres déjà ouverts)."
        return True, "Secteur sain"

    def can_open_stock_trade(self, candidate: dict, now_utc: datetime, exp_stats: dict, positions_map: dict) -> tuple[bool, str]:
        """Garde-fous Trader Pro adaptés au swing trading actions."""
        symbol = candidate["symbol"]
        is_pyramid = bool(candidate.get("is_pyramid"))
        score = int(candidate["score"])

        sector_ok, sector_msg = self.check_sector_concentration(symbol, positions_map, is_pyramid)
        if not sector_ok:
            return False, sector_msg

        closed_trades = [
            t for t in self.trade_history.get("trades", [])
            if t.get("type") == "CLÔTURE" and isinstance(t.get("realized_plpc"), (int, float))
        ]

        # 1. Pause défensive sur les nouvelles entrées après >= 2 pertes en 5 jours
        if not is_pyramid:
            cutoff_5d = now_utc - timedelta(days=5)
            recent_losses_5d = 0
            for t in closed_trades:
                dt = self._parse_iso_utc(t.get("date"))
                if float(t.get("realized_plpc", 0.0)) < 0 and dt and dt >= cutoff_5d:
                    recent_losses_5d += 1
            if recent_losses_5d >= MAX_LOSSES_IN_5_DAYS:
                return False, f"Pause nouvelles entrées : {recent_losses_5d} clôture(s) perdante(s) en 5j."

        # 2. Cooldown sur le titre après une clôture perdante
        cooldown_cutoff = now_utc - timedelta(days=COOLDOWN_DAYS_AFTER_LOSS)
        for t in reversed(closed_trades):
            if t.get("symbol") != symbol:
                continue
            if float(t.get("realized_plpc", 0.0)) < 0:
                dt = self._parse_iso_utc(t.get("date"))
                if dt and dt >= cooldown_cutoff:
                    days_left = max(0.1, (dt + timedelta(days=COOLDOWN_DAYS_AFTER_LOSS) - now_utc).total_seconds() / 86400.0)
                    return False, f"Cooldown actif sur {symbol} après perte ({t.get('realized_plpc'):+.2f}%) — ~{days_left:.1f}j restants."
            break

        # 3. Cohérence avec l'espérance historique réalisée
        if not is_pyramid and exp_stats.get("count", 0) >= 4 and exp_stats.get("expectancy", 0.0) < -1.0 and score < 85:
            return False, f"Espérance historique négative ({exp_stats['expectancy']:+.2f}%) et score {score} < 85."

        return True, "Validé"

    def _record_and_evaluate_observations(self, snapshots: list, today_str: str):
        """Enregistre les signaux >= seuil sur les 30 actions et évalue ceux d'au moins 7 jours calendaires (~5 séances)."""
        journal = self.trade_history.setdefault("market_journal", {"signals": [], "stats": {"total_5d": 0, "wins_5d": 0}})
        signals = journal.setdefault("signals", [])
        stats = journal.setdefault("stats", {"total_5d": 0, "wins_5d": 0})
        price_map = {item["symbol"]: item["price"] for item in snapshots}
        today_date = datetime.strptime(today_str, "%Y-%m-%d").date()

        for obs in signals:
            if not obs.get("evaluated_5d") and obs["symbol"] in price_map:
                try:
                    obs_date = datetime.strptime(obs["date"][:10], "%Y-%m-%d").date()
                except ValueError:
                    continue
                if (today_date - obs_date).days >= 7 and obs["price"] > 0:
                    exit_ref = price_map[obs["symbol"]]
                    ret_pct = ((exit_ref - obs["price"]) / obs["price"]) * 100.0
                    obs["evaluated_5d"] = True
                    obs["return_5d_pct"] = round(ret_pct, 2)
                    stats["total_5d"] += 1
                    if ret_pct > 0:
                        stats["wins_5d"] += 1

        for item in snapshots:
            if item["score"] >= MIN_NEW_ENTRY_SCORE:
                already_logged_today = any(
                    s["symbol"] == item["symbol"] and s["date"][:10] == today_str for s in signals
                )
                if not already_logged_today:
                    signals.append({
                        "date": today_str,
                        "symbol": item["symbol"],
                        "price": round(item["price"], 2),
                        "score": item["score"],
                        "regime": item["regime"],
                        "evaluated_5d": False,
                    })

        journal["signals"] = signals[-100:]
        self.library.sync_from_journal(signals)

    def _build_market_memory_summary(self, snapshots: list, target_symbol: str | None = None, regime_tag: str = "NEUTRE") -> str:
        """Construit un résumé statistique sobre des 30 actions, de la mémoire et de la bibliothèque pour le comité IA."""
        evaluated_count = sum(1 for s in snapshots if s.get("analyzed"))
        above_sma20_count = sum(1 for s in snapshots if s.get("above_sma20"))
        breadth_pct = (above_sma20_count / evaluated_count * 100.0) if evaluated_count > 0 else 0.0

        stats = self.trade_history.get("market_journal", {}).get("stats", {})
        total_5d = stats.get("total_5d", 0)
        wins_5d = stats.get("wins_5d", 0)

        if total_5d >= 10:
            win_rate = (wins_5d / total_5d) * 100.0
            perf_str = f"Taux rebonds positifs ~5j: {win_rate:.1f}% ({wins_5d}/{total_5d})"
        else:
            perf_str = f"Historique signaux ~5j en cours ({total_5d}/10 évalués)"

        exp_stats = self.compute_expectancy_stats()
        exp_str = (
            f"Espérance trades clôturés ({exp_stats['count']}): {exp_stats['expectancy']:+.2f}% "
            f"(Win {exp_stats['win_rate']:.1f}%, Gain moy. +{exp_stats['avg_win']:.2f}%, Perte moy. -{exp_stats['avg_loss']:.2f}%)"
            if exp_stats["count"] > 0 else "Espérance trades clôturés: aucun trade fermé"
        )

        lib_txt = self.library.get_summary_for_committee(target_symbol, regime_tag) if target_symbol else ""
        mem_txt = self.memory.get_recent_lessons_summary(limit=3)

        extra = " | ".join(p for p in (lib_txt, mem_txt) if p)
        return (
            f"Largeur univers ({above_sma20_count}/{evaluated_count} titres > SMA20: {breadth_pct:.1f}%) | "
            f"{perf_str} | {exp_str}"
            f"{' | ' + extra if extra else ''}. "
            f"Vérifier corrélation sectorielle, volume relatif et absence d'Earnings sous 48h."
        )

    def _cancel_open_orders_for_symbol(self, symbol: str):
        """Annule les ordres Stop/Trailing liés avant de clôturer ou d'ajuster une position."""
        orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        for order in orders:
            try:
                self.trading_client.cancel_order_by_id(order.id)
            except Exception:
                pass

    def manage_open_positions(self, open_positions, active_capital, regime_tag="NEUTRE"):
        """Gère les sorties avec Stop-Loss dynamique (Break-Even), sortie anticipée et protection Anti-PDT."""
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
            weight_pct = (market_value / active_capital) * 100.0 if active_capital > 0 else 0.0
            tranches_est = max(1, round(weight_pct / (self.tranche_pct * 100.0)))

            if tranches_est >= 3:
                dynamic_sl = 0.00
            elif tranches_est == 2:
                dynamic_sl = -0.01
            else:
                dynamic_sl = self.base_stop_loss_pct

            print(
                f"   • {symbol:6} | Palier {tranches_est}/4 ({weight_pct:4.1f}%) | "
                f"P&L: {unrealized_plpc*100:+6.2f}% | SL actif: {dynamic_sl*100:+.1f}%",
                flush=True,
            )

            last_buy_date = self.trade_history.get("last_buy_dates", {}).get(symbol)
            if last_buy_date == today_str and unrealized_plpc > -0.05:
                print(f"     ⏳ Position ouverte aujourd'hui ({today_str}) : maintien jusqu'à demain (Règle Swing / Anti-PDT).", flush=True)
                continue

            early_reason = self._check_early_risk_exit(symbol, unrealized_plpc, regime_tag)

            if unrealized_plpc >= self.take_profit_pct:
                print(f"   🎯 TAKE-PROFIT atteint sur {symbol} ({unrealized_plpc*100:+.2f}%) -> Clôture !", flush=True)
                self._cancel_open_orders_for_symbol(symbol)
                self.trading_client.close_position(symbol)
                self.trade_history.get("last_buy_dates", {}).pop(symbol, None)
                cushion_added = self._record_closed_trade(symbol, "TAKE-PROFIT", unrealized_plpc, unrealized_pl)
                cushion_msg = f" | Coussin 50/50: `+{cushion_added:.2f}$`" if cushion_added > 0 else ""
                send_telegram_alert(
                    f"🎯 *TAKE-PROFIT* sur `{symbol}` : `{unrealized_plpc*100:+.2f}%` (`{unrealized_pl:+.2f}$`){cushion_msg}"
                )
                closed_count += 1

            elif unrealized_plpc <= dynamic_sl:
                sl_type = "BREAK-EVEN" if dynamic_sl == 0.0 else "STOP-LOSS"
                print(f"   🛑 {sl_type} atteint sur {symbol} ({unrealized_plpc*100:+.2f}%) -> Clôture !", flush=True)
                self._cancel_open_orders_for_symbol(symbol)
                self.trading_client.close_position(symbol)
                self.trade_history.get("last_buy_dates", {}).pop(symbol, None)
                cushion_added = self._record_closed_trade(symbol, sl_type, unrealized_plpc, unrealized_pl)
                cushion_msg = f" | Coussin 50/50: `+{cushion_added:.2f}$`" if cushion_added > 0 else ""
                send_telegram_alert(
                    f"🛑 *{sl_type}* sur `{symbol}` (Palier {tranches_est}/4) : `{unrealized_plpc*100:+.2f}%` (`{unrealized_pl:+.2f}$`){cushion_msg}"
                )
                closed_count += 1

            elif early_reason:
                print(f"   ⚠️ SORTIE ANTICIPÉE sur {symbol} ({unrealized_plpc*100:+.2f}%) : {early_reason}", flush=True)
                self._cancel_open_orders_for_symbol(symbol)
                self.trading_client.close_position(symbol)
                self.trade_history.get("last_buy_dates", {}).pop(symbol, None)
                cushion_added = self._record_closed_trade(symbol, "SORTIE-ANTICIPÉE", unrealized_plpc, unrealized_pl)
                cushion_msg = f" | Coussin 50/50: `+{cushion_added:.2f}$`" if cushion_added > 0 else ""
                send_telegram_alert(
                    f"⚠️ *SORTIE ANTICIPÉE* sur `{symbol}` : `{unrealized_plpc*100:+.2f}%` (`{unrealized_pl:+.2f}$`) — {early_reason}{cushion_msg}"
                )
                closed_count += 1

        return closed_count

    def run_cycle(self):
        now_utc = datetime.now(timezone.utc)
        print(f"[{now_utc.isoformat()}] Lancement du cycle Swing Trading Actions...", flush=True)

        if not self.is_market_open():
            print("💤 Le marché boursier est actuellement FERMÉ. Cycle ignoré.", flush=True)
            return

        account = self.trading_client.get_account()
        total_portfolio = float(account.equity)
        cushion_reserve = max(0.0, float(self.trade_history.get("cushion_reserve", 0.0)))
        active_capital = max(0.0, total_portfolio - cushion_reserve)

        nominal_tranche = active_capital * self.tranche_pct
        max_per_stock_capital = active_capital * self.max_per_stock_pct
        max_exposure_capital = active_capital * self.max_capital_usage

        market_trend = self.get_market_macro_trend()
        regime_tag = "HAUSSIER" if "HAUSSIER" in market_trend else ("BAISSIER" if "BAISSIER" in market_trend else "NEUTRE")
        today_str = now_utc.strftime("%Y-%m-%d")

        open_positions = self.trading_client.get_all_positions()
        if self.manage_open_positions(open_positions, active_capital, regime_tag=regime_tag) > 0:
            open_positions = self.trading_client.get_all_positions()
            cushion_reserve = max(0.0, float(self.trade_history.get("cushion_reserve", 0.0)))
            active_capital = max(0.0, total_portfolio - cushion_reserve)
            nominal_tranche = active_capital * self.tranche_pct
            max_per_stock_capital = active_capital * self.max_per_stock_pct
            max_exposure_capital = active_capital * self.max_capital_usage

        open_orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        pending_buy_symbols = {o.symbol for o in open_orders if o.side == OrderSide.BUY}

        positions_map = {pos.symbol: pos for pos in open_positions}
        current_invested = sum(float(pos.market_value) for pos in open_positions)
        current_exposure_pct = (current_invested / active_capital) * 100.0 if active_capital > 0 else 0.0

        exp_stats = self.compute_expectancy_stats()
        print(
            f"\n💰 Compte Trading (actif) : {active_capital:.2f} $ | Coussin 50/50 : {cushion_reserve:.2f} $ "
            f"(Équité Alpaca : {total_portfolio:.2f} $)",
            flush=True,
        )
        print(f"💵 Tranche nominale (2% actif) : {nominal_tranche:.2f} $", flush=True)
        print(f"📊 Exposition globale : {current_invested:.2f} $ ({current_exposure_pct:.1f}% / 40.0% max)", flush=True)
        print(f"🌍 Météo Marché : {market_trend}", flush=True)
        if exp_stats["count"] > 0:
            print(
                f"📈 Espérance réelle ({exp_stats['count']} clôtures) : {exp_stats['expectancy']:+.2f}% "
                f"| Win rate : {exp_stats['win_rate']:.1f}% | Gain moy. : +{exp_stats['avg_win']:.2f}% | Perte moy. : -{exp_stats['avg_loss']:.2f}%",
                flush=True,
            )

        candidates = []
        snapshots = []
        print(f"\n🔍 Scan des {len(self.symbols)} actions de la watchlist...", flush=True)

        for symbol in self.symbols:
            try:
                df = self.get_historical_bars(symbol, calendar_days=120)
                if df.empty:
                    continue

                current_price = float(df["Close"].iloc[-1])
                score, reason, metrics = self.evaluate_stock_opportunity(df)
                tranche_capital, adj_pct, adr_pct = self.compute_volatility_adjusted_tranche(df, active_capital)

                snapshots.append({
                    "symbol": symbol,
                    "price": current_price,
                    "score": score,
                    "regime": regime_tag,
                    "analyzed": len(df) >= 55,
                    "above_sma20": metrics["above_sma20"],
                })

                if symbol in pending_buy_symbols:
                    print(f"   • {symbol:10} | Ordre d'achat déjà en attente chez Alpaca - Ignoré", flush=True)
                    continue

                existing_pos = positions_map.get(symbol)
                current_stock_value = float(existing_pos.market_value) if existing_pos else 0.0
                current_stock_pct = (current_stock_value / active_capital) * 100.0 if active_capital > 0 else 0.0

                if not existing_pos:
                    print(f"   • {symbol:10} | Prix: {current_price:8.2f}$ | ADR14: {adr_pct:4.1f}% ({adj_pct:4.1f}%) | Score: {score:3}/100 | {reason}", flush=True)
                    if score >= MIN_NEW_ENTRY_SCORE and (current_stock_value + tranche_capital) <= (max_per_stock_capital * 1.05):
                        candidates.append({
                            "symbol": symbol,
                            "price": current_price,
                            "score": score,
                            "tranche_capital": tranche_capital,
                            "adj_pct": adj_pct,
                            "reason": f"[Entrée 1/4 - {adj_pct:.1f}%] {reason}",
                            "is_pyramid": False,
                            "tranche_num": 1,
                        })
                else:
                    unrealized_plpc = float(existing_pos.unrealized_plpc)
                    next_tranche = min(4, round(current_stock_pct / 2.0) + 1)
                    print(
                        f"   • {symbol:10} | Prix: {current_price:8.2f}$ | ADR14: {adr_pct:4.1f}% | "
                        f"Score: {score:3}/100 | En cours ({current_stock_pct:.1f}%, P&L: {unrealized_plpc*100:+.1f}%)",
                        flush=True,
                    )

                    if (current_stock_value + tranche_capital) <= (max_per_stock_capital * 1.05):
                        if score >= MIN_PYRAMID_SCORE and unrealized_plpc >= self.min_gain_to_pyramid:
                            candidates.append({
                                "symbol": symbol,
                                "price": current_price,
                                "score": min(100, score + 5),
                                "tranche_capital": tranche_capital,
                                "adj_pct": adj_pct,
                                "reason": f"[Renforcement {next_tranche}/4 -> +{adj_pct:.1f}%] P&L {unrealized_plpc*100:+.1f}% | {reason}",
                                "is_pyramid": True,
                                "tranche_num": next_tranche,
                            })
            except Exception as e:
                print(f"   ⚠️ Erreur d'analyse sur {symbol}: {e}", flush=True)

        self._record_and_evaluate_observations(snapshots, today_str)

        if candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)

            best_choice = None
            for cand in candidates:
                allowed, block_reason = self.can_open_stock_trade(cand, now_utc, exp_stats, positions_map)
                if not allowed:
                    print(f"🛡️ [Filtre Pro Actions] {cand['symbol']} ignoré : {block_reason}", flush=True)
                    continue

                lib_eval = self.library.evaluate_candidate(
                    symbol=cand["symbol"],
                    base_score=cand["score"],
                    regime=regime_tag,
                    is_pyramid=bool(cand.get("is_pyramid")),
                )
                if lib_eval["veto"]:
                    print(f"📚 [Bibliothèque] {cand['symbol']} refusé : {lib_eval['summary']}", flush=True)
                    continue

                cand["score"] = lib_eval["adjusted_score"]
                cand["reason"] = f"{cand['reason']} | Biblio ({lib_eval['delta']:+d}): {lib_eval['summary']}"
                best_choice = cand
                break

            if not best_choice:
                print("😴 Aucun candidat ne passe les garde-fous pro actions et la bibliothèque (mode flat / maintien).", flush=True)
                self.print_dual_accounts_summary(total_portfolio, current_invested)
                self.save_state()
                return

            chosen_tranche = round(float(best_choice["tranche_capital"]), 2)
            if chosen_tranche <= 0 or (current_invested + chosen_tranche) > max_exposure_capital:
                print(f"🛑 Plafond global de 40% atteint ({current_exposure_pct:.1f}%) ou capital actif insuffisant. Aucun nouvel achat.", flush=True)
                self.print_dual_accounts_summary(total_portfolio, current_invested)
                self.save_state()
                return

            clean_symbol = best_choice["symbol"]
            action_label = f"RENFORCEMENT ({best_choice['tranche_num']}/4)" if best_choice["is_pyramid"] else "NOUVELLE ENTRÉE (1/4)"
            enriched_memory = self._build_market_memory_summary(snapshots, target_symbol=clean_symbol, regime_tag=regime_tag)

            print(f"\n🧠 Soumission du dossier Swing {clean_symbol} [{action_label}] au comité multi-IA...", flush=True)
            decision, feedback = ask_dual_ai_decision(
                symbol=clean_symbol,
                price=best_choice["price"],
                score=best_choice["score"],
                reason=best_choice["reason"],
                trend=market_trend,
                recent_history=list(positions_map.keys()),
                memory_summary=enriched_memory,
            )

            if decision != "VALIDÉ":
                print(f"🛡️ Ordre annulé pour {clean_symbol} : {feedback}", flush=True)
                self.print_dual_accounts_summary(total_portfolio, current_invested)
                self.save_state()
                return

            order_req = MarketOrderRequest(
                symbol=clean_symbol,
                notional=chosen_tranche,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_req)
            self.trade_history.setdefault("last_buy_dates", {})[clean_symbol] = today_str
            self.trade_history.setdefault("trades", []).append({
                "date": now_utc.isoformat(),
                "type": "OUVERTURE",
                "symbol": clean_symbol,
                "notional": chosen_tranche,
                "price": round(float(best_choice["price"]), 2),
                "score": best_choice["score"],
                "order_id": str(getattr(order, "id", "")),
            })
            current_invested += chosen_tranche
            print(f"✅ Ordre d'achat soumis pour {clean_symbol} ({chosen_tranche:.2f}$ — {action_label}).", flush=True)
            send_telegram_alert(
                f"🚀 *ACHAT {action_label}* : `{clean_symbol}` à `{best_choice['price']:.2f}$` "
                f"(`{chosen_tranche:.2f}$`, score `{best_choice['score']}/100`)"
            )
        else:
            print("\n😴 Aucun signal d'entrée ou de renforcement détecté sur ce cycle.", flush=True)

        self.print_dual_accounts_summary(total_portfolio, current_invested)
        self.save_state()
