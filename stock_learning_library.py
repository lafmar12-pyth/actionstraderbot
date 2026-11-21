# stock_learning_library.py
import os
import json
from datetime import datetime, timezone

LIBRARY_FILE = "stock_learning_library.json"


class StockLearningLibrary:
    """
    Bibliothèque d'apprentissage local (0 $ d'API) pour le swing trading actions.
    Consolide les performances par titre et par régime macro (SPY) afin d'orienter
    les décisions et les garde-fous.
    """

    def __init__(self, library_file: str = LIBRARY_FILE):
        self.library_file = library_file
        self.data = self._default_data()
        self.load()

    @staticmethod
    def _default_data() -> dict:
        return {
            "updated_at": None,
            "pairs": {},    # stats par symbole (ex: AAPL, NVDA, TD)
            "regimes": {},  # stats par régime marché (HAUSSIER, BAISSIER, NEUTRE)
        }

    def load(self):
        if os.path.exists(self.library_file):
            try:
                with open(self.library_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    base = self._default_data()
                    if isinstance(loaded.get("pairs"), dict):
                        base["pairs"] = loaded["pairs"]
                    if isinstance(loaded.get("regimes"), dict):
                        base["regimes"] = loaded["regimes"]
                    base["updated_at"] = loaded.get("updated_at")
                    self.data = base
                    return
            except Exception:
                pass
        self.data = self._default_data()

    def save(self):
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.library_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def record_closed_trade(self, symbol: str, plpc: float, pl_dollars: float, reason_tag: str = ""):
        """Met à jour les statistiques historiques d'un titre à la clôture d'une position."""
        pairs = self.data.setdefault("pairs", {})
        stats = pairs.setdefault(symbol, {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pl_dollars": 0.0,
            "sum_pl_pct": 0.0,
            "avg_pl_pct": 0.0,
            "last_pl_pct": 0.0,
            "last_reason": "",
            "last_date": None,
        })

        pnl_pct = float(plpc) * 100.0
        pnl_usd = float(pl_dollars)

        stats["closed_trades"] = int(stats.get("closed_trades", 0)) + 1
        if pnl_pct > 0:
            stats["wins"] = int(stats.get("wins", 0)) + 1
        else:
            stats["losses"] = int(stats.get("losses", 0)) + 1

        stats["total_pl_dollars"] = round(float(stats.get("total_pl_dollars", 0.0)) + pnl_usd, 2)
        stats["sum_pl_pct"] = round(float(stats.get("sum_pl_pct", 0.0)) + pnl_pct, 2)
        stats["avg_pl_pct"] = round(stats["sum_pl_pct"] / max(1, stats["closed_trades"]), 2)
        stats["last_pl_pct"] = round(pnl_pct, 2)
        stats["last_reason"] = str(reason_tag)
        stats["last_date"] = datetime.now(timezone.utc).isoformat()

        self.save()

    def sync_from_journal(self, signals: list):
        """Reconsolide les statistiques par régime à partir des signaux évalués à ~5 jours."""
        if not isinstance(signals, list):
            return

        evaluated_by_regime: dict[str, list[float]] = {}
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            if not sig.get("evaluated_5d"):
                continue
            ret = sig.get("return_5d_pct")
            if not isinstance(ret, (int, float)):
                continue
            regime = str(sig.get("regime") or "NEUTRE").upper()
            evaluated_by_regime.setdefault(regime, []).append(float(ret))

        regimes_out = {}
        for regime, returns in evaluated_by_regime.items():
            n = len(returns)
            wins = sum(1 for r in returns if r > 0)
            regimes_out[regime] = {
                "signals_5d": n,
                "wins_5d": wins,
                "win_rate_5d_pct": round((wins / n) * 100.0, 1) if n > 0 else 0.0,
                "avg_ret_5d_pct": round(sum(returns) / n, 2) if n > 0 else 0.0,
            }

        self.data["regimes"] = regimes_out
        self.save()

    def evaluate_candidate(self, symbol: str, base_score: int, regime: str, is_pyramid: bool = False) -> dict:
        """
        Ajuste le score ou applique un veto local basé sur la mémoire historique du titre et du régime.
        Borné entre -18 et +8 points pour éviter toute dérive.
        """
        score = int(base_score)
        delta = 0
        veto = False
        notes = []

        reg_key = str(regime or "NEUTRE").upper()
        reg_stats = (self.data.get("regimes") or {}).get(reg_key, {})
        pair_stats = (self.data.get("pairs") or {}).get(symbol, {})

        # 1. Contexte historique du régime (SPY)
        reg_n = int(reg_stats.get("signals_5d", 0))
        reg_wr = float(reg_stats.get("win_rate_5d_pct", 0.0))
        reg_avg = float(reg_stats.get("avg_ret_5d_pct", 0.0))

        if reg_n >= 10:
            if reg_wr < 42.0 or reg_avg < -0.8:
                delta -= 8
                notes.append(f"régime {reg_key} faible sur 5j (WR {reg_wr:.1f}%, moy {reg_avg:+.2f}%)")
                if not is_pyramid and score < 85:
                    veto = True
            elif reg_wr >= 60.0 and reg_avg > 0.8:
                delta += 4
                notes.append(f"régime {reg_key} porteur sur 5j (WR {reg_wr:.1f}%, moy {reg_avg:+.2f}%)")

        # 2. Historique propre à l'action
        closed_cnt = int(pair_stats.get("closed_trades", 0))
        avg_pl = float(pair_stats.get("avg_pl_pct", 0.0))
        losses = int(pair_stats.get("losses", 0))

        if closed_cnt >= 2:
            if avg_pl < -1.5:
                delta -= 8
                notes.append(f"historique faible sur {symbol} (moy {avg_pl:+.2f}% sur {closed_cnt} trades)")
                if not is_pyramid and losses >= 2 and score < 88:
                    veto = True
            elif avg_pl > 2.0 and losses == 0:
                delta += 4
                notes.append(f"historique solide sur {symbol} (moy {avg_pl:+.2f}%)")

        delta = max(-18, min(8, delta))
        adjusted_score = max(0, min(100, score + delta))

        return {
            "veto": veto,
            "adjusted_score": adjusted_score,
            "delta": delta,
            "summary": "; ".join(notes) if notes else "bibliothèque neutre",
        }

    def get_summary_for_committee(self, symbol: str, regime: str) -> str:
        """Résumé court à injecter dans le prompt du comité IA."""
        reg_key = str(regime or "NEUTRE").upper()
        reg_stats = (self.data.get("regimes") or {}).get(reg_key, {})
        pair_stats = (self.data.get("pairs") or {}).get(symbol, {})

        if pair_stats.get("closed_trades", 0) > 0:
            pair_txt = (
                f"{symbol}: {pair_stats['closed_trades']} trade(s) clos, "
                f"moy {pair_stats['avg_pl_pct']:+.2f}%, cumul {pair_stats['total_pl_dollars']:+.2f}$"
            )
        else:
            pair_txt = f"{symbol}: aucun trade clos en bibliothèque"

        if reg_stats.get("signals_5d", 0) > 0:
            reg_txt = (
                f"Régime {reg_key} (5j, n={reg_stats['signals_5d']}): "
                f"WR {reg_stats['win_rate_5d_pct']:.1f}%, moy {reg_stats['avg_ret_5d_pct']:+.2f}%"
            )
        else:
            reg_txt = f"Régime {reg_key}: collecte 5j en cours"

        return f"Bibliothèque -> {pair_txt} | {reg_txt}"
