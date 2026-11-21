# market_memory.py
import os
import json
from datetime import datetime, timezone

MEMORY_FILE = "stock_market_memory.json"


class MarketMemoryManager:
    def __init__(self, memory_file: str = MEMORY_FILE, max_lessons: int = 80):
        self.memory_file = memory_file
        self.max_lessons = max_lessons
        self.lessons = []
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("lessons"), list):
                    self.lessons = data["lessons"][-self.max_lessons :]
                    return
            except Exception:
                pass
        self.lessons = []

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump({"lessons": self.lessons[-self.max_lessons :]}, f, indent=4, ensure_ascii=False)

    def add_lesson(self, lesson_text: str, symbol: str | None = None, pnl_pct: float | None = None):
        entry = {
            "date": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "pnl_pct": round(float(pnl_pct), 2) if isinstance(pnl_pct, (int, float)) else None,
            "lesson": str(lesson_text),
        }
        self.lessons.append(entry)
        self.lessons = self.lessons[-self.max_lessons :]
        self.save_memory()

    def get_recent_lessons_summary(self, limit: int = 5) -> str:
        if not self.lessons:
            return "Aucune leçon mémoire enregistrée."
        recent = self.lessons[-limit:]
        parts = []
        for item in recent:
            if isinstance(item, dict):
                sym = f"[{item['symbol']}] " if item.get("symbol") else ""
                parts.append(f"{sym}{item.get('lesson', '')}")
            else:
                parts.append(str(item))
        return " | ".join(p for p in parts if p)
