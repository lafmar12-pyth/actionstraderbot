import os
import json

STATE_FILE = "stock_bot_state.json"
MEMORY_FILE = "stock_market_memory.json"

class MarketMemoryManager:
    def __init__(self):
        self.load_state()
        self.load_memory()

    def load_state(self):
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

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    self.lessons = json.load(f).get('lessons', [])
                    return
            except Exception:
                pass
        self.lessons = []

    def add_lesson(self, lesson_text):
        self.lessons.append(lesson_text)
        if len(self.lessons) > 50:
            self.lessons = self.lessons[-50:]
        with open(MEMORY_FILE, 'w') as f:
            json.dump({'lessons': self.lessons}, f, indent=4)
