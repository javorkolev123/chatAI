from dataclasses import dataclass
from collections import deque


@dataclass
class History:
    human: str
    ai: str

    def text(self) -> str:
        return f'human: {self.human}\nai: {self.ai}'


class ChatHistory:
    def __init__(self, retention_window=15):
        self.retention_window = retention_window
        self.history = deque(maxlen=self.retention_window)

    def add_history(self, human: str, ai: str):
        if len(self.history) == self.retention_window:
            self.history.popleft()
        self.history.append(History(human, ai))

    def text(self, _: str) -> str:
        t = ""
        for h in self.history:
            t += f'\n{h.text()}'
        return t
