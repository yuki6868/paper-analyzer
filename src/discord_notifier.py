# discord_notifier.py
from __future__ import annotations

import time
from typing import Iterable

import requests


class DiscordNotifier:
    def __init__(self, webhook_url: str, username: str = "Paper Analyzer") -> None:
        self.webhook_url = webhook_url
        self.username = username

    def send_message(self, content: str) -> None:
        payload = {
            "username": self.username,
            "content": content,
        }

        response = requests.post(self.webhook_url, json=payload, timeout=20)
        response.raise_for_status()

    def send_messages(self, messages: Iterable[str], wait_sec: float = 1.0) -> None:
        for message in messages:
            self.send_message(message)
            time.sleep(wait_sec)

    @staticmethod
    def split_message(text: str, limit: int = 1800) -> list[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + limit

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            split_pos = text.rfind("\n", start, end)
            if split_pos == -1 or split_pos <= start:
                split_pos = text.rfind(" ", start, end)
            if split_pos == -1 or split_pos <= start:
                split_pos = end

            chunks.append(text[start:split_pos].strip())
            start = split_pos

        return [chunk for chunk in chunks if chunk]