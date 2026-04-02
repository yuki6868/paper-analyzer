# translator.py
from __future__ import annotations

import time
from typing import List

from libretranslatepy import LibreTranslateAPI


class LibreTranslator:
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:7860",
        source_lang: str = "en",
        target_lang: str = "ja",
        max_chars: int = 1200,
        wait_sec: float = 0.3,
    ) -> None:
        self.api = LibreTranslateAPI(api_url)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_chars = max_chars
        self.wait_sec = wait_sec

    def chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.max_chars:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + self.max_chars

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            split_pos = text.rfind(" ", start, end)
            if split_pos == -1 or split_pos <= start:
                split_pos = end

            chunks.append(text[start:split_pos].strip())
            start = split_pos

        return [chunk for chunk in chunks if chunk]

    def translate_text(self, text: str) -> str:
        chunks = self.chunk_text(text)
        translated_chunks: List[str] = []

        for chunk in chunks:
            translated = self.api.translate(
                chunk,
                self.source_lang,
                self.target_lang,
            )
            translated_chunks.append(translated)
            time.sleep(self.wait_sec)

        return " ".join(translated_chunks)

    def translate_text_with_retry(
        self,
        text: str,
        retries: int = 3,
        retry_wait_sec: float = 2.0,
    ) -> str:
        last_error: Exception | None = None

        for _ in range(retries):
            try:
                return self.translate_text(text)
            except Exception as e:
                last_error = e
                time.sleep(retry_wait_sec)

        raise RuntimeError(f"翻訳に失敗しました: {last_error}")