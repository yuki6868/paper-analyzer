# main.py
from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
import time

from arxiv_keyword_search import search_arxiv
from translator import LibreTranslator
from discord_notifier import DiscordNotifier


def build_paper_message(
    no: int,
    title: str,
    summary_ja: str,
    abs_url: str,
    published: str,
    authors: list[str],
) -> str:
    authors_text = ", ".join(authors[:5])
    if len(authors) > 5:
        authors_text += " ほか"

    return (
        f"## No.{no}\n"
        f"**Title**: {title}\n"
        f"**Published**: {published}\n"
        f"**Authors**: {authors_text}\n"
        f"**URL**: {abs_url}\n\n"
        f"**日本語要約**\n{summary_ja}"
    )


def main(notify: bool = False, keyword: str = "large language model") -> None:
    load_dotenv()

    papers = search_arxiv(
        keyword=keyword,
        field="all",
        start=0,
        max_results=3,
        sort_by="relevance",
        sort_order="descending",
    )

    translator = LibreTranslator(
        api_url="http://127.0.0.1:7860",
        source_lang="en",
        target_lang="ja",
    )

    notifier = None
    if notify:
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        if not webhook_url:
            raise ValueError("DISCORD_WEBHOOK_URL が設定されていません")
            
        notifier = DiscordNotifier(
            webhook_url=webhook_url,
            username="arXiv Translator",
        )

    for i, paper in enumerate(papers, start=1):
        print(f"[{i}/{len(papers)}] 翻訳中: {paper.title}")
        summary_ja = translator.translate_text_with_retry(paper.summary)

        message = build_paper_message(
            no=i,
            title=paper.title,
            summary_ja=summary_ja,
            abs_url=paper.abs_url,
            published=paper.published,
            authors=paper.authors,
        )

        print("=" * 80)
        print(message)
        print()

        if notify and notifier is not None:
            chunks = notifier.split_message(message, limit=1800)
            notifier.send_messages(chunks, wait_sec=1.0)
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notify", action="store_true", help="Discordに通知する")
    parser.add_argument("--keyword", type=str, default="large language model", help="検索キーワード")
    args = parser.parse_args()

    main(notify=args.notify, keyword=args.keyword)