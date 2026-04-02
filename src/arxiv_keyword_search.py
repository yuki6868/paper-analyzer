# arxiv_keyword_search.py
from __future__ import annotations

import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List

import feedparser


BASE_URL = "https://export.arxiv.org/api/query"


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    published: str
    authors: List[str]
    pdf_url: str
    abs_url: str


def build_search_query(keyword: str, field: str = "all") -> str:
    """
    arXivの検索クエリを作る。
    field:
        - all      : 全文対象
        - ti       : タイトル
        - abs      : 概要
        - au       : 著者
        - cat      : カテゴリ
    """
    return f"{field}:{keyword}"


def search_arxiv(
    keyword: str,
    field: str = "all",
    start: int = 0,
    max_results: int = 5,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[ArxivPaper]:
    """
    arXiv APIでキーワード検索して論文一覧を返す。
    """
    search_query = build_search_query(keyword, field=field)

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url) as response:
        raw_data = response.read()

    feed = feedparser.parse(raw_data)
    papers: List[ArxivPaper] = []

    for entry in feed.entries:
        pdf_url = ""
        abs_url = getattr(entry, "id", "")

        for link in entry.get("links", []):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
            elif link.get("rel") == "alternate":
                abs_url = link.get("href", abs_url)

        arxiv_id = entry.id.split("/abs/")[-1] if "id" in entry else ""

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=entry.get("title", "").replace("\n", " ").strip(),
                summary=entry.get("summary", "").replace("\n", " ").strip(),
                published=entry.get("published", ""),
                authors=[author.name for author in entry.get("authors", [])],
                pdf_url=pdf_url,
                abs_url=abs_url,
            )
        )

    return papers


def print_papers(papers: List[ArxivPaper]) -> None:
    if not papers:
        print("検索結果はありませんでした。")
        return

    for i, paper in enumerate(papers, start=1):
        print("=" * 80)
        print(f"No. {i}")
        print(f"arXiv ID : {paper.arxiv_id}")
        print(f"Title    : {paper.title}")
        print(f"Published: {paper.published}")
        print(f"Authors  : {', '.join(paper.authors)}")
        print(f"abs URL  : {paper.abs_url}")
        print(f"PDF URL  : {paper.pdf_url}")
        print("Summary  :")
        print(paper.summary[:600] + ("..." if len(paper.summary) > 600 else ""))
        print()


if __name__ == "__main__":
    keyword = "large language model"
    papers = search_arxiv(
        keyword=keyword,
        field="all",
        start=0,
        max_results=5,
        sort_by="relevance",
        sort_order="descending",
    )
    print_papers(papers)

    # 複数回連続で呼ぶときは、公式推奨に従って少し待つ
    time.sleep(3)