from __future__ import annotations

import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List
from urllib.error import HTTPError, URLError

import feedparser


BASE_URL = "https://export.arxiv.org/api/query"
DEFAULT_USER_AGENT = "paper-analyzer/1.0 (contact: local-user)"


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


def _fetch_with_retry(url: str, *, retries: int = 5, base_wait: float = 3.0) -> bytes:
    """arXiv API を User-Agent 付きで呼び、429/5xx で指数バックオフ再試行する。"""
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    last_error: Exception | None = None

    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except HTTPError as e:
            last_error = e
            if e.code in (429, 500, 502, 503, 504):
                wait = base_wait * (2 ** attempt)
                print(f"arXiv API が一時的に混雑しています (HTTP {e.code})。{wait:.1f}秒待って再試行します... [{attempt + 1}/{retries}]")
                time.sleep(wait)
                continue
            raise
        except URLError as e:
            last_error = e
            wait = base_wait * (2 ** attempt)
            print(f"arXiv API への接続に失敗しました。{wait:.1f}秒待って再試行します... [{attempt + 1}/{retries}]")
            time.sleep(wait)
            continue

    raise RuntimeError(f"arXiv API の取得に失敗しました: {last_error}")


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
    429等に備えてリトライする。
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
    raw_data = _fetch_with_retry(url)

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


def search_latest_by_category(category: str, max_results: int = 200) -> List[ArxivPaper]:
    """
    指定カテゴリの最新論文を取得する。
    いきなり大きすぎる件数は避け、必要なら呼び出し側で分割取得する。
    """
    if max_results <= 0:
        return []

    # arXiv 側の負荷と 429 を少し避けるため、100件ずつに分割取得
    batch_size = 100
    results: List[ArxivPaper] = []

    for start in range(0, max_results, batch_size):
        size = min(batch_size, max_results - start)
        batch = search_arxiv(
            keyword=category,
            field="cat",
            start=start,
            max_results=size,
            sort_by="submittedDate",
            sort_order="descending",
        )
        results.extend(batch)

        if start + batch_size < max_results:
            time.sleep(3)

    return results


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
