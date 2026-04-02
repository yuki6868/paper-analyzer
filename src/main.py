# main.py
from arxiv_keyword_search import search_arxiv, print_papers
from translator import LibreTranslator


def main() -> None:
    keyword = "large language model"

    papers = search_arxiv(
        keyword=keyword,
        field="all",
        start=0,
        max_results=5,
        sort_by="relevance",
        sort_order="descending",
    )

    translator = LibreTranslator(
        api_url="http://127.0.0.1:7860",
        source_lang="en",
        target_lang="ja",
    )

    for i, paper in enumerate(papers, start=1):
        print(f"[{i}/{len(papers)}] 翻訳中: {paper.title}")
        # translated_summary = translator.translate_text_with_retry(paper.summary)
        paper.summary_ja = translator.translate_text_with_retry(paper.summary)

        print("=" * 80)
        print(f"Title      : {paper.title}")
        print(f"Summary EN : {paper.summary}")
        print()
        print(f"Summary JA : {paper.summary_ja}")
        print()

    print_papers(papers)


if __name__ == "__main__":
    main()