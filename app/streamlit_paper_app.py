from __future__ import annotations

import os
import sys
import time
import urllib.error
from dataclasses import asdict
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from arxiv_keyword_search import ArxivPaper, search_arxiv, search_latest_by_category
from discord_notifier import DiscordNotifier

try:
    from translator import LibreTranslator as ProjectLibreTranslator
except Exception:
    ProjectLibreTranslator = None

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CATEGORY = "cs.AI"


class SimpleLibreTranslator:
    def __init__(self, api_url: str, source_lang: str = "en", target_lang: str = "ja") -> None:
        self.api_url = api_url.rstrip("/")
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_text_with_retry(self, text: str, retries: int = 3, wait_sec: float = 1.5) -> str:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                return self.translate_text(text)
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(wait_sec * (attempt + 1))
        if last_error is not None:
            raise last_error
        return text

    def translate_text(self, text: str) -> str:
        response = requests.post(
            f"{self.api_url}/translate",
            json={
                "q": text,
                "source": self.source_lang,
                "target": self.target_lang,
                "format": "text",
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        translated = data.get("translatedText")
        if not translated:
            raise ValueError("LibreTranslate の応答に translatedText がありません。")
        return translated


def build_translator(api_url: str):
    if ProjectLibreTranslator is not None:
        return ProjectLibreTranslator(
            api_url=api_url,
            source_lang="en",
            target_lang="ja",
        )
    return SimpleLibreTranslator(api_url=api_url, source_lang="en", target_lang="ja")


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def create_embeddings(texts: tuple[str, ...], model_name: str) -> np.ndarray:
    model = load_sentence_transformer(model_name)
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


@st.cache_data(show_spinner=False)
def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    n_samples = len(embeddings)
    if n_samples < 3:
        raise ValueError("論文マップ作成には少なくとも3件必要です。")

    perplexity = min(30.0, max(2.0, float(n_samples - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def build_document_text(paper: ArxivPaper) -> str:
    return f"{paper.title}\n\n{paper.summary}".strip()


def find_top_similar_indices(embeddings: np.ndarray, top_k: int = 5) -> list[list[int]]:
    sim = cosine_similarity(embeddings)
    neighbors: list[list[int]] = []
    for i in range(sim.shape[0]):
        row = sim[i].copy()
        row[i] = -1.0
        top = np.argsort(row)[::-1][:top_k]
        neighbors.append(top.tolist())
    return neighbors


def papers_to_records(papers: list[ArxivPaper]) -> list[dict[str, Any]]:
    return [asdict(p) for p in papers]


def records_to_papers(records: list[dict[str, Any]]) -> list[ArxivPaper]:
    return [ArxivPaper(**record) for record in records]


def fetch_papers(category: str, max_results: int, keyword: str | None = None) -> list[ArxivPaper]:
    if keyword:
        return search_arxiv(
            keyword=keyword,
            field="all",
            start=0,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending",
        )
    return search_latest_by_category(category=category, max_results=max_results)


def translate_papers_if_needed(
    papers: list[ArxivPaper],
    translate_enabled: bool,
    translator_api_url: str,
) -> dict[str, str]:
    translations: dict[str, str] = st.session_state.get("translations", {})
    if not translate_enabled:
        return translations

    translator = build_translator(api_url=translator_api_url)
    progress = st.progress(0.0, text="日本語訳を作成しています...")

    for idx, paper in enumerate(papers, start=1):
        key = paper.arxiv_id or paper.abs_url or paper.title
        if key not in translations:
            try:
                translations[key] = translator.translate_text_with_retry(paper.summary)
            except Exception as exc:
                translations[key] = f"翻訳に失敗しました: {exc}"
        progress.progress(idx / len(papers), text=f"日本語訳を作成しています... {idx}/{len(papers)}")

    progress.empty()
    st.session_state["translations"] = translations
    return translations


def build_paper_message(paper: ArxivPaper, summary_ja: str, no: int | None = None) -> str:
    authors_text = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors_text += " ほか"

    prefix = f"## No.{no}\n" if no is not None else ""
    return (
        f"{prefix}"
        f"**Title**: {paper.title}\n"
        f"**Published**: {paper.published}\n"
        f"**Authors**: {authors_text}\n"
        f"**URL**: {paper.abs_url}\n\n"
        f"**日本語要約**\n{summary_ja}"
    )


def send_paper_to_discord(webhook_url: str, paper: ArxivPaper, summary_ja: str, username: str) -> None:
    notifier = DiscordNotifier(webhook_url=webhook_url, username=username)
    message = build_paper_message(paper=paper, summary_ja=summary_ja)
    chunks = notifier.split_message(message, limit=1800)
    notifier.send_messages(chunks, wait_sec=0.6)


def build_map_figure(papers: list[ArxivPaper], coords: np.ndarray, selected_index: int | None) -> go.Figure:
    selected_points = [selected_index] if selected_index is not None else None

    fig = go.Figure(
        data=[
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                text=[paper.title for paper in papers],
                customdata=list(range(len(papers))),
                selectedpoints=selected_points,
                marker={"size": 11, "opacity": 0.85},
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=700,
        margin={"l": 20, "r": 20, "t": 40, "b": 40},
        title="論文マップ",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        clickmode="event+select",
    )
    return fig


def render_paper_card(
    paper: ArxivPaper,
    summary_ja: str,
    index: int,
    webhook_url: str,
    discord_username: str,
) -> None:
    with st.expander(f"{index + 1}. {paper.title}", expanded=index == 0):
        st.write(f"**Published**: {paper.published}")
        st.write(f"**Authors**: {', '.join(paper.authors)}")
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("Abstract", paper.abs_url)
        with col2:
            st.link_button("PDF", paper.pdf_url)

        st.write("**日本語要約**")
        st.write(summary_ja)

        if webhook_url:
            if st.button("Discord に送る", key=f"send_{paper.arxiv_id}_{index}"):
                try:
                    send_paper_to_discord(
                        webhook_url=webhook_url,
                        paper=paper,
                        summary_ja=summary_ja,
                        username=discord_username,
                    )
                    st.success("Discord に送信しました。")
                except Exception as exc:
                    st.error(f"Discord送信に失敗しました: {exc}")
        else:
            st.caption("Webhook URL を入れると Discord に送れます。")


def render_selected_paper_detail(
    papers: list[ArxivPaper],
    translations: dict[str, str],
    similar_indices: list[list[int]],
    selected_index: int,
) -> None:
    paper = papers[selected_index]
    key = paper.arxiv_id or paper.abs_url or paper.title
    summary_ja = translations.get(key, paper.summary)

    st.subheader("選択中の論文")
    st.write(f"### {paper.title}")
    st.write(f"**Published**: {paper.published}")
    st.write(f"**Authors**: {', '.join(paper.authors)}")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("Abstract を開く", paper.abs_url)
    with col2:
        st.link_button("PDF を開く", paper.pdf_url)

    st.write("**日本語要約**")
    st.write(summary_ja)

    st.write("**類似論文 5 件**")
    for rank, idx in enumerate(similar_indices[selected_index], start=1):
        sim_paper = papers[idx]
        if st.button(f"{rank}. {sim_paper.title}", key=f"similar_{selected_index}_{idx}"):
            st.session_state["selected_paper_index"] = idx
            st.rerun()
        st.caption(f"{sim_paper.published} / {', '.join(sim_paper.authors[:3])}")


def main() -> None:
    st.set_page_config(page_title="Paper Analyzer Web", layout="wide")
    st.title("arXiv Paper Analyzer")
    st.caption("カテゴリ検索、日本語要約、Discord送信、論文マップ表示を1つにまとめたWebアプリです。")

    with st.sidebar:
        st.header("設定")
        category = st.text_input("カテゴリ", value=DEFAULT_CATEGORY, help="例: cs.AI, cs.LG, stat.ML")
        keyword = st.text_input("キーワード検索（任意）", value="", help="空ならカテゴリの最新論文を取得します")
        max_results = st.slider("取得件数", min_value=10, max_value=200, value=50, step=10)
        translate_enabled = st.checkbox("日本語訳を作成する", value=True)
        translator_api_url = st.text_input(
            "LibreTranslate API URL",
            value=os.getenv("LIBRETRANSLATE_API_URL", "http://127.0.0.1:7860"),
        )
        model_name = st.text_input("埋め込みモデル", value=DEFAULT_MODEL_NAME)
        webhook_url = st.text_input(
            "Discord Webhook URL",
            value=os.getenv("DISCORD_WEBHOOK_URL", ""),
            type="password",
        )
        discord_username = st.text_input("Discord username", value="arXiv Translator")

        search_clicked = st.button("論文検索", use_container_width=True)
        map_clicked = st.button("論文マップ作成", use_container_width=True)

    if "papers" not in st.session_state:
        st.session_state["papers"] = []
    if "translations" not in st.session_state:
        st.session_state["translations"] = {}
    if "selected_paper_index" not in st.session_state:
        st.session_state["selected_paper_index"] = 0

    if search_clicked:
        try:
            with st.spinner("論文を取得しています..."):
                papers = fetch_papers(category=category, max_results=max_results, keyword=keyword.strip() or None)
            st.session_state["papers"] = papers_to_records(papers)
            st.session_state["selected_paper_index"] = 0
            if translate_enabled and papers:
                translations = translate_papers_if_needed(papers, translate_enabled, translator_api_url)
                st.session_state["translations"] = translations
            st.success(f"{len(papers)} 件の論文を取得しました。")
        except urllib.error.HTTPError as exc:
            st.error(f"論文取得に失敗しました: HTTP {exc.code}")
        except Exception as exc:
            st.error(f"論文取得に失敗しました: {exc}")

    papers = records_to_papers(st.session_state.get("papers", []))
    translations = st.session_state.get("translations", {})

    if papers:
        st.subheader("論文一覧")
        for index, paper in enumerate(papers):
            key = paper.arxiv_id or paper.abs_url or paper.title
            summary_ja = translations.get(key, paper.summary)
            render_paper_card(
                paper=paper,
                summary_ja=summary_ja,
                index=index,
                webhook_url=webhook_url,
                discord_username=discord_username,
            )

    if map_clicked:
        if not papers:
            st.warning("先に論文検索をしてください。")
        else:
            try:
                with st.spinner("論文マップを作成しています..."):
                    texts = tuple(build_document_text(paper) for paper in papers)
                    embeddings = create_embeddings(texts=texts, model_name=model_name)
                    coords = reduce_to_2d(embeddings)
                    similar_indices = find_top_similar_indices(embeddings, top_k=5)
                st.session_state["map_coords"] = coords.tolist()
                st.session_state["similar_indices"] = similar_indices
                st.success("論文マップを作成しました。")
            except Exception as exc:
                st.error(f"論文マップ作成に失敗しました: {exc}")

    if papers and "map_coords" in st.session_state and "similar_indices" in st.session_state:
        coords = np.asarray(st.session_state["map_coords"], dtype=np.float32)
        similar_indices = st.session_state["similar_indices"]

        st.subheader("論文マップ")
        left, right = st.columns([1.6, 1.0], gap="large")
        with left:
            fig = build_map_figure(
                papers=papers,
                coords=coords,
                selected_index=st.session_state.get("selected_paper_index", 0),
            )
            event = st.plotly_chart(fig, use_container_width=True, key="paper_map", on_select="rerun")
            selection = event.selection if event else None
            if selection and selection.points:
                point = selection.points[0]
                point_index = point.get("customdata", point.get("point_index", 0))
                if point_index != st.session_state.get("selected_paper_index"):
                    st.session_state["selected_paper_index"] = int(point_index)
                    st.rerun()

        with right:
            render_selected_paper_detail(
                papers=papers,
                translations=translations,
                similar_indices=similar_indices,
                selected_index=st.session_state.get("selected_paper_index", 0),
            )


if __name__ == "__main__":
    main()
