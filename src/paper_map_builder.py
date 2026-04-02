from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from arxiv_keyword_search import ArxivPaper, search_latest_by_category


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_document_text(paper: ArxivPaper) -> str:
    return f"{paper.title}\n\n{paper.summary}".strip()


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers がインストールされていません。"
            " `pip install sentence-transformers` を実行してください。"
        ) from exc

    return SentenceTransformer(model_name)


def create_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    model = load_sentence_transformer(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def reduce_to_2d(embeddings: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    n_samples = len(embeddings)
    if n_samples < 3:
        raise ValueError("論文数が少なすぎます。少なくとも3件以上にしてください。")

    adjusted_perplexity = min(perplexity, max(2.0, float(n_samples - 1) / 3.0))

    reducer = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def find_top_similar_indices(embeddings: np.ndarray, top_k: int = 5) -> list[list[int]]:
    sim = cosine_similarity(embeddings)
    all_neighbors: list[list[int]] = []

    for i in range(sim.shape[0]):
        row = sim[i].copy()
        row[i] = -math.inf
        neighbor_indices = np.argsort(row)[::-1][:top_k]
        all_neighbors.append(neighbor_indices.tolist())

    return all_neighbors


def build_paper_payload(
    papers: list[ArxivPaper],
    coords: np.ndarray,
    similar_indices: list[list[int]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    for i, paper in enumerate(papers):
        payload.append(
            {
                "index": i,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "summary": paper.summary,
                "published": paper.published,
                "authors": paper.authors,
                "abs_url": paper.abs_url,
                "pdf_url": paper.pdf_url,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "similar_indices": similar_indices[i],
            }
        )

    return payload


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>arXiv Paper Map</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fb;
      color: #1f2937;
    }
    .page {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(360px, 1fr);
      gap: 16px;
      padding: 16px;
      min-height: 100vh;
      box-sizing: border-box;
    }
    .panel {
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
      overflow: hidden;
    }
    .panel-header {
      padding: 16px 18px;
      border-bottom: 1px solid #e5e7eb;
    }
    .panel-header h1,
    .panel-header h2 {
      margin: 0;
      font-size: 18px;
    }
    .meta {
      margin-top: 8px;
      color: #6b7280;
      font-size: 14px;
    }
    #plot {
      width: 100%;
      height: calc(100vh - 120px);
      min-height: 680px;
    }
    .detail-body {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: calc(100vh - 120px);
      overflow-y: auto;
    }
    .paper-title {
      margin: 0;
      font-size: 22px;
      line-height: 1.5;
    }
    .sub {
      font-size: 14px;
      color: #6b7280;
      line-height: 1.7;
    }
    .links {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .links a {
      text-decoration: none;
      color: #2563eb;
      font-weight: 600;
    }
    .summary {
      white-space: pre-wrap;
      line-height: 1.8;
      font-size: 15px;
    }
    .similar-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .similar-item {
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 12px;
      background: #fafafa;
    }
    .similar-item button {
      border: none;
      background: transparent;
      padding: 0;
      margin: 0 0 6px;
      cursor: pointer;
      color: #111827;
      font-size: 15px;
      font-weight: 700;
      text-align: left;
    }
    .similar-item button:hover {
      color: #2563eb;
    }
    .empty {
      color: #6b7280;
      line-height: 1.8;
    }
    @media (max-width: 980px) {
      .page {
        grid-template-columns: 1fr;
      }
      #plot {
        height: 60vh;
        min-height: 420px;
      }
      .detail-body {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="panel">
      <div class="panel-header">
        <h1>arXiv Paper Map</h1>
        <div class="meta">カテゴリ: __CATEGORY__ / 件数: __COUNT__ / モデル: __MODEL__</div>
      </div>
      <div id="plot"></div>
    </section>

    <aside class="panel">
      <div class="panel-header">
        <h2>論文詳細</h2>
      </div>
      <div class="detail-body" id="detail-panel">
        <div class="empty">左のマップから論文をクリックすると、概要と類似論文5件が表示されます。</div>
      </div>
    </aside>
  </div>

  <script>
    const papers = __PAPERS_JSON__;

    const trace = {
      x: papers.map(p => p.x),
      y: papers.map(p => p.y),
      text: papers.map(p => p.title),
      mode: 'markers',
      type: 'scattergl',
      customdata: papers.map(p => p.index),
      marker: {
        size: 11,
        opacity: 0.85,
      },
      hovertemplate: '<b>%{text}</b><extra></extra>',
    };

    const layout = {
      margin: { l: 20, r: 20, t: 20, b: 40 },
      xaxis: { title: 'Dimension 1', zeroline: false },
      yaxis: { title: 'Dimension 2', zeroline: false },
      hovermode: 'closest',
    };

    Plotly.newPlot('plot', [trace], layout, { responsive: true, displaylogo: false });

    function escapeHtml(text) {
      return text
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    function renderDetail(index) {
      const paper = papers[index];
      const authors = paper.authors.length ? paper.authors.join(', ') : '不明';
      const similarHtml = paper.similar_indices.map(simIndex => {
        const simPaper = papers[simIndex];
        return `
          <div class="similar-item">
            <button onclick="focusPaper(${simIndex})">${escapeHtml(simPaper.title)}</button>
            <div class="sub">${escapeHtml(simPaper.published)} / ${escapeHtml(simPaper.authors.slice(0, 3).join(', '))}</div>
          </div>
        `;
      }).join('');

      document.getElementById('detail-panel').innerHTML = `
        <div>
          <h3 class="paper-title">${escapeHtml(paper.title)}</h3>
          <div class="sub">${escapeHtml(paper.published)}</div>
          <div class="sub">${escapeHtml(authors)}</div>
        </div>
        <div class="links">
          <a href="${paper.abs_url}" target="_blank" rel="noreferrer">Abstract</a>
          <a href="${paper.pdf_url}" target="_blank" rel="noreferrer">PDF</a>
        </div>
        <div>
          <strong>Abstract</strong>
          <div class="summary">${escapeHtml(paper.summary)}</div>
        </div>
        <div>
          <strong>類似論文 5 件</strong>
          <div class="similar-list">${similarHtml}</div>
        </div>
      `;
    }

    function focusPaper(index) {
      renderDetail(index);
      Plotly.Fx.hover('plot', [{ curveNumber: 0, pointNumber: index }]);
    }

    document.getElementById('plot').on('plotly_click', function(data) {
      const pointIndex = data.points[0].customdata;
      renderDetail(pointIndex);
    });

    if (papers.length > 0) {
      renderDetail(0);
    }

    window.focusPaper = focusPaper;
  </script>
</body>
</html>
"""


def render_html(
    paper_payload: list[dict[str, Any]],
    category: str,
    model_name: str,
) -> str:
    return (
        HTML_TEMPLATE
        .replace("__CATEGORY__", category)
        .replace("__COUNT__", str(len(paper_payload)))
        .replace("__MODEL__", model_name)
        .replace("__PAPERS_JSON__", json.dumps(paper_payload, ensure_ascii=False))
    )


def save_html(html: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="arXivの最新論文を2次元マップ化し、類似論文を見られるHTMLを生成する。"
    )
    parser.add_argument("--category", type=str, default="cs.AI", help="arXivカテゴリ。例: cs.AI")
    parser.add_argument("--max-results", type=int, default=200, help="取得件数")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformers のモデル名",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/paper_map.html",
        help="出力HTMLファイル",
    )
    args = parser.parse_args()

    papers = search_latest_by_category(category=args.category, max_results=args.max_results)
    if not papers:
        raise ValueError("論文が取得できませんでした。カテゴリ指定を確認してください。")

    texts = [build_document_text(paper) for paper in papers]
    embeddings = create_embeddings(texts, model_name=args.model_name)
    coords = reduce_to_2d(embeddings)
    similar_indices = find_top_similar_indices(embeddings, top_k=5)

    payload = build_paper_payload(papers, coords, similar_indices)
    html = render_html(payload, category=args.category, model_name=args.model_name)

    output_path = Path(args.output)
    save_html(html, output_path)
    print(f"HTMLを出力しました: {output_path.resolve()}")


if __name__ == "__main__":
    main()
