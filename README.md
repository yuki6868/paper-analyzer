# Paper Analyzer

arXiv論文を **取得・翻訳・分析・可視化** するためのローカル完結型アプリです。
Streamlit を使ったWeb UIと、Pythonベースの分析ロジックで構成されています。

---

## 概要

このプロジェクトは以下を目的としています。

* 最新論文の効率的なキャッチアップ
* 日本語での高速理解
* 類似論文の発見
* 研究トレンドの把握

完全ローカル環境（無料）で動作します。

---

## 主な機能

### 論文検索

* arXiv APIからカテゴリ別に論文取得
* キーワード検索対応
* 最大200件まで取得可能

---

### 日本語要約

* LibreTranslate を利用した翻訳
* キャッシュによる高速化

---

### 類似論文分析

* Sentence Transformers によるベクトル化
* コサイン類似度で類似論文抽出
* 各論文に対して **上位5件** を表示

---

### 論文マップ（可視化）

* t-SNE による2次元可視化
* 類似論文が近くに配置される
* クリックで詳細表示

---

### Discord連携

* `.env` にWebhookを設定するだけ
* ボタン1つで論文を送信
* 長文は自動分割対応

---

## ディレクトリ構成

```
.
├── app/                    # Web UI (Streamlit)
│   └── streamlit_paper_app.py
│
├── src/                    # コアロジック
│   ├── arxiv_keyword_search.py
│   ├── discord_notifier.py
│   ├── translator.py
│   └── paper_map_builder.py
│
├── outputs/                # 出力ファイル
│   └── paper_map.html
│
├── sandbox/                # テスト用コード
│
├── .env                    # 環境変数
└── README.md
```

---

## セットアップ

### 1. インストール

```bash
pip install -r app/requirements_streamlit_paper_app.txt
```

---

### 2. 環境変数設定

`.env` を作成

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxx
```

---

### 3. 起動

```bash
streamlit run app/streamlit_paper_app.py
```

---

## 使い方

### ① 論文検索

* カテゴリ（例: `cs.AI`）を入力
* 「論文検索」をクリック

---

### ② 日本語要約

* チェックONで自動翻訳
* 初回のみ時間がかかる

---

### ③ Discord送信

* ボタンを押すだけ
* `.env` のWebhookが自動使用される

---

### ④ 論文マップ

* 「論文マップ作成」をクリック
* 点をクリック → 詳細＋類似論文表示

---

## 技術スタック

| 分野    | 技術                    |
| ----- | --------------------- |
| フロント  | Streamlit             |
| データ取得 | arXiv API             |
| 翻訳    | LibreTranslate        |
| NLP   | Sentence Transformers |
| 可視化   | Plotly                |
| ML    | scikit-learn (t-SNE)  |

---

## 特徴

* 完全無料（API課金なし）
* ローカルLLMと組み合わせ可能
* 拡張しやすい構造（src/app分離）
* UI付きで直感的に操作可能

---

## 今後の拡張

* トピッククラスタリング
* 流行分析（時系列）
* キーワード自動抽出
* 論文比較（差分抽出）
* ローカルLLM統合

---

## 💡 コンセプト

> 「論文を読む」から「論文を俯瞰する」へ

---

## ライセンス

MIT License

---
