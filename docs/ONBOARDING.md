# 環境セットアップ・オンボーディングガイド

**作成日**: `2025-11-21`
**対象**: `新規セッション / エージェント / 開発メンバー`
**プロジェクト**: `sam3_with_gradio_apps`
**目的**: `同一環境を素早く再現し、作業を継続できる状態を共有する`

---

## 目次
1. [プロジェクト概要](#1-プロジェクト概要)
2. [現在のプロジェクト状態](#2-現在のプロジェクト状態)
3. [前提条件の確認](#3-前提条件の確認)
4. [環境セットアップ手順](#4-環境セットアップ手順)
5. [動作確認](#5-動作確認)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [次のステップ](#7-次のステップ)
8. [環境セットアップ完了チェックリスト](#8-環境セットアップ完了チェックリスト)
9. [更新履歴](#9-更新履歴)

---

## 1. プロジェクト概要

### プロジェクト名
**sam3_with_gradio_apps**
`SAM3モデルのGradioデモ（既存UIとHF互換UI）、評価スクリプト、付随ツールを含むPythonプロジェクト`

### 最終目標
`SAM3の画像/動画推論デモと関連ユーティリティを安定稼働させ、CPU/GPU環境で再現可能なテストフローを整備する。`

### 主要コンポーネント
* Gradio UI: `scripts/gradio_app/`（Image/Sequenceタブ）と暫定HF互換UI `video_app_hf_original.py`
* モデル管理: `sam3.model_builder`, `Sam3ModelManager`
* 推論ロジック: `sam3/model/*`、評価スクリプト `scripts/eval/*`
* テスト: `tests/`（ベッドルーム動画スモークなど）

---

## 2. 現在のプロジェクト状態

### 完了済み
| 分類 | 状態 | 説明 |
| --- | --- | --- |
| 依存定義 | 🟢 | `pyproject.toml`/`uv.lock` で依存管理済み |
| 作業計画書 | 🟢 | `docs/nov21_2025_refactor_plan.md` に現行リファクタ方針を記載 |
| テスト | 🟡 | `tests/test_bedroom_inference.py`（CUDAでパス、CPUでも実行可）
| ドキュメント | 🟡 | 本ONBOARDINGと計画書を追加済み |

### 依存パッケージのインストール状態
新規セッションでは未インストールの可能性があるため、4章の `uv sync` を必ず実行してください。

### 未実装・これから着手する項目
- Gradio UI一本化（HF互換UIの統合/廃止）
- dtype混在クラッシュ防止の恒久対応（bf16/float32ポリシー整理）
- CPU向け短縮テストの整備

### 重要なファイル／ディレクトリ（抜粋）
```
/home/inaho/Project/sam3_with_gradio_apps/
├── README.md
├── justfile
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── ONBOARDING.md                 # 本ドキュメント
│   └── nov21_2025_refactor_plan.md   # リファクタ計画書
├── scripts/gradio_app/               # 既存Gradio UI
├── video_app_hf_original.py          # HF互換UI（要統合・廃止予定）
├── tests/                            # テスト類
└── models/sam3.pt                    # チェックポイント（必要なら配置）
```

---

## 3. 前提条件の確認

### 3.1 システム情報の確認
```bash
cat /etc/os-release | grep -E "^(NAME|VERSION)="
uname -r
nproc
free -h
pwd
```

### 3.2 必須ツールの存在確認
```bash
which uv && uv --version
which just && just --version
```

### 3.3 Git ブランチ・コミット確認
```bash
git branch --show-current
git log --oneline -1
```

---

## 4. 環境セットアップ手順

### 4.1 システムパッケージ確認（必要に応じて）
```bash
apt-get update -qq
# 必要なら追加: apt-get install -y -qq build-essential libgl1-mesa-dev etc.
# Playwright（ヘッドレスE2E）に必要なランタイム
apt-get install -y -qq \
  wget ca-certificates libnss3 libatk1.0-0 libatk-bridge2.0-0 \
  libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
  libxrandr2 libgbm1 libpango-1.0-0 libcairo2 xvfb
```

### 4.2 Python 依存パッケージのインストール（最重要）
```bash
uv sync --extra dev
# Playwrightを使う場合（必要に応じて）
uv pip install playwright pytest pytest-asyncio
```

### 4.3 特殊環境（必要なら）
- GUIが必要な場合: Xvfb 起動例
```bash
Xvfb :99 -screen 0 1280x800x24 > /tmp/xvfb.log 2>&1 &
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
```
- Playwrightのブラウザバイナリを取得する場合
```bash
uv run playwright install-deps chromium
uv run playwright install chromium
```

### 4.4 環境変数の永続化（任意）
```bash
echo 'export DISPLAY=:99' >> ~/.bashrc
echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. 動作確認

### 5.1 依存・ツール確認
```bash
uv pip list | head -20
uv run pytest --version | grep pytest
```

### 5.2 最小スモーク（CUDAあれば実行）
```bash
uv run pytest tests/test_bedroom_inference.py -q
```

### 5.3 just レシピ表示
```bash
just --list
```

---

## 6. トラブルシューティング

- **コマンドが見つからない**: `uv sync --extra dev` を実行。PATHに `~/.local/bin` が通っているか確認。
- **CUDA初期化エラー**: CPUで実行するか、GPUドライバ/環境を確認。dtype混在が疑われる場合はfloat32統一を検討。
- **DISPLAY/OpenGLエラー**: `export DISPLAY=:99`, `export LIBGL_ALWAYS_SOFTWARE=1` を設定。
- **依存取得失敗**: ネットワーク制限下では事前に`uv.lock`を用いたオフライン同期を検討。

---

## 7. 次のステップ

- `docs/nov21_2025_refactor_plan.md` を確認し、UI統合・dtype方針の実装に着手。
- CPU向け短縮テストの追加・調整。
- Gradio UI一本化に向けて `video_app_hf_original.py` の廃止計画を具体化。

---

## 8. 環境セットアップ完了チェックリスト
- [ ] システム情報確認済み
- [ ] uv/just の存在確認済み
- [ ] `uv sync --extra dev` 完了
- [ ] 主要ライブラリ import OK
- [ ] （可能なら）`tests/test_bedroom_inference.py` 実行
- [ ] 作業計画書（docs/nov21_2025_refactor_plan.md）の確認

---

## 9. 更新履歴
- `2025-11-21`: 初版作成

---

本ガイドは、新しいセッションや新規参加メンバーが迅速に同一環境を再現し、既存タスクを引き継ぐためのものです。環境セットアップで問題が発生した場合は、本ドキュメントのトラブルシューティングや作業計画書、リポジトリのissue/コミットログを参照してください。
