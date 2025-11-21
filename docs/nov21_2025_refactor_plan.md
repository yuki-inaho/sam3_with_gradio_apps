# 作業計画書 兼 記録書

---

**日付：** `2025年11月21日`
**作業ディレクトリ・リポジトリ:** `sam3_with_gradio_apps`
**作業者：** `inaho`

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

*   **目標1:** Gradio UIの二重実装を解消し、既存`scripts/gradio_app`構造に統合した設計を固める。
*   **目標2:** dtype混在（Float×BF16）によるクラッシュを防ぐ設計方針を決定し、state変換の方針を策定する。
*   **目標3:** CPU環境でも再現・検証可能なテストフローを整備し、作業後の確認手順を明文化する。

---

## 2. 作業内容

### フェーズ 1: 調査・設計フェーズ (見積: 1.5h)
このフェーズでは、実装に着手する前の準備作業を行います。

1. **現状のアーキテクチャ分析：**
   * **タスク内容：** `video_app_hf_original.py` と `scripts/gradio_app`（`sam3_gradio_app.py`, `sequence_tracker_app.py`, `image_predictor_app.py`, `model_manager.py`）を精査し、重複責務と差分を洗い出す。
   * **目的：** UI/モデル管理の統合ポイントを特定し、移行計画を立案する。
   * **TDD**: 既存テスト `tests/test_bedroom_inference.py` を事前実行し、現状動作のベースラインを確認する。

2. **依存コンポーネントの確認：**
   * **タスク内容：** dtype周り（bf16/float32）とstate初期化 (`init_state`, `propagate_in_video`, `add_prompt`)、キャッシュ生成（`cached_frame_outputs`）の前提を、`sam3/model` コアコードで確認。
   * **目的：** 型混在クラッシュ防止のための統一ポリシー決定（例: float32固定 vs bf16統一＋autocast）。

3. **設計方針の文書化：**
   * **タスク内容：**
     * UI一本化方針：`video_app_hf_original.py`の責務を既存`Sam3GradioApp`側へ寄せる案を文書化。
     * dtypeポリシー：モデル/stateのキャスト手順とautocast利用有無を決める。
     * 初回cache生成：Tracker初回クリック時の`cached_frame_outputs`確保手順（正規APIでの一度きりのforwardなど）を決定。
   * **目的：** 実装ロードマップを明確化し、混乱や二重修正を防ぐ。

### フェーズ 2: 実装（リファクタリング方針の適用） (見積: 2.0h)
このフェーズでは、設計方針に基づいて主要なリファクタリングを実施します。CPU環境で作業し、`uv` 環境と `just` 利用を前提とします。

1. **UI統合タスク：**
   * **タスク内容：**
     - `video_app_hf_original.py`のロジックから重複部分を削り、`scripts/gradio_app/sam3_gradio_app.py`側へテキスト動画UIを統合する骨子を作る（コード移動方針の草案）。
     - 暫定で`video_app_hf_original.py`の起動を停止（エントリポイントを無効化または削除予定の明記）。
   * **目的：** 人的負債の低減と修正箇所の一元化。

2. **dtype/state統一タスク：**
   * **タスク内容：**
     - 統一dtypeをfloat32に固定（CPU前提で安全）。GPU時の方針は文書化し、作業後に切り替え可能とする。
     - stateキャストヘルパー（再帰的にTensorを`to(device, dtype)`）を共通化し、init後/propagate前に適用。
   * **目的：** Float/BF16混在クラッシュを防止し、処理一貫性を担保。

3. **初回cache生成タスク（Tracker）：**
   * **タスク内容：**
     - 正規API経由で初期cacheを作るラッパー関数を用意し、クリック前に1ステップのforwardを明示的に行うか、SDK側に初期化メソッドを追加。
     - UI側の「強制forward」ハックを除去。
   * **目的：** 安定した動作と責務分離。

### フェーズ 3: テストと動作検証 (見積: 1.5h)
最終フェーズでは、実装した変更をテストで検証します。`uv`環境での実行を前提とし、CPUでも回る軽量テストを優先。GPU検証は別途（可能なら実行）。

1. **単体/スモークテストの作成と実行：**
   * **タスク内容：** 既存テストを調整し、CPUでも完走する最小ケースを用意。`tests/test_bedroom_inference.py`（frame=5伝播）をベースに必要ならframe数をさらに削減。
   * **目的：** 主要フローが落ちないことを保証。

2. **簡易結合テスト（CUI）:**
   * **タスク内容：** CUIでの擬似Gradioフロー（テキスト/ポイント）をCPU向けに短縮して確認。テキストは1ステップ、ポイントはキャッシュ生成確認のみ。
   * **目的：** UI統合後の主要経路がクラッシュしないことを確認。

3. **品質チェック：**
   * **タスク内容：** `just format` / `just lint`（または`uv run black/ruff`）を実行し、フォーマット・Lintを通す。必要ならmypyは対象ファイル限定で`--ignore-missing-imports`を指定。
   * **目的：** コード規約遵守と回帰防止。

---

## 3. 作業チェックリスト（原子ステップ）

### フェーズ 1: 調査・設計フェーズ

### 手順 1: 既存UI・モデル管理の重複を洗い出す
- [ ] 🖐 **操作**: `uv run pytest tests/test_bedroom_inference.py -q` を実行し現状動作を確認。続いて `rg -n "Sam3GradioApp|SequenceTrackerApp|video_app_hf_original" scripts/gradio_app video_app_hf_original.py` で重複箇所をリストアップ。
- [ ] 🔎 **確認**: テストが通る（スキップ可）こと、重複箇所が一覧化できていること。
- [ ] 🧪 **テスト**: `tests/test_bedroom_inference.py`（現状パス/スキップ）。
- [ ] 🛠 **エラー時対処**: pytestが見つからない→`uv sync --extra dev`; 依存不足→`uv pip install` で補う。

### 手順 2: dtype/stateの現状把握
- [ ] 🖐 **操作**: `rg -n "bfloat16|float16|autocast" video_app_hf_original.py sam3/model` でdtype使用箇所を抽出し、state初期化 (`init_state`, `add_prompt`, `propagate_in_video`) の型を確認。
- [ ] 🔎 **確認**: dtype混在の原因となる箇所を特定（bf16ウェイト×float入力）。
- [ ] 🧪 **テスト**: なし（設計調査）。
- [ ] 🛠 **エラー時対処**: 該当箇所が見つからない場合は`sed -n`で周辺コードを直接確認。

### 手順 3: 設計方針の文書化
- [ ] 🖐 **操作**: `docs/nov21_2025_refactor_plan.md` にUI統合案、dtype統一案、初回cache生成案を書く（本書を更新）。
- [ ] 🔎 **確認**: 3案とも具体的な移行ステップが文章化されている。
- [ ] 🧪 **テスト**: なし（ドキュメント）。
- [ ] 🛠 **エラー時対処**: 記述漏れがあれば追記。

### フェーズ 2: 実装（リファクタリング方針の適用）

### 手順 4: `video_app_hf_original.py` のエントリポイント停止（統合準備）
- [ ] 🖐 **操作**: 起動部をコメントアウトまたは警告表示に差し替え、`python video_app_hf_original.py` でGradioを起動しないようにする（廃止予定の明記）。
- [ ] 🔎 **確認**: 実行してもサーバを立てずに終了する／警告のみで終わる。
- [ ] 🧪 **テスト**: `uv run python video_app_hf_original.py --help` が正常終了すること。
- [ ] 🛠 **エラー時対処**: SyntaxError等が出たら末尾の`__main__`ブロックを再確認。

### 手順 5: dtype/state統一ユーティリティの追加
- [ ] 🖐 **操作**: 共通ヘルパー（例: `_cast_state(obj, device, dtype)`）を定義し、init直後・propagate前に適用。モデルは`to(device, dtype=torch.float32)`で統一。
- [ ] 🔎 **確認**: dtype = float32 に統一され、state内のTensorも同じdtype/deviceになっているログを確認（任意でprint/log）。
- [ ] 🧪 **テスト**: `uv run pytest tests/test_bedroom_inference.py -q`（CPUで可）。
- [ ] 🛠 **エラー時対処**: 型混在エラーが出たら未キャスト箇所を追加キャスト。

### 手順 6: Tracker初回cache生成を正規APIで実施
- [ ] 🖐 **操作**: SequenceTracker側の初期化に「初回forwardでcache生成するラッパー」を追加し、UI側の強制forwardを削除。
- [ ] 🔎 **確認**: `cached_frame_outputs`関連アサートが発生しないこと。
- [ ] 🧪 **テスト**: 既存スモーク（Point/Box相当の簡易CUIテスト）を必要なら短縮して実行。
- [ ] 🛠 **エラー時対処**: アサートが残る場合、cache生成の呼び出し位置をinit直後に修正。

### 手順 7: UI統合に向けた骨子作成
- [ ] 🖐 **操作**: `scripts/gradio_app/sam3_gradio_app.py` にText動画タブの骨子を追加（実装は後日でも、プレースホルダとTODO明記）。`video_app_hf_original.py`の残余ロジックは削減。
- [ ] 🔎 **確認**: Gradio Blocksが1系統にまとまり、冗長なイベント/状態が減っていること。
- [ ] 🧪 **テスト**: `uv run pytest tests/test_bedroom_inference.py -q`（最低限の回帰）。
- [ ] 🛠 **エラー時対処**: UI依存エラーが出た場合、一旦プレースホルダ実装に戻してビルドを通す。

### フェーズ 3: テストと動作検証

### 手順 8: CPU用スモークテストの短縮・実行
- [ ] 🖐 **操作**: テストのフレーム数や処理ステップを最小限に調整（必要なら`bedroom`より小さいサンプルを用意）。`uv run pytest tests/test_bedroom_inference.py -q`を実行。
- [ ] 🔎 **確認**: CPUでタイムアウトせず完走すること。
- [ ] 🧪 **テスト**: 上記pytest。
- [ ] 🛠 **エラー時対処**: タイムアウト→フレーム数削減／動画を短尺に差し替え。

### 手順 9: 型チェック（限定範囲）
- [ ] 🖐 **操作**: `uv run mypy video_app_hf_original.py scripts/gradio_app/sam3_gradio_app.py --ignore-missing-imports` を実行（対象を限定）。
- [ ] 🔎 **確認**: 対象ファイルでの致命的な型エラーが解消されていること。
- [ ] 🧪 **テスト**: 上記mypy実行。
- [ ] 🛠 **エラー時対処**: 外部スタブ不足は`--ignore-missing-imports`、自前未注釈は必要最小限で追加。

### 手順 10: フォーマット・Lint
- [ ] 🖐 **操作**: `just format`（または`uv run black ...`）と`just lint`（または`uv run ruff ...`）を実行。
- [ ] 🔎 **確認**: フォーマット差分が消え、Lintエラーがないこと。
- [ ] 🧪 **テスト**: 上記コマンド。
- [ ] 🛠 **エラー時対処**: 自動修正できないLintは該当箇所を手修正。

---

## 4. 作業に使用するコマンド参考情報
- 依存同期: `uv sync --extra dev`
- テスト実行: `uv run pytest <path> -q`
- フォーマット: `just format` もしくは `uv run black ...`
- Lint: `just lint` もしくは `uv run ruff ...`
- 型チェック（限定）: `uv run mypy <files> --ignore-missing-imports`

---

## 6. 完了の定義
- [ ] UIが一本化され、廃止予定フローが停止している
- [ ] dtypeポリシーが明記され、主要パスで型混在エラーが発生しない
- [ ] CPUスモークテストが完走する
- [ ] フォーマット・Lintを通過する

---

## 7. 作業記録

**重要な注意事項：**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
*   コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
*   エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| `[YYYY-MM-DD]` | `[HH:MM:SS TZ]` | `[作業者名]` | フェーズ1開始: `[タスク名]` | 作業計画書確認完了、`[タスク]`の要件を把握 |
| `[YYYY-MM-DD]` | `[HH:MM:SS TZ]` | `[作業者名]` | `[ファイル名]`の実装状況確認 | **重要発見**: `[問題の原因や新しい発見、想定外の仕様など]` |
| `[YYYY-MM-DD]` | `[HH:MM:SS TZ]` | `[作業者名]` | フェーズ1完了: `[タスク名]` | `[設計方針が固まる、実装不要と判明するなど]` |
| `[YYYY-MM-DD]` | `[HH:MM:SS TZ]` | `[作業者名]` | フェーズ2開始: `[タスク名]` | `[実行コマンド]` で開発環境を起動 |
| `[YYYY-MM-DD]` | `[HH:MM:SS TZ]` | `[作業者名]` | `[具体的な実装作業]` | ✅成功: `[期待した結果]` / ❌失敗: `[エラーメッセージをここに貼る]` |
| `[YYYY-MM-DD]` | `[HH:MM:SS TZ]` | `[作業者名]` | エラー修正: `[エラーの原因特定]` | ✅解決: `[解決策や適用した修正]`を適用し、再テストで成功 |
| `[YYYY-MM-DD]` | `[HH:MM:SS TZ]` | `[作業者名]` | `[作業内容]` | ✅全体成功: `[その日の最終的な成果]` |
| | | | | |
