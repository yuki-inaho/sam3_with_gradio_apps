# EdgeTAM Gradio App 構造分析

**分析日時**: 2025-11-20 20:01:03 JST+0900
**目的**: SAM3 Sequence Tracker イベントハンドラー実装の参考

---

## 1. EdgeTAM 概要

EdgeTAMは2つのGradioアプリケーションを提供：
1. **multi_image_app.py**: 複数画像（画像シーケンス）をアップロードして追跡
2. **gradio_app.py**: 動画ファイルをアップロードして追跡

### Gradioバージョン情報

- **EdgeTAM**: Gradio 4.44.0
  - 確認方法: `cd /home/inaho-omen/Project/EdgeTAM && uv pip show gradio`
  - 依存: gradio-image-prompter（カスタムコンポーネント）
- **SAM3 (本プロジェクト)**: Gradio 5.49.1
  - 主な違い:
    - API変更の可能性あり（イベントハンドラー、コンポーネント仕様）
    - Gradio 5.x では gr.Image の select イベント動作が異なる可能性（要検証）
    - BBoxAnnotator（gradio-bbox-annotator==0.1.1）はGradio 5.49.1で動作確認済み

---

## 2. multi_image_app.py 構造分析

### 主要関数

| 関数名 | 用途 | 入力 | 出力 | 備考 |
|-------|------|------|------|------|
| `preprocess_images` | 画像アップロード処理 | image_files, session_state | points_map, output_image, gallery, hsv_info, session_state | 複数画像 → フレーム化、最初のフレーム表示 |
| `segment_with_points` | 点クリック処理 | point_type, session_state | points_map, output_image, session_state | include/exclude点追加、即座にセグメント |
| `propagate_to_all` | Track ボタン処理 | session_state | output_gallery, hsv_info, session_state | 全フレーム追跡、マスク生成、HSVパラメータ抽出 |
| `clear_points` | Clear Points ボタン | session_state | points_map, output_image, gallery, hsv_info, session_state | プロンプトのみクリア |
| `reset` | Reset ボタン | session_state | images_in, drawer, points_map, output_image, gallery, hsv_info, session_state | 完全リセット |
| `download_tracked_masks` | マスクダウンロード | session_state | download_file | ZIPファイル生成 |

### イベントバインディング

```python
# 画像アップロード
images_in.upload(
    fn=preprocess_images,
    inputs=[images_in, session_state],
    outputs=[images_in_drawer, points_map, output_image, output_gallery, hsv_info, session_state],
    queue=False,
)

# 画像クリック（点追加）
points_map.select(
    fn=segment_with_points,
    inputs=[point_type, session_state],
    outputs=[points_map, output_image, session_state],
    queue=False,
)

# Track ボタン
propagate_btn.click(
    fn=propagate_to_all,
    inputs=[session_state],
    outputs=[output_gallery, hsv_info, session_state],
    queue=True,  # 重要: 長時間処理のためqueue=True
)

# Clear Points ボタン
clear_points_btn.click(
    fn=clear_points,
    inputs=session_state,
    outputs=[points_map, output_image, output_gallery, hsv_info, session_state],
    queue=False,
)

# Reset ボタン
reset_btn.click(
    fn=reset,
    inputs=session_state,
    outputs=[images_in, images_in_drawer, points_map, output_image, output_gallery, hsv_info, session_state],
    queue=False,
)
```

### session_state 構造

```python
session_state = {
    "first_frame": None,        # 最初のフレーム（PIL Image）
    "all_frames": None,         # 全フレーム（ndarray list）
    "input_points": [],         # 点プロンプト座標リスト
    "input_labels": [],         # 点ラベルリスト（1=include, 0=exclude）
    "inference_state": None,    # SAM2推論状態
    "image_paths": [],          # 画像パスリスト
    "temp_video_path": None,    # 一時動画ファイルパス
    "hsv_params": None,         # HSV色空間パラメータ
    "tracked_masks": None,      # 追跡結果マスク
}
```

---

## 3. gradio_app.py 構造分析

### 主要関数

| 関数名 | 用途 | 入力 | 出力 | 備考 |
|-------|------|------|------|------|
| `get_video_fps` | 動画FPS取得 | video_path | fps | cv2.VideoCapture使用 |
| `preprocess_video_in` | 動画アップロード処理 | video_path, session_state | video, points_map, session_state | cv2で全フレーム抽出 |
| `segment_with_points` | 点クリック処理 | point_type, session_state | points_map, output_image, session_state | 最初のフレームに点追加 |
| `propagate_to_all` | Track ボタン処理 | session_state | output_video, session_state | 全フレーム追跡、動画生成 |
| `clear_points` | Clear Points ボタン | session_state | points_map, output_image, session_state | プロンプトクリア |
| `reset` | Reset ボタン | session_state | video, points_map, output_image, session_state | 完全リセット |

### イベントバインディング

```python
# 動画アップロード
video.upload(
    fn=preprocess_video_in,
    inputs=[video, session_state],
    outputs=[video, points_map, session_state],
    queue=False,
)

# 画像クリック
points_map.select(
    fn=segment_with_points,
    inputs=[point_type, session_state],
    outputs=[points_map, output_image, session_state],
    queue=False,
)

# Track ボタン
propagate_btn.click(
    fn=propagate_to_all,
    inputs=[session_state],
    outputs=[output_video, session_state],
    queue=True,
)
```

---

## 4. SAM3 SequenceTrackerApp への適用

### 既存実装（SequenceTrackerApp）

SequenceTrackerApp は以下のメソッドを既に実装済み：

- `set_video(video_path)`: 動画メタデータ取得
- `set_prompt_frame(frame_index)`: 指定フレーム抽出
- `add_point(x, y, label)`: 点プロンプト追加
- `add_box(x, y, w, h)`: 矩形プロンプト追加
- `preview_tracking(num_frames)`: プレビュー追跡（10フレーム程度）
- `run_tracking()`: 全フレーム追跡
- `run_tracking_stream()`: ストリーミング追跡（プログレスバー対応）
- `export_masks_zip()`: マスクZIPエクスポート
- `export_tracks_json()`: トラッキング情報JSONエクスポート

### 実装すべきイベントハンドラー（手順4-6）

EdgeTAMの知見を基に、以下のハンドラーを実装：

#### 4-6-1: 動画/画像シーケンスアップロード処理
- **EdgeTAM参考**: `preprocess_video_in` / `preprocess_images`
- **実装内容**:
  - `input_type` Radio変更時: input_video/input_images の visibility 切り替え
  - 動画アップロード時: `set_video()` 呼び出し、メタデータ表示、スライダー更新
  - 画像シーケンスアップロード時: 画像→動画変換、`set_video()` 呼び出し
- **出力**: video_metadata, frame_slider更新

#### 4-6-2: フレーム抽出処理
- **EdgeTAM参考**: preprocess後の最初のフレーム表示
- **実装内容**:
  - Extract Frameボタンクリック時: `set_prompt_frame(frame_index)` 呼び出し
  - 抽出したフレームを prompt_frame_image に表示
- **出力**: prompt_frame_image更新

#### 4-6-3: プロンプト追加処理
- **EdgeTAM参考**: `segment_with_points`
- **実装内容**:
  - prompt_frame_image クリック時: `add_point()` 呼び出し（Point mode）
  - BBoxAnnotator 変更時: `add_box()` 呼び出し（Box mode）
  - オーバーレイ画像更新
- **出力**: prompt_frame_image更新

#### 4-6-4: Previewボタン処理
- **EdgeTAM参考**: `propagate_to_all` の軽量版
- **実装内容**:
  - Previewボタンクリック時: `preview_tracking(num_frames=10)` 呼び出し
  - 短い動画を output_video に表示
- **出力**: output_video更新

#### 4-6-5: Trackボタン処理（プログレスバー付き）
- **EdgeTAM参考**: `propagate_to_all` with `queue=True`
- **実装内容**:
  - Trackボタンクリック時: `run_tracking_stream()` 呼び出し（generator）
  - gr.Progress でプログレスバー表示
  - 完了後、結果動画を output_video に表示
- **重要**: `queue=True`, `concurrency_limit=1` 設定
- **出力**: output_video, download_masks_button, download_tracks_button の visibility更新

#### 4-6-6: ダウンロードボタン処理
- **EdgeTAM参考**: `download_tracked_masks`
- **実装内容**:
  - Download Masks ボタン: `export_masks_zip()` 呼び出し
  - Download Tracks ボタン: `export_tracks_json()` 呼び出し
- **出力**: download_masks_button, download_tracks_button にファイルパス設定

---

## 5. 重要な実装ポイント

### Point Type（include/exclude）
EdgeTAMでは `gr.Radio` で "include"/"exclude" を選択。SAM3でも同様の実装が必要（Image Predictorでは既に実装済み）。

### queue設定
- **短時間処理**: `queue=False`（画像アップロード、点追加、クリアなど）
- **長時間処理**: `queue=True`（Track、全フレーム追跡）

### session_state vs self.sequence_tracker
EdgeTAMは `gr.State` で状態管理。SAM3では `self.sequence_tracker` に状態を保持。イベントハンドラーは `self.sequence_tracker` のメソッドを呼び出すラッパーとして実装。

### プログレスバー実装
```python
def _on_track_video(self, progress=gr.Progress()):
    for update in self.sequence_tracker.run_tracking_stream():
        if update["type"] == "progress":
            progress(update["progress"], desc=update["desc"])
        elif update["type"] == "result":
            return update["video_path"]
```

### 動画/画像シーケンス切り替え
- `input_type` Radioで切り替え
- 画像シーケンスは一時動画ファイルに変換してから処理

---

## 6. テスト戦略

EdgeTAMにはテストコードが存在しないため、SAM3では厳格なTDDアプローチを継続：

1. 各イベントハンドラーごとにテストケース作成
2. モックを使用して SequenceTrackerApp のメソッド呼び出しを検証
3. UIコンポーネントの更新を gr.update() で検証
4. プログレスバーのテストは generator の yield を検証

---

## 参考ファイル

- `/home/inaho-omen/Project/EdgeTAM/multi_image_app.py`: 画像シーケンス版（1088行）
- `/home/inaho-omen/Project/EdgeTAM/gradio_app.py`: 動画版（約350行）
- Gradio version: 4.44.0（SAM3は5.49.1使用、互換性注意）

---

## まとめ

EdgeTAMの構造は以下の流れ：

1. **アップロード** → メタデータ表示、最初のフレーム表示
2. **点追加** → 即座にセグメンテーション、オーバーレイ表示
3. **Track** → 全フレーム追跡、動画/ギャラリー表示
4. **ダウンロード** → ZIPファイル生成

SAM3では、この流れを SequenceTrackerApp の既存メソッドを活用して実装する。
