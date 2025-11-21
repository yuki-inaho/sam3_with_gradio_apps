"""Lightweight Gradio demo for SAM3 (image + sequence tabs).

Notes
-----
- 依存: torch 2.3.1+cu118, torchvision 0.18.1+cu118, gradio>=4, pillow, numpy, decord/opencv (動画入力時)。
- モデル: models/sam3.pt を手元に配置してください。無い場合は UI 上で警告して実行不可とします。
- CUDA が使える場合は自動で cuda を選択。それ以外は cpu/ローカルで動きます。
- シンプルさ優先の最小実装です。性能チューニングや緻密な例外処理は適宜拡張してください。

Why this file exists
--------------------
ドキュメントでまとめた設計を最低限のコードに起こし、動作イメージを持てるようにするための「サンプル実装」です。
プロンプトの渡し方やモデル初期化の流れ、Gradio での入出力の結線パターンを示します。
本番実装では、チェックポイントの有無/メモリ監視/ログ出力/よりリッチな可視化などを追加してください。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor


CKPT_PATH = Path("models/sam3.pt")


# -----------------------------------------------------------------------------
# サンプルプリセット（UI から選択して入力欄を埋めるためのデモ用データ）
# -----------------------------------------------------------------------------
IMAGE_PRESETS = {
    "Yellow bus": {
        "text": "a yellow bus",
        # 例: 640x480 の画像を想定した bbox（左上→右下, 前景ラベル=1）
        "boxes": [[120, 200, 520, 380, 1]],
    },
    "Striped cat": {
        "text": "a striped cat",
        "boxes": [[150, 150, 400, 420, 1]],
    },
    "No text (box only)": {
        "text": "",
        "boxes": [[100, 120, 360, 360, 1]],
    },
}

VIDEO_PRESETS = {
    "Person walk (fwd)": {
        "text": "a person",
        "box_text": "120,120,260,360",
        "direction": "forward",
    },
    "Car both": {
        "text": "car",
        "box_text": "80,160,320,320",
        "direction": "both",
    },
}


def device_pick() -> str:
    """Pick default device. Prefer CUDA if available."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelBundle:
    image_model: Optional[torch.nn.Module] = None
    image_processor: Optional[Sam3Processor] = None
    video_predictor: Optional[object] = None  # Sam3VideoPredictor
    device: str = device_pick()

    def ensure_image(self) -> Tuple[torch.nn.Module, Sam3Processor]:
        """Lazy-load image model + processor (singleton)."""
        if self.image_model is None:
            if not CKPT_PATH.exists():
                raise FileNotFoundError("models/sam3.pt が見つかりません")
            self.image_model = build_sam3_image_model(
                checkpoint_path=str(CKPT_PATH),
                load_from_HF=False,
                device=self.device,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
            self.image_processor = Sam3Processor(
                self.image_model, resolution=1008, device=self.device
            )
        return self.image_model, self.image_processor

    def ensure_video(self):
        """Lazy-load video predictor (singleton)."""
        if self.video_predictor is None:
            if not CKPT_PATH.exists():
                raise FileNotFoundError("models/sam3.pt が見つかりません")
            gpus = [torch.cuda.current_device()] if self.device == "cuda" else None
            self.video_predictor = build_sam3_video_predictor(
                checkpoint_path=str(CKPT_PATH),
                load_from_HF=False,
                gpus_to_use=gpus,
                compile=False,
            )
        return self.video_predictor


MODELS = ModelBundle()


def _normalize_box_xyxy(box_xyxy: List[float], h: int, w: int) -> List[float]:
    """Convert absolute xyxy -> normalized cxcywh for the processor."""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return [float(cx), float(cy), float(bw), float(bh)]


def run_image_prompt(
    image: np.ndarray,
    text_prompt: str,
    boxes_df: List[List[float]],
    score_thresh: float,
):
    """Image tab callback.

    1) 画像をセットし、テキスト・ボックスプロンプトを SAM3 に投入。
    2) confidence threshold を適用し、簡易オーバーレイを返す。
    """
    if image is None:
        return None, "画像をアップロードしてください"

    model, processor = MODELS.ensure_image()

    pil_image = Image.fromarray(image.astype(np.uint8))
    h, w = pil_image.height, pil_image.width

    state: Dict = {}
    state = processor.set_image(pil_image, state=state)

    if text_prompt:
        state = processor.set_text_prompt(text_prompt, state)
    else:
        # テキストなしの場合でも geometric prompt に進むため dummy text を設定
        state = processor.set_text_prompt("visual", state)

    # boxes_df は [[x1,y1,x2,y2,label], ...] を想定
    if boxes_df:
        geo_prompt = state.get("geometric_prompt")
        if geo_prompt is None:
            geo_prompt = model._get_dummy_prompt()
            state["geometric_prompt"] = geo_prompt
        for row in boxes_df:
            if len(row) < 5:
                continue
            x1, y1, x2, y2, label = row
            bbox = _normalize_box_xyxy([x1, y1, x2, y2], h=h, w=w)
            state = processor.add_geometric_prompt(bbox, bool(label), state)

    # 最終的に confidence_threshold を調整
    processor.confidence_threshold = score_thresh
    state = processor._forward_grounding(state)

    # 簡易オーバーレイ生成
    mask = state["masks"].float().squeeze(0).cpu().numpy()
    overlay = np.array(pil_image).astype(np.float32)
    color = np.array([0, 255, 0], dtype=np.float32)
    alpha = 0.45
    overlay = overlay * (1 - alpha * mask[..., None]) + color * (alpha * mask[..., None])
    overlay = overlay.clip(0, 255).astype(np.uint8)

    return overlay, "推論完了"


def run_video_tracking(
    video_path: str,
    text_prompt: str,
    box_prompt: List[float],
    direction: str,
    max_frames: int,
):
    """Video tab callback (簡易版).

    - 先頭フレームにテキスト/ボックスを与え、指定フレーム数まで伝播。
    - 可視化は省略し、進捗テキストを返すだけのプレースホルダ。
    """
    if not video_path:
        return None, "動画をアップロードしてください"

    predictor = MODELS.ensure_video()

    # start session
    session = predictor.start_session(video_path)
    session_id = session["session_id"]

    # add prompt (box_prompt: [x1,y1,x2,y2])
    predictor.add_prompt(
        session_id=session_id,
        frame_idx=0,
        text=text_prompt or None,
        bounding_boxes=[box_prompt] if box_prompt else None,
        bounding_box_labels=[1] if box_prompt else None,
    )

    frames = []
    for i, out in enumerate(
        predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction=direction,
            start_frame_idx=0,
            max_frame_num_to_track=max_frames,
        )
    ):
        frame_idx = out["frame_index"]
        # out["outputs"] は具体的なキーが例により異なる。ここでは画像表示は省略し、進捗のみ返す。
        frames.append(frame_idx)

    predictor.close_session(session_id)
    predictor.shutdown()
    return "", f"処理フレーム: {len(frames)}"  # プレースホルダ


def build_demo():
    with gr.Blocks(title="SAM3 Gradio Demo") as demo:
        gr.Markdown("## SAM3 Gradio Demo — Image & Sequence")

        if not CKPT_PATH.exists():
            gr.Markdown("**Warning**: models/sam3.pt が見つかりません。配置してください。")

        with gr.Tab("Image Predictor"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="Input Image")
                    text_prompt = gr.Textbox(label="Text prompt", placeholder="e.g., yellow bus")
                    preset_img = gr.Dropdown(
                        choices=list(IMAGE_PRESETS.keys()),
                        label="Load sample preset",
                        value=None,
                    )
                    boxes_df = gr.Dataframe(
                        headers=["x1", "y1", "x2", "y2", "label(1=fg,0=bg)"],
                        datatype=["number", "number", "number", "number", "number"],
                        row_count=(0, "dynamic"),
                        col_count=5,
                        label="Box prompts (optional)",
                    )
                    score_thresh = gr.Slider(0.0, 1.0, value=0.5, label="Confidence threshold")
                    run_btn = gr.Button("Run")
                    status = gr.Markdown()
                with gr.Column():
                    result_img = gr.Image(type="numpy", label="Mask overlay")

            def _apply_image_preset(name):
                if not name:
                    return gr.update(), gr.update()
                preset = IMAGE_PRESETS.get(name, {})
                text = preset.get("text", "")
                boxes = preset.get("boxes", [])
                return text, boxes

            preset_img.change(
                fn=_apply_image_preset,
                inputs=[preset_img],
                outputs=[text_prompt, boxes_df],
            )

            run_btn.click(
                fn=run_image_prompt,
                inputs=[img_input, text_prompt, boxes_df, score_thresh],
                outputs=[result_img, status],
            )

        with gr.Tab("Sequence Tracker"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Input video (mp4)")
                    text_prompt_v = gr.Textbox(label="Text prompt", placeholder="e.g., yellow bus")
                    preset_vid = gr.Dropdown(
                        choices=list(VIDEO_PRESETS.keys()),
                        label="Load sample preset",
                        value=None,
                    )
                    box_prompt_v = gr.Textbox(
                        label="Box prompt [x1,y1,x2,y2] (optional)", placeholder="e.g., 10,20,200,220"
                    )
                    direction = gr.Radio(
                        ["forward", "backward", "both"], value="forward", label="Direction"
                    )
                    max_frames = gr.Slider(1, 200, value=30, step=1, label="Max frames")
                    run_track = gr.Button("Run tracking")
                    status_v = gr.Markdown()
                with gr.Column():
                    preview = gr.Textbox(label="Result placeholder", interactive=False)

            def _parse_box(text: str):
                if not text:
                    return None
                try:
                    vals = [float(x.strip()) for x in text.split(",")]
                    return vals if len(vals) == 4 else None
                except Exception:
                    return None

            def _wrap_video(path, text, box_text, direction, max_frames):
                box = _parse_box(box_text)
                return run_video_tracking(path, text, box, direction, int(max_frames))

            run_track.click(
                fn=_wrap_video,
                inputs=[video, text_prompt_v, box_prompt_v, direction, max_frames],
                outputs=[preview, status_v],
            )

            def _apply_video_preset(name):
                if not name:
                    return gr.update(), gr.update(), gr.update()
                preset = VIDEO_PRESETS.get(name, {})
                return (
                    preset.get("text", ""),
                    preset.get("box_text", ""),
                    preset.get("direction", "forward"),
                )

            preset_vid.change(
                fn=_apply_video_preset,
                inputs=[preset_vid],
                outputs=[text_prompt_v, box_prompt_v, direction],
            )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
