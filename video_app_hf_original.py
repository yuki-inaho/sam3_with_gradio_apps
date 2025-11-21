"""
https://huggingface.co/spaces/merve/SAM3-video-segmentation
をベースとしたコード
HF Spaces版 SAM3 Video Gradio デモのUIを保ちつつ、ローカル sam3 実装で動く互換レイヤーを内蔵したスクリプト。
外部のtransformers実装には依存せず、このファイル単体で完結する。
"""

import argparse
import colorsys
import gc
import os
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from gradio.themes import Soft
from PIL import Image, ImageDraw, ImageFont

from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from sam3.model_builder import build_sam3_video_model


# ------------------------------------------------------------
# デバイス/チェックポイント設定
# ------------------------------------------------------------
DEFAULT_CKPT = os.getenv("SAM3_CKPT", "models/sam3.pt")


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 安定性を優先し、推論はfloat32に強制
    dtype = torch.float32
    return device, dtype


# ------------------------------------------------------------
# 互換ラッパー
# ------------------------------------------------------------
class LocalSession(SimpleNamespace):
    """UIで使う属性を保持する簡易セッションオブジェクト."""

    def __init__(self, session_id: str, width: int, height: int, num_frames: int):
        super().__init__(
            session_id=session_id,
            video_width=width,
            video_height=height,
            num_frames=num_frames,
            obj_ids=[],
            text_prompt=None,
        )


class _BaseProcessor:
    def __init__(self, predictor: Sam3VideoPredictor):
        self.predictor = predictor

    def init_video_session(
        self,
        *,
        video_path: str,
        width: int,
        height: int,
        num_frames: int,
        **__,
    ) -> LocalSession:
        # Sam3VideoPredictorはリソースパスからセッション開始する
        resp = self.predictor.start_session(video_path)
        return LocalSession(resp["session_id"], width, height, num_frames)

    @staticmethod
    def _normalize_points(points, width: int, height: int):
        normed = []
        for p in points:
            x, y = p
            normed.append([x / width, y / height])
        return normed

    @staticmethod
    def _normalize_box(box, width: int, height: int):
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        return [x1 / width, y1 / height, w / width, h / height]


class Sam3TrackerVideoProcessor(_BaseProcessor):
    def add_inputs_to_inference_session(
        self,
        *,
        inference_session: LocalSession,
        frame_idx: int,
        obj_ids: int,
        input_boxes=None,
        input_points=None,
        input_labels=None,
    ):
        # input_boxes: [[[x1,y1,x2,y2]]]
        # input_points: [[[[x,y],...]]], input_labels: [[[0/1,...]]]
        boxes_xywh = None
        box_labels = None
        points = None
        point_labels = None

        if input_boxes is not None:
            flat = input_boxes[0][0]
            boxes_xywh = [self._normalize_box(flat, inference_session.video_width, inference_session.video_height)]
            box_labels = [1]

        if input_points is not None and input_labels is not None:
            pts = input_points[0][0]
            labs = input_labels[0][0]
            points = [self._normalize_points(pts, inference_session.video_width, inference_session.video_height)]
            point_labels = [int(l) for l in labs]

        # obj_ids は単一IDを想定
        obj_id = int(obj_ids)
        inference_session.obj_ids = list({*inference_session.obj_ids, obj_id})

        # Sam3VideoPredictorのadd_promptを利用
        self.predictor.add_prompt(
            session_id=inference_session.session_id,
            frame_idx=frame_idx,
            text=None,
            points=points[0] if points else None,
            point_labels=point_labels,
            bounding_boxes=boxes_xywh,
            bounding_box_labels=box_labels,
            obj_id=obj_id,
        )

    def post_process_masks(self, masks_list, original_sizes, binarize=False):
        # masks_list: list of torch.Tensor [N,H,W]
        outputs = []
        for m in masks_list:
            arr = m.float().cpu().numpy()
            if binarize:
                arr = (arr > 0).astype(np.float32)
            outputs.append(arr)
        return outputs


class Sam3TrackerVideoModel:
    @classmethod
    def from_pretrained(cls, *_, **__):
        # Sam3VideoPredictorは内部でモデルをロード
        predictor = Sam3VideoPredictor(checkpoint_path=DEFAULT_CKPT)
        return cls(predictor)

    def __init__(self, predictor: Sam3VideoPredictor):
        self.predictor = predictor

    def eval(self):
        return self

    def __call__(self, *, inference_session: LocalSession, frame_idx: int):
        # 1フレーム分だけ前進伝播
        gen = self.predictor.propagate_in_video(
            session_id=inference_session.session_id,
            propagation_direction="forward",
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
        )
        first = next(iter(gen), None)
        if first is None:
            raise RuntimeError("No outputs from propagate_in_video")
        obj_ids = first["outputs"]["out_obj_ids"]
        masks = first["outputs"]["out_binary_masks"]  # bool array [N,H,W]
        inference_session.obj_ids = list(obj_ids)
        # とりあえずscoreは0配列に
        out = SimpleNamespace(
            frame_idx=first["frame_index"],
            pred_masks=torch.tensor(masks.astype(np.float32)),
            scores=torch.zeros(len(obj_ids)),
        )
        return out

    def propagate_in_video_iterator(self, *, inference_session: LocalSession):
        for item in self.predictor.propagate_in_video(
            session_id=inference_session.session_id,
            propagation_direction="forward",
            start_frame_idx=0,
            max_frame_num_to_track=inference_session.num_frames,
        ):
            obj_ids = item["outputs"]["out_obj_ids"]
            masks = item["outputs"]["out_binary_masks"]
            inference_session.obj_ids = list(obj_ids)
            yield SimpleNamespace(
                frame_idx=item["frame_index"],
                pred_masks=torch.tensor(masks.astype(np.float32)),
                scores=torch.zeros(len(obj_ids)),
            )


class Sam3VideoProcessor(_BaseProcessor):
    def add_text_prompt(self, *, inference_session: LocalSession, text: str):
        inference_session.text_prompt = text.strip() if text else None
        return inference_session

    def postprocess_outputs(self, inference_session: LocalSession, model_outputs):
        # model_outputs: SimpleNamespace(frame_idx, masks, scores, obj_ids)
        return {
            "object_ids": model_outputs.object_ids,
            "masks": model_outputs.masks,
            "scores": model_outputs.scores,
        }


class Sam3VideoModel:
    @classmethod
    def from_pretrained(cls, *_, **__):
        # Sam3VideoInferenceはbuild_sam3_video_modelから構築（CPU/GPU自動判定）
        device, _ = get_device_and_dtype()
        dtype = torch.float32
        model = build_sam3_video_model(
            checkpoint_path=DEFAULT_CKPT,
            device=device,
            load_from_HF=False,
            apply_temporal_disambiguation=True,
            compile=False,
        )
        return cls(model, device, dtype)

    def __init__(self, model, device, dtype):
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device = device
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        return self

    def to(self, device, dtype=None):
        self.device = device
        target_dtype = dtype if dtype is not None else self.dtype
        self.model = self.model.to(device=device, dtype=target_dtype)
        return self

    def eval(self):
        return self

    def propagate_in_video_iterator(
        self,
        *,
        inference_session: LocalSession,
        start_frame_idx: int,
        max_frame_num_to_track: int,
    ):
        text = inference_session.text_prompt
        if text:
            # textをセット (frame_idxに依存しないため0でOK)
            self.model.add_prompt(
                inference_session._state,  # set later
                frame_idx=start_frame_idx,
                text_str=text,
                boxes_xywh=None,
                box_labels=None,
            )

        for frame_idx, outputs in self.model.propagate_in_video(
            inference_state=inference_session._state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
        ):
            # outputs は Sam3VideoInference._postprocess_output の戻り値
            obj_ids = outputs["out_obj_ids"]
            masks = outputs["out_binary_masks"].astype(np.float32)
            scores = outputs["out_probs"].astype(np.float32) if "out_probs" in outputs else np.zeros(len(obj_ids))
            yield SimpleNamespace(
                frame_idx=frame_idx,
                object_ids=obj_ids,
                masks=masks,
                scores=scores,
            )


# ------------------------------------------------------------
# グローバル初期化
# ------------------------------------------------------------
device, dtype = get_device_and_dtype()

# Tracker系はSam3VideoPredictorを使用
_GLOBAL_TRACKER_MODEL = Sam3TrackerVideoModel.from_pretrained()
_GLOBAL_TRACKER_PROCESSOR = Sam3TrackerVideoProcessor(_GLOBAL_TRACKER_MODEL.predictor)

# Text系はSam3VideoModelをそのまま利用
_GLOBAL_TEXT_VIDEO_MODEL = Sam3VideoModel.from_pretrained()
_GLOBAL_TEXT_VIDEO_PROCESSOR = Sam3VideoProcessor(Sam3VideoPredictor(checkpoint_path=DEFAULT_CKPT))
print(f"Models loaded on {device} (dtype={dtype})")


# ------------------------------------------------------------
# 既存UIロジック
# ------------------------------------------------------------
def try_load_video_frames(video_path_or_url: str) -> tuple[list[Image.Image], dict]:
    frames: list[Image.Image] = []
    fps_val: Optional[float] = None

    path = Path(video_path_or_url)
    if path.is_dir():
        # ディレクトリにある連番画像を動画として扱う
        imgs = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        for img_path in imgs:
            try:
                frames.append(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
        fps_val = None  # 不明なのでNone
    else:
        cap = cv2.VideoCapture(video_path_or_url)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    info = {
        "num_frames": len(frames),
        "fps": float(fps_val) if fps_val and fps_val > 0 else None,
    }
    return frames, info


def overlay_masks_on_frame(
    frame: Image.Image,
    masks_per_object: dict[int, np.ndarray],
    color_by_obj: dict[int, tuple[int, int, int]],
    alpha: float = 0.5,
) -> Image.Image:
    base = np.array(frame).astype(np.float32) / 255.0
    overlay = base.copy()

    for obj_id, mask in masks_per_object.items():
        if mask is None:
            continue
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask = np.clip(mask, 0.0, 1.0)
        color = np.array(color_by_obj.get(obj_id, (255, 0, 0)), dtype=np.float32) / 255.0
        a = alpha
        m = mask[..., None]
        overlay = (1.0 - a * m) * overlay + (a * m) * color

    out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def pastel_color_for_object(obj_id: int) -> tuple[int, int, int]:
    golden_ratio_conjugate = 0.61
    hue = (obj_id * golden_ratio_conjugate) % 1.0
    saturation = 0.45
    value = 1.0
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r_f * 255), int(g_f * 255), int(b_f * 255)


class AppState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_frames: list[Image.Image] = []
        self.inference_session = None
        self.video_fps: float | None = None
        self.masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        self.color_by_obj: dict[int, tuple[int, int, int]] = {}
        self.clicks_by_frame_obj: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        self.boxes_by_frame_obj: dict[int, dict[int, list[tuple[int, int, int, int]]]] = {}
        self.text_prompts_by_frame_obj: dict[int, dict[int, str]] = {}
        self.composited_frames: dict[int, Image.Image] = {}
        self.current_frame_idx: int = 0
        self.current_obj_id: int = 1
        self.current_label: str = "positive"
        self.current_clear_old: bool = True
        self.current_prompt_type: str = "Points"
        self.pending_box_start: tuple[int, int] | None = None
        self.pending_box_start_frame_idx: int | None = None
        self.pending_box_start_obj_id: int | None = None
        self.active_tab: str = "point_box"

    def __repr__(self):
        return f"AppState(video_frames={len(self.video_frames)}, video_fps={self.video_fps}, masks_by_frame={len(self.masks_by_frame)}, color_by_obj={len(self.color_by_obj)})"

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)


def init_video_session(
    GLOBAL_STATE: gr.State, video: str | dict, active_tab: str = "point_box"
) -> tuple[AppState, int, int, Image.Image, str]:
    GLOBAL_STATE.video_frames = []
    GLOBAL_STATE.masks_by_frame = {}
    GLOBAL_STATE.color_by_obj = {}
    GLOBAL_STATE.text_prompts_by_frame_obj = {}
    GLOBAL_STATE.clicks_by_frame_obj = {}
    GLOBAL_STATE.boxes_by_frame_obj = {}
    GLOBAL_STATE.composited_frames = {}
    GLOBAL_STATE.inference_session = None
    GLOBAL_STATE.active_tab = active_tab

    video_path: Optional[str] = None
    def _materialize_zip(file_dict: dict) -> Optional[str]:
        # gr.Fileからのzipを展開し、画像ディレクトリパスを返す
        orig = file_dict.get("orig_name") or ""
        if not orig.lower().endswith(".zip"):
            return None
        data_path = file_dict.get("name") or file_dict.get("path") or file_dict.get("data")
        if not data_path:
            return None
        out_dir = Path(tempfile.mkdtemp(prefix="sam3_frames_"))
        try:
            with zipfile.ZipFile(data_path, "r") as zf:
                for member in zf.namelist():
                    # パス traversal 防止
                    m_path = Path(member)
                    if m_path.is_dir():
                        continue
                    if m_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    dest = out_dir / m_path.name
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
            return str(out_dir)
        except Exception:
            return None

    if isinstance(video, dict):
        video_path = _materialize_zip(video) or video.get("name") or video.get("path") or video.get("data")
    elif isinstance(video, str):
        video_path = video
    else:
        video_path = None

    if not video_path:
        raise gr.Error("Invalid video input.")

    frames, info = try_load_video_frames(video_path)
    if len(frames) == 0:
        raise gr.Error("No frames could be loaded from the video.")

    MAX_SECONDS = 8.0
    trimmed_note = ""
    fps_in = info.get("fps")
    max_frames_allowed = int(MAX_SECONDS * fps_in) if fps_in else len(frames)
    if len(frames) > max_frames_allowed:
        frames = frames[:max_frames_allowed]
        trimmed_note = f" (trimmed to {int(MAX_SECONDS)}s = {len(frames)} frames)"
        if isinstance(info, dict):
            info["num_frames"] = len(frames)
    GLOBAL_STATE.video_frames = frames
    GLOBAL_STATE.video_fps = float(fps_in) if fps_in else None

    width, height = frames[0].size
    num_frames = len(frames)

    if active_tab == "text":
        processor = _GLOBAL_TEXT_VIDEO_PROCESSOR
        session = processor.init_video_session(
            video_path=video_path,
            width=width,
            height=height,
            num_frames=num_frames,
        )
        # Sam3VideoModel は init_state を介して inference_state を生成
        _state = _GLOBAL_TEXT_VIDEO_MODEL.model.init_state(
            resource_path=video_path,
            async_loading_frames=False,
            video_loader_type="image_dir" if Path(video_path).is_dir() else "cv2",
        )
        # state内テンソルをモデルdtypeに揃える
        def _cast_leaf(x):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=_GLOBAL_TEXT_VIDEO_MODEL.dtype, device=device)
            return x

        def _cast_state(obj):
            if isinstance(obj, dict):
                return {k: _cast_state(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cast_state(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_cast_state(v) for v in obj)
            return _cast_leaf(obj)

        session._state = _cast_state(_state)
        GLOBAL_STATE.inference_session = session
    else:
        processor = _GLOBAL_TRACKER_PROCESSOR
        GLOBAL_STATE.inference_session = processor.init_video_session(
            video_path=video_path,
            width=width,
            height=height,
            num_frames=num_frames,
        )
        # Tracker側は初回クリック前にキャッシュが無いため、frame0で先に1ステップ前進させてcacheを埋める
        _ = _GLOBAL_TRACKER_MODEL(
            inference_session=GLOBAL_STATE.inference_session,
            frame_idx=0,
        )

    first_frame = frames[0]
    max_idx = len(frames) - 1
    status = (
        f"Loaded {len(frames)} frames @ {GLOBAL_STATE.video_fps or 'unknown'} fps{trimmed_note}. "
        f"Device: {device}, dtype: {dtype}. Ready."
    )
    return GLOBAL_STATE, 0, max_idx, first_frame, status


def compose_frame(state: AppState, frame_idx: int) -> Image.Image:
    if state is None or state.video_frames is None or len(state.video_frames) == 0:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))
    frame = state.video_frames[frame_idx]
    masks = state.masks_by_frame.get(frame_idx, {})
    out_img = frame
    if len(masks) != 0:
        out_img = overlay_masks_on_frame(out_img, masks, state.color_by_obj, alpha=0.65)

    clicks_map = state.clicks_by_frame_obj.get(frame_idx)
    if clicks_map:
        draw = ImageDraw.Draw(out_img)
        cross_half = 6
        for obj_id, pts in clicks_map.items():
            for x, y, lbl in pts:
                color = (0, 255, 0) if int(lbl) == 1 else (255, 0, 0)
                draw.line([(x - cross_half, y), (x + cross_half, y)], fill=color, width=2)
                draw.line([(x, y - cross_half), (x, y + cross_half)], fill=color, width=2)
    if (
        state.pending_box_start is not None
        and state.pending_box_start_frame_idx == frame_idx
        and state.pending_box_start_obj_id is not None
    ):
        draw = ImageDraw.Draw(out_img)
        x, y = state.pending_box_start
        cross_half = 6
        color = state.color_by_obj.get(state.pending_box_start_obj_id, (255, 255, 255))
        draw.line([(x - cross_half, y), (x + cross_half, y)], fill=color, width=2)
        draw.line([(x, y - cross_half), (x, y + cross_half)], fill=color, width=2)
    box_map = state.boxes_by_frame_obj.get(frame_idx)
    if box_map:
        draw = ImageDraw.Draw(out_img)
        for obj_id, boxes in box_map.items():
            color = state.color_by_obj.get(obj_id, (255, 255, 255))
            for x1, y1, x2, y2 in boxes:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

    text_prompts_by_obj = {}
    for frame_texts in state.text_prompts_by_frame_obj.values():
        for obj_id, text_prompt in frame_texts.items():
            if obj_id not in text_prompts_by_obj:
                text_prompts_by_obj[obj_id] = text_prompt

    if text_prompts_by_obj and len(masks) > 0:
        draw = ImageDraw.Draw(out_img)
        font = ImageFont.load_default()

        for obj_id, text_prompt in text_prompts_by_obj.items():
            obj_mask = masks.get(obj_id)
            if obj_mask is not None:
                mask_array = np.array(obj_mask)
                if mask_array.size > 0 and np.any(mask_array):
                    rows = np.any(mask_array, axis=1)
                    cols = np.any(mask_array, axis=0)
                    if np.any(rows) and np.any(cols):
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        label_x = int(x_min)
                        label_y = int(y_min) - 20
                        label_y = max(5, label_y)

                        obj_color = state.color_by_obj.get(obj_id, (255, 255, 255))

                        label_text = f"{text_prompt} (ID: {obj_id})"
                        bbox = draw.textbbox((label_x, label_y), label_text, font=font)
                        padding = 4
                        draw.rectangle(
                            [
                                (bbox[0] - padding, bbox[1] - padding),
                                (bbox[2] + padding, bbox[3] + padding),
                            ],
                            fill=obj_color,
                            outline=None,
                            width=0,
                        )
                        draw.text((label_x, label_y), label_text, fill=(255, 255, 255), font=font)

    state.composited_frames[frame_idx] = out_img
    return out_img


def update_frame_display(state: AppState, frame_idx: int) -> Image.Image:
    if state is None or state.video_frames is None or len(state.video_frames) == 0:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))
    cached = state.composited_frames.get(frame_idx)
    if cached is not None:
        return cached
    return compose_frame(state, frame_idx)


def _ensure_color_for_obj(state: AppState, obj_id: int):
    if obj_id not in state.color_by_obj:
        state.color_by_obj[obj_id] = pastel_color_for_object(obj_id)


def on_image_click(
    img: Image.Image | np.ndarray,
    state: AppState,
    frame_idx: int,
    obj_id: int,
    label: str,
    clear_old: bool,
    evt: gr.SelectData,
) -> Image.Image:
    if state is None or state.inference_session is None:
        return img

    model = _GLOBAL_TRACKER_MODEL
    processor = _GLOBAL_TRACKER_PROCESSOR

    x = y = None
    if evt is not None:
        try:
            if hasattr(evt, "index") and isinstance(evt.index, (list, tuple)) and len(evt.index) == 2:
                x, y = int(evt.index[0]), int(evt.index[1])
            elif hasattr(evt, "value") and isinstance(evt.value, dict) and "x" in evt.value and "y" in evt.value:
                x, y = int(evt.value["x"]), int(evt.value["y"])
        except Exception:
            x = y = None

    if x is None or y is None:
        raise gr.Error("Could not read click coordinates.")

    _ensure_color_for_obj(state, int(obj_id))
    ann_frame_idx = int(frame_idx)
    ann_obj_id = int(obj_id)

    if state.current_prompt_type == "Boxes":
        if state.pending_box_start is None:
            frame_clicks = state.clicks_by_frame_obj.setdefault(ann_frame_idx, {})
            frame_clicks[ann_obj_id] = []
            state.composited_frames.pop(ann_frame_idx, None)
            state.pending_box_start = (int(x), int(y))
            state.pending_box_start_frame_idx = ann_frame_idx
            state.pending_box_start_obj_id = ann_obj_id
            state.composited_frames.pop(ann_frame_idx, None)
            return update_frame_display(state, ann_frame_idx)
        else:
            x1, y1 = state.pending_box_start
            x2, y2 = int(x), int(y)
            state.pending_box_start = None
            state.pending_box_start_frame_idx = None
            state.pending_box_start_obj_id = None
            state.composited_frames.pop(ann_frame_idx, None)
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)

            box = [[[x_min, y_min, x_max, y_max]]]
            processor.add_inputs_to_inference_session(
                inference_session=state.inference_session,
                frame_idx=ann_frame_idx,
                obj_ids=ann_obj_id,
                input_boxes=box,
            )

            frame_boxes = state.boxes_by_frame_obj.setdefault(ann_frame_idx, {})
            obj_boxes = frame_boxes.setdefault(ann_obj_id, [])
            obj_boxes.clear()
            obj_boxes.append((x_min, y_min, x_max, y_max))
            state.composited_frames.pop(ann_frame_idx, None)
    else:
        label_int = 1 if str(label).lower().startswith("pos") else 0

        frame_clicks = state.clicks_by_frame_obj.setdefault(ann_frame_idx, {})
        obj_clicks = frame_clicks.setdefault(ann_obj_id, [])

        if bool(clear_old):
            obj_clicks.clear()
            frame_boxes = state.boxes_by_frame_obj.setdefault(ann_frame_idx, {})
            frame_boxes[ann_obj_id] = []

        obj_clicks.append((int(x), int(y), int(label_int)))

        points = [[[[click[0], click[1]] for click in obj_clicks]]]
        labels = [[[click[2] for click in obj_clicks]]]

        processor.add_inputs_to_inference_session(
            inference_session=state.inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=points,
            input_labels=labels,
        )
        state.composited_frames.pop(ann_frame_idx, None)

    with torch.no_grad():
        outputs = model(
            inference_session=state.inference_session,
            frame_idx=ann_frame_idx,
        )

    out_mask_logits = processor.post_process_masks(
        [outputs.pred_masks],
        [[state.inference_session.video_height, state.inference_session.video_width]],
        binarize=False,
    )[0]

    mask_2d = (out_mask_logits[0] > 0.0).astype(np.float32)
    masks_for_frame = state.masks_by_frame.setdefault(ann_frame_idx, {})
    masks_for_frame[ann_obj_id] = mask_2d

    state.composited_frames.pop(ann_frame_idx, None)

    return update_frame_display(state, ann_frame_idx)


def on_text_prompt(
    state: AppState,
    frame_idx: int,
    text_prompt: str,
) -> tuple[Image.Image, str]:
    if state is None or state.inference_session is None:
        return None, "Upload a video and enter text prompt."

    model = _GLOBAL_TEXT_VIDEO_MODEL
    processor = _GLOBAL_TEXT_VIDEO_PROCESSOR

    if not text_prompt or not text_prompt.strip():
        return update_frame_display(state, int(frame_idx)), "Please enter a text prompt."

    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))

    state.inference_session = processor.add_text_prompt(
        inference_session=state.inference_session,
        text=text_prompt.strip(),
    )

    masks_for_frame = state.masks_by_frame.setdefault(frame_idx, {})
    frame_texts = state.text_prompts_by_frame_obj.setdefault(int(frame_idx), {})

    num_objects = 0
    detected_obj_ids = []
    with torch.no_grad():
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=state.inference_session,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
        ):
            current_frame_idx = model_outputs.frame_idx
            if current_frame_idx == frame_idx:
                object_ids = model_outputs.object_ids
                masks = model_outputs.masks
                scores = model_outputs.scores

                num_objects = len(object_ids)
                if num_objects > 0:
                    if len(scores) > 0:
                        sorted_indices = np.argsort(scores)[::-1].tolist()
                    else:
                        sorted_indices = list(range(num_objects))

                    for mask_idx in sorted_indices:
                        current_obj_id = int(object_ids[mask_idx])
                        detected_obj_ids.append(current_obj_id)
                        _ensure_color_for_obj(state, current_obj_id)
                        mask_2d = masks[mask_idx]
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze()
                        mask_2d = (mask_2d > 0.0).astype(np.float32)
                        masks_for_frame[current_obj_id] = mask_2d
                        frame_texts[current_obj_id] = text_prompt.strip()

    state.composited_frames.pop(frame_idx, None)

    if detected_obj_ids:
        obj_ids_str = ", ".join(map(str, detected_obj_ids))
        status = f"Processed text prompt '{text_prompt.strip()}' on frame {frame_idx}. Found {num_objects} object(s) with IDs: {obj_ids_str}."
    else:
        status = f"Processed text prompt '{text_prompt.strip()}' on frame {frame_idx}. No objects detected."
    return update_frame_display(state, int(frame_idx)), status


def propagate_masks(GLOBAL_STATE: gr.State):
    if GLOBAL_STATE is None:
        return GLOBAL_STATE, "Load a video first.", gr.update()

    if GLOBAL_STATE.active_tab != "text" and GLOBAL_STATE.inference_session is None:
        return GLOBAL_STATE, "Load a video first.", gr.update()

    total = max(1, GLOBAL_STATE.num_frames)
    processed = 0

    yield GLOBAL_STATE, f"Propagating masks: {processed}/{total}", gr.update()

    last_frame_idx = 0

    with torch.no_grad():
        if GLOBAL_STATE.active_tab == "text":
            if GLOBAL_STATE.inference_session is None:
                yield GLOBAL_STATE, "Text video model not loaded.", gr.update()
                return

            model = _GLOBAL_TEXT_VIDEO_MODEL
            processor = _GLOBAL_TEXT_VIDEO_PROCESSOR

            if not GLOBAL_STATE.text_prompts_by_frame_obj:
                yield GLOBAL_STATE, "No text prompts found. Please add a text prompt first.", gr.update()
                return

            earliest_frame = min(GLOBAL_STATE.text_prompts_by_frame_obj.keys())
            frames_to_track = GLOBAL_STATE.num_frames - earliest_frame

            for model_outputs in model.propagate_in_video_iterator(
                inference_session=GLOBAL_STATE.inference_session,
                start_frame_idx=earliest_frame,
                max_frame_num_to_track=frames_to_track,
            ):
                frame_idx = model_outputs.frame_idx
                object_ids = model_outputs.object_ids
                masks = model_outputs.masks
                scores = model_outputs.scores

                masks_for_frame = GLOBAL_STATE.masks_by_frame.setdefault(frame_idx, {})
                frame_texts = GLOBAL_STATE.text_prompts_by_frame_obj.setdefault(frame_idx, {})

                num_objects = len(object_ids)
                if num_objects > 0:
                    if len(scores) > 0:
                        sorted_indices = np.argsort(scores)[::-1].tolist()
                    else:
                        sorted_indices = list(range(num_objects))

                    for mask_idx in sorted_indices:
                        current_obj_id = int(object_ids[mask_idx])
                        _ensure_color_for_obj(GLOBAL_STATE, current_obj_id)
                        mask_2d = masks[mask_idx]
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze()
                        mask_2d = (mask_2d > 0.0).astype(np.float32)
                        masks_for_frame[current_obj_id] = mask_2d

                        found_prompt = None
                        for existing_frame_idx, existing_frame_texts in GLOBAL_STATE.text_prompts_by_frame_obj.items():
                            if current_obj_id in existing_frame_texts:
                                found_prompt = existing_frame_texts[current_obj_id]
                                break

                        if found_prompt:
                            frame_texts[current_obj_id] = found_prompt

                GLOBAL_STATE.composited_frames.pop(frame_idx, None)
                last_frame_idx = frame_idx
                processed += 1
                if processed % 30 == 0 or processed == total:
                    yield GLOBAL_STATE, f"Propagating masks: {processed}/{total}", gr.update(value=frame_idx)
        else:
            if GLOBAL_STATE.inference_session is None:
                yield GLOBAL_STATE, "Tracker model not loaded.", gr.update()
                return

            model = _GLOBAL_TRACKER_MODEL
            processor = _GLOBAL_TRACKER_PROCESSOR

            for sam_output in model.propagate_in_video_iterator(
                inference_session=GLOBAL_STATE.inference_session
            ):
                frame_idx = sam_output.frame_idx
                masks = processor.post_process_masks(
                    [sam_output.pred_masks],
                    [[GLOBAL_STATE.inference_session.video_height, GLOBAL_STATE.inference_session.video_width]],
                )[0]

                masks_for_frame = GLOBAL_STATE.masks_by_frame.setdefault(frame_idx, {})
                for i, out_obj_id in enumerate(GLOBAL_STATE.inference_session.obj_ids):
                    _ensure_color_for_obj(GLOBAL_STATE, int(out_obj_id))
                    mask_2d = masks[i]
                    masks_for_frame[int(out_obj_id)] = mask_2d
                    GLOBAL_STATE.composited_frames.pop(frame_idx, None)

                last_frame_idx = frame_idx
                processed += 1
                if processed % 30 == 0 or processed == total:
                    yield GLOBAL_STATE, f"Propagating masks: {processed}/{total}", gr.update(value=frame_idx)

    text = f"Propagated masks across {processed} frames."
    yield GLOBAL_STATE, text, gr.update(value=last_frame_idx)


def reset_session(GLOBAL_STATE: gr.State) -> tuple[AppState, Image.Image, int, int, str]:
    if not GLOBAL_STATE.video_frames:
        return GLOBAL_STATE, None, 0, 0, "Session reset. Load a new video."

    # 既存セッションは保持しつつ、プロンプトやキャッシュをクリアする
    GLOBAL_STATE.masks_by_frame.clear()
    GLOBAL_STATE.clicks_by_frame_obj.clear()
    GLOBAL_STATE.boxes_by_frame_obj.clear()
    GLOBAL_STATE.text_prompts_by_frame_obj.clear()
    GLOBAL_STATE.composited_frames.clear()
    GLOBAL_STATE.pending_box_start = None
    GLOBAL_STATE.pending_box_start_frame_idx = None
    GLOBAL_STATE.pending_box_start_obj_id = None

    gc.collect()

    current_idx = int(getattr(GLOBAL_STATE, "current_frame_idx", 0))
    current_idx = max(0, min(current_idx, GLOBAL_STATE.num_frames - 1))
    preview_img = update_frame_display(GLOBAL_STATE, current_idx)
    slider_minmax = gr.update(minimum=0, maximum=max(GLOBAL_STATE.num_frames - 1, 0), interactive=True)
    slider_value = gr.update(value=current_idx)
    status = "Session reset. Prompts cleared; video preserved."
    return GLOBAL_STATE, preview_img, slider_minmax, slider_value, status


def _on_video_change_pointbox(GLOBAL_STATE: gr.State, video):
    GLOBAL_STATE, min_idx, max_idx, first_frame, status = init_video_session(GLOBAL_STATE, video, "point_box")
    return (
        GLOBAL_STATE,
        gr.update(minimum=min_idx, maximum=max_idx, value=min_idx, interactive=True),
        first_frame,
        status,
    )


def _on_video_change_text(GLOBAL_STATE: gr.State, video):
    GLOBAL_STATE, min_idx, max_idx, first_frame, status = init_video_session(GLOBAL_STATE, video, "text")
    return (
        GLOBAL_STATE,
        gr.update(minimum=min_idx, maximum=max_idx, value=min_idx, interactive=True),
        first_frame,
        status,
    )


theme = Soft(primary_hue="blue", secondary_hue="rose", neutral_hue="slate")

with gr.Blocks(title="SAM3", theme=theme) as demo:
    GLOBAL_STATE = gr.State(AppState())

    gr.Markdown(
        """
        ### SAM3 Video Tracking · powered by local SAM3
        Segment and track objects across a video with SAM3 (Segment Anything 3).
        """
    )

    with gr.Tabs() as main_tabs:
        with gr.Tab("Text Prompting"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Quick start**
                        - **Load a video**: Upload your own or pick an example below.
                        - Select a frame and enter a text description to segment objects.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Working with results**
                        - **Preview**: Use the slider to navigate frames and see the current masks.
                        - **Propagate** across video.
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    video_in_text = gr.Video(label="Upload video", sources=["upload", "webcam"], interactive=True)
                    load_status_text = gr.Markdown(visible=True)
                    reset_btn_text = gr.Button("Reset Session", variant="secondary")
                with gr.Column(scale=2):
                    preview_text = gr.Image(label="Preview", interactive=True)
                    with gr.Row():
                        frame_slider_text = gr.Slider(
                            label="Frame", minimum=0, maximum=0, step=1, value=0, interactive=True
                        )
                        with gr.Column(scale=0):
                            propagate_btn_text = gr.Button("Propagate across video", variant="primary")
                            propagate_status_text = gr.Markdown(visible=True)
                    with gr.Row():
                        text_prompt_input = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Enter a text description (e.g., 'person', 'red car', 'short hair')",
                            lines=2,
                        )
                        text_apply_btn = gr.Button("Apply Text Prompt", variant="primary")
                    text_status = gr.Markdown(visible=True)

            with gr.Row():
                render_btn_text = gr.Button("Render MP4 for smooth playback", variant="primary")
            playback_video_text = gr.Video(label="Rendered Playback", interactive=False)

            examples_list_text = [
                [None, "./deers.mp4"],
                [None, "./penguins.mp4"],
                [None, "./foot.mp4"],
            ]
            with gr.Row():
                gr.Examples(
                    examples=examples_list_text,
                    inputs=[GLOBAL_STATE, video_in_text],
                    fn=_on_video_change_text,
                    outputs=[GLOBAL_STATE, frame_slider_text, preview_text, load_status_text],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=5,
                )

        with gr.Tab("Point/Box Prompting"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Quick start**
                        - **Load a video** and click points/boxes to guide tracking.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Working with results**
                        - **Preview** / **Propagate** / **Export MP4**.
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    video_in_pointbox = gr.Video(label="Upload video", sources=["upload", "webcam"], interactive=True, max_length=7)
                    load_status_pointbox = gr.Markdown(visible=True)
                    reset_btn_pointbox = gr.Button("Reset Session", variant="secondary")
                with gr.Column(scale=2):
                    preview_pointbox = gr.Image(label="Preview", interactive=True)
                    with gr.Row():
                        frame_slider_pointbox = gr.Slider(
                            label="Frame", minimum=0, maximum=0, step=1, value=0, interactive=True
                        )
                        with gr.Column(scale=0):
                            propagate_btn_pointbox = gr.Button("Propagate across video", variant="primary")
                            propagate_status_pointbox = gr.Markdown(visible=True)

            with gr.Row():
                obj_id_inp = gr.Number(value=1, precision=0, label="Object ID", scale=0)
                label_radio = gr.Radio(choices=["positive", "negative"], value="positive", label="Point label")
                clear_old_chk = gr.Checkbox(value=False, label="Clear old inputs for this object")
                prompt_type = gr.Radio(choices=["Points", "Boxes"], value="Points", label="Prompt type")

            with gr.Row():
                render_btn_pointbox = gr.Button("Render MP4 for smooth playback", variant="primary")
            playback_video_pointbox = gr.Video(label="Rendered Playback", interactive=False)

            examples_list_pointbox = [
                [None, "./deers.mp4"],
                [None, "./penguins.mp4"],
                [None, "./foot.mp4"],
            ]
            with gr.Row():
                gr.Examples(
                    examples=examples_list_pointbox,
                    inputs=[GLOBAL_STATE, video_in_pointbox],
                    fn=_on_video_change_pointbox,
                    outputs=[GLOBAL_STATE, frame_slider_pointbox, preview_pointbox, load_status_pointbox],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=5,
                )

    video_in_pointbox.change(
        _on_video_change_pointbox,
        inputs=[GLOBAL_STATE, video_in_pointbox],
        outputs=[GLOBAL_STATE, frame_slider_pointbox, preview_pointbox, load_status_pointbox],
        show_progress=True,
    )

    def _sync_frame_idx_pointbox(state_in: AppState, idx: int):
        if state_in is not None:
            state_in.current_frame_idx = int(idx)
        return update_frame_display(state_in, int(idx))

    frame_slider_pointbox.change(
        _sync_frame_idx_pointbox,
        inputs=[GLOBAL_STATE, frame_slider_pointbox],
        outputs=preview_pointbox,
    )

    video_in_text.change(
        _on_video_change_text,
        inputs=[GLOBAL_STATE, video_in_text],
        outputs=[GLOBAL_STATE, frame_slider_text, preview_text, load_status_text],
        show_progress=True,
    )

    def _sync_frame_idx_text(state_in: AppState, idx: int):
        if state_in is not None:
            state_in.current_frame_idx = int(idx)
        return update_frame_display(state_in, int(idx))

    frame_slider_text.change(
        _sync_frame_idx_text,
        inputs=[GLOBAL_STATE, frame_slider_text],
        outputs=preview_text,
    )

    def _sync_obj_id(s: AppState, oid):
        if s is not None and oid is not None:
            s.current_obj_id = int(oid)
        return

    obj_id_inp.change(_sync_obj_id, inputs=[GLOBAL_STATE, obj_id_inp], outputs=[])

    def _sync_label(s: AppState, lab: str):
        if s is not None and lab is not None:
            s.current_label = str(lab)
        return gr.update()

    label_radio.change(_sync_label, inputs=[GLOBAL_STATE, label_radio], outputs=[])

    def _sync_prompt_type(s: AppState, val: str):
        if s is not None and val is not None:
            s.current_prompt_type = str(val)
            s.pending_box_start = None
        is_points = str(val).lower() == "points"
        updates = [
            gr.update(visible=is_points),
            gr.update(interactive=is_points) if is_points else gr.update(value=True, interactive=False),
        ]
        return updates

    prompt_type.change(
        _sync_prompt_type,
        inputs=[GLOBAL_STATE, prompt_type],
        outputs=[label_radio, clear_old_chk],
    )

    preview_pointbox.select(
        on_image_click,
        [preview_pointbox, GLOBAL_STATE, frame_slider_pointbox, obj_id_inp, label_radio, clear_old_chk],
        preview_pointbox,
    )

    def _on_text_apply(state: AppState, frame_idx: int, text: str):
        img, status = on_text_prompt(state, frame_idx, text)
        return img, status

    text_apply_btn.click(
        _on_text_apply,
        inputs=[GLOBAL_STATE, frame_slider_text, text_prompt_input],
        outputs=[preview_text, text_status],
    )

    def _render_video(s: AppState):
        if s is None or s.num_frames == 0:
            raise gr.Error("Load a video first.")
        fps = s.video_fps if s.video_fps and s.video_fps > 0 else 12
        frames_np = []
        first = compose_frame(s, 0)
        h, w = first.size[1], first.size[0]
        for idx in range(s.num_frames):
            img = s.composited_frames.get(idx)
            if img is None:
                img = compose_frame(s, idx)
            frames_np.append(np.array(img)[:, :, ::-1])
            if (idx + 1) % 60 == 0:
                gc.collect()
        out_path = "/tmp/sam3_playback.mp4"
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            for fr_bgr in frames_np:
                writer.write(fr_bgr)
            writer.release()
            return out_path
        except Exception as e:
            print(f"Failed to render video with cv2: {e}")
            raise gr.Error(f"Failed to render video: {e}")
        finally:
            del frames_np

    render_btn_pointbox.click(_render_video, inputs=[GLOBAL_STATE], outputs=[playback_video_pointbox])
    render_btn_text.click(_render_video, inputs=[GLOBAL_STATE], outputs=[playback_video_text])

    propagate_btn_pointbox.click(
        propagate_masks,
        inputs=[GLOBAL_STATE],
        outputs=[GLOBAL_STATE, propagate_status_pointbox, frame_slider_pointbox],
    )

    propagate_btn_text.click(
        propagate_masks,
        inputs=[GLOBAL_STATE],
        outputs=[GLOBAL_STATE, propagate_status_text, frame_slider_text],
    )

    reset_btn_pointbox.click(
        reset_session,
        inputs=GLOBAL_STATE,
        outputs=[GLOBAL_STATE, preview_pointbox, frame_slider_pointbox, frame_slider_pointbox, load_status_pointbox],
    )

    reset_btn_text.click(
        reset_session,
        inputs=GLOBAL_STATE,
        outputs=[GLOBAL_STATE, preview_text, frame_slider_text, frame_slider_text, load_status_text],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.queue(api_open=False).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
