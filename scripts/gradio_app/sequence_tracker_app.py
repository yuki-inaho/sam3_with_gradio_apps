"""Sequence tracker application for SAM3 Gradio interface.

This module provides the SequenceTrackerApp class, which manages the state
and operations for video tracking with SAM3.
"""

import logging
import os
from typing import Optional, Dict, Any, Generator, Literal

import cv2
import numpy as np
import torch
from PIL import Image

from .inference_results import VideoInferenceResult
from .model_manager import Sam3ModelManager
from .prompt_state import VideoPromptState
from .temp_file_manager import TempFileManager

logger = logging.getLogger(__name__)


class SequenceTrackerApp:
    """Application class for SAM3 video tracking.

    This class manages the state and operations for interactive video
    tracking, including prompt handling, model inference, and result
    visualization.
    """

    def __init__(
        self,
        model_manager: Optional[Sam3ModelManager] = None,
        temp_manager: Optional[TempFileManager] = None,
    ):
        """Initialize the sequence tracker app.

        Args:
            model_manager: Optional Sam3ModelManager instance. If None, creates new.
            temp_manager: Optional TempFileManager instance. If None, creates new.
        """
        self.model_manager = (
            model_manager if model_manager is not None else Sam3ModelManager()
        )
        self.temp_manager = (
            temp_manager if temp_manager is not None else TempFileManager()
        )
        self.prompt_state = VideoPromptState()
        self.current_result: Optional[VideoInferenceResult] = None
        self.video_path: Optional[str] = None
        self.video_metadata: Optional[Dict[str, Any]] = None

        logger.info("SequenceTrackerApp initialized")

    @staticmethod
    def _extract_masks_and_scores(
        outputs: Dict[str, Any], width: int, height: int
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract masks and scores from predictor outputs with key fallback.

        Masks are upsampled (nearest) to the original video resolution and
        converted to uint8 [0,255]. If masks are missing, raises RuntimeError.
        Scores are optional and returned as float32 if length matches masks.
        """

        masks = (
            outputs.get("masks")
            or outputs.get("pred_masks")
            or outputs.get("low_res_masks")
        )
        if masks is None:
            raise RuntimeError("No masks found in model outputs")

        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)

        if masks.ndim < 3:
            raise RuntimeError(f"Masks have invalid shape: {masks.shape}")

        # Ensure shape is (N, H, W)
        if masks.ndim == 4:
            # Remove batch dim if present: (B, N, H, W) -> (N, H, W)
            masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])

        resized_masks: list[np.ndarray] = []
        for m in masks:
            if m.shape[-2:] != (height, width):
                m_resized = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                m_resized = m
            # Binarize if logits/float
            if m_resized.dtype != np.uint8:
                m_resized = (m_resized > 0).astype(np.uint8)
            resized_masks.append(m_resized * 255)

        masks_uint8 = np.stack(resized_masks, axis=0)

        scores = outputs.get("scores")
        if scores is None:
            return masks_uint8, None

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32)
        return masks_uint8, scores

    def set_video(self, video_path: str) -> Dict[str, Any]:
        """Set the input video for tracking.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing video metadata (fps, num_frames, width, height).

        Raises:
            ValueError: If video file does not exist or cannot be opened.
        """
        # Check if file exists
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Open video and extract metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.video_metadata = {
                "fps": fps,
                "num_frames": num_frames,
                "width": width,
                "height": height,
            }
            self.video_path = video_path

            # Update prompt_state with video metadata
            self.prompt_state.set_video(video_path)
            # Use dummy frame list (indices) to set num_frames without loading all frames
            dummy_frames = list(range(num_frames))
            self.prompt_state.set_frames(dummy_frames, fps)

            logger.info(
                f"Video loaded: {video_path}, "
                f"{num_frames} frames, {width}x{height}, {fps:.2f} fps"
            )

            return self.video_metadata
        finally:
            cap.release()

    def set_prompt_frame(self, frame_idx: int) -> Image.Image:
        """Set the prompt frame and extract it from the video.

        Args:
            frame_idx: Index of the frame to extract (0-based).

        Returns:
            PIL Image of the extracted frame.

        Raises:
            ValueError: If no video is loaded or frame index is out of range.
        """
        if self.video_path is None or self.video_metadata is None:
            raise ValueError("No video loaded. Call set_video() first.")

        num_frames = self.video_metadata["num_frames"]
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(
                f"Frame index out of range. Video has {num_frames} frames, "
                f"got index {frame_idx}."
            )

        # Open video and seek to frame
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        try:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_idx}")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            frame_image = Image.fromarray(frame_rgb)

            # Update prompt_state
            self.prompt_state.set_prompt_frame(frame_idx)
            self.prompt_state.set_current_frame(frame_image)

            logger.info(f"Extracted frame {frame_idx} from video")

            return frame_image
        finally:
            cap.release()

    def add_point(self, x: float, y: float, label: int) -> None:
        """Add a point prompt to the prompt frame.

        Args:
            x: X coordinate of the point.
            y: Y coordinate of the point.
            label: Prompt label (1 for include, 0 for exclude).

        Raises:
            ValueError: If label is invalid or coordinates are out of bounds.
        """
        self.prompt_state.add_point(x, y, label)
        logger.debug(f"Point added: ({x}, {y}), label={label}")

    def add_box(self, x1: float, y1: float, x2: float, y2: float, label: int) -> None:
        """Add a box prompt to the prompt frame.

        Args:
            x1: X coordinate of top-left corner.
            y1: Y coordinate of top-left corner.
            x2: X coordinate of bottom-right corner.
            y2: Y coordinate of bottom-right corner.
            label: Prompt label (1 for include, 0 for exclude).

        Raises:
            ValueError: If label is invalid or coordinates are invalid.
        """
        self.prompt_state.add_box(x1, y1, x2, y2, label)
        logger.debug(f"Box added: ({x1}, {y1}, {x2}, {y2}), label={label}")

    def export_masks_zip(self) -> str:
        """Export tracking masks to a ZIP file.

        Returns:
            Path to the created ZIP file.

        Raises:
            ValueError: If no tracking result is available.
        """
        if self.current_result is None:
            raise ValueError("No tracking result available. Run tracking first.")

        zip_path = self.current_result.to_download_zip(self.temp_manager)
        logger.info(f"Masks exported to ZIP: {zip_path}")
        return zip_path

    def export_tracks_json(self) -> str:
        """Export tracking information to a JSON file.

        Returns:
            Path to the created JSON file.

        Raises:
            ValueError: If no tracking result is available.
        """
        if self.current_result is None:
            raise ValueError("No tracking result available. Run tracking first.")

        json_path = self.current_result.export_tracks(self.temp_manager)
        logger.info(f"Tracking info exported to JSON: {json_path}")
        return json_path

    def preview_tracking(
        self, max_frames: int = 10, num_frames: Optional[int] = None
    ) -> VideoInferenceResult:
        """Run tracking on a limited number of frames for preview.

        Args:
            max_frames: Maximum number of frames to process for preview.
            num_frames: Optional alias for max_frames (for backward compatibility).

        Returns:
            VideoInferenceResult with tracking results for preview frames.

        Raises:
            ValueError: If no video loaded, no prompt frame set, or no prompts added.
        """
        # Validate state
        if self.video_path is None or self.video_metadata is None:
            raise ValueError("No video loaded. Call set_video() first.")

        if self.prompt_state.prompt_frame_idx is None:
            raise ValueError("No prompt frame set. Call set_prompt_frame() first.")

        if not self.prompt_state.points and not self.prompt_state.boxes:
            raise ValueError("No prompts added. Call add_point() or add_box() first.")

        # Determine effective preview length
        effective_max = num_frames if num_frames is not None else max_frames
        if effective_max is None or effective_max <= 0:
            raise ValueError("max_frames must be > 0")

        # Get video predictor
        predictor = self.model_manager.get_video_predictor()

        # Start session
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=self.video_path,
            )
        )
        session_id = response["session_id"]

        try:
            # Convert prompts to SAM3 format
            prompt_frame_idx = int(self.prompt_state.prompt_frame_idx)
            total_frames = self.video_metadata["num_frames"]
            prompt_frame_idx = max(0, min(prompt_frame_idx, total_frames - 1))
            width = self.video_metadata["width"]
            height = self.video_metadata["height"]

            # Add prompts (points and boxes)
            if self.prompt_state.points or self.prompt_state.boxes:
                req = dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame_idx,
                    obj_id=1,  # Use obj_id=1 for the first object (as in official example)
                )

                if self.prompt_state.points:
                    points_abs = np.array(
                        [[p.x, p.y] for p in self.prompt_state.points], dtype=np.float32
                    )
                    labels = np.array(
                        [p.label for p in self.prompt_state.points], dtype=np.int32
                    )
                    points_rel = points_abs.copy()
                    points_rel[:, 0] /= width
                    points_rel[:, 1] /= height
                    req["points"] = torch.tensor(points_rel, dtype=torch.float32)
                    req["point_labels"] = torch.tensor(labels, dtype=torch.int32)

                if self.prompt_state.boxes:
                    boxes_abs = np.array(
                        [[b.x1, b.y1, b.x2, b.y2] for b in self.prompt_state.boxes],
                        dtype=np.float32,
                    )
                    labels = np.array(
                        [b.label for b in self.prompt_state.boxes], dtype=np.int32
                    )
                    boxes_rel = boxes_abs.copy()
                    boxes_rel[:, 0] /= width
                    boxes_rel[:, 1] /= height
                    boxes_rel[:, 2] /= width
                    boxes_rel[:, 3] /= height
                    req["bounding_boxes"] = torch.tensor(boxes_rel, dtype=torch.float32)
                    req["bounding_box_labels"] = torch.tensor(labels, dtype=torch.int32)

                predictor.handle_request(request=req)

            # Determine frame range for preview
            start_frame = prompt_frame_idx
            end_frame = min(prompt_frame_idx + effective_max, total_frames)

            # Propagate and collect outputs
            masks_per_frame = {}
            obj_ids_set = set()
            scores_per_frame = {}

            for response in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = response["frame_index"]

                # Only collect frames in preview range
                if start_frame <= frame_idx < end_frame:
                    outputs = response["outputs"]

                    obj_ids = outputs.get("obj_ids")
                    if obj_ids is None:
                        logger.warning("Skipping frame %s: obj_ids missing", frame_idx)
                        continue
                    if isinstance(obj_ids, torch.Tensor):
                        obj_ids = obj_ids.cpu().numpy().tolist()
                    elif isinstance(obj_ids, np.ndarray):
                        obj_ids = obj_ids.tolist()

                    obj_ids_set.update(obj_ids)

                    try:
                        masks, scores = self._extract_masks_and_scores(
                            outputs, width, height
                        )
                    except RuntimeError as e:
                        logger.warning("Skipping frame %s: %s", frame_idx, e)
                        continue

                    if masks.shape[0] != len(obj_ids):
                        logger.warning(
                            "Skipping frame %s: masks count %s != obj_ids count %s",
                            frame_idx,
                            masks.shape[0],
                            len(obj_ids),
                        )
                        continue

                    masks_per_frame[frame_idx] = masks

                    if scores is not None:
                        if scores.shape[0] != len(obj_ids):
                            logger.warning(
                                "Skipping scores for frame %s: scores count %s != obj_ids count %s",
                                frame_idx,
                                scores.shape[0],
                                len(obj_ids),
                            )
                        else:
                            scores_per_frame[frame_idx] = scores
                    else:
                        scores_per_frame[frame_idx] = np.zeros(
                            len(obj_ids), dtype=np.float32
                        )

                # Stop after reaching end of preview range
                if frame_idx >= end_frame - 1:
                    break

            if len(masks_per_frame) == 0:
                raise RuntimeError("No masks were produced during preview propagation.")

            # Create result
            result = VideoInferenceResult(
                masks_per_frame=masks_per_frame,
                obj_ids=sorted(list(obj_ids_set)),
                scores_per_frame=scores_per_frame,
                meta={
                    "video_path": self.video_path,
                    "prompt_frame_idx": prompt_frame_idx,
                    "preview_range": (start_frame, end_frame),
                },
            )

            self.current_result = result
            logger.info(
                f"Preview tracking completed: {len(masks_per_frame)} frames "
                f"({start_frame} to {end_frame - 1})"
            )

            return result

        finally:
            # Always close session to free resources
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

    def run_tracking(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        direction: Literal["forward", "backward", "both"] = "forward",
        max_frames: Optional[int] = None,
    ) -> VideoInferenceResult:
        """Run tracking on the entire video or specified frame range.

        Args:
            start_frame: Start frame index (default: 0).
            end_frame: End frame index (default: last frame).
            direction: Tracking direction - "forward" (default), "backward", or "both".

        Returns:
            VideoInferenceResult with tracking results for all frames.

        Raises:
            ValueError: If no video loaded, no prompt frame set, or no prompts added.
        """
        # Validate state
        if self.video_path is None or self.video_metadata is None:
            raise ValueError("No video loaded. Call set_video() first.")

        if self.prompt_state.prompt_frame_idx is None:
            raise ValueError("No prompt frame set. Call set_prompt_frame() first.")

        if not self.prompt_state.points and not self.prompt_state.boxes:
            raise ValueError("No prompts added. Call add_point() or add_box() first.")

        # Set default frame range and clamp
        total_frames = self.video_metadata["num_frames"]
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        if max_frames is not None and max_frames > 0:
            end_frame = min(end_frame, start_frame + max_frames)

        # Get video predictor
        predictor = self.model_manager.get_video_predictor()

        # Start session
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=self.video_path,
            )
        )
        session_id = response["session_id"]

        try:
            # Convert prompts to SAM3 format
            prompt_frame_idx = int(self.prompt_state.prompt_frame_idx)
            prompt_frame_idx = max(0, min(prompt_frame_idx, total_frames - 1))
            width = self.video_metadata["width"]
            height = self.video_metadata["height"]

            # Add prompts (points and boxes)
            if self.prompt_state.points or self.prompt_state.boxes:
                req = dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame_idx,
                    obj_id=1,  # Use obj_id=1 for the first object (as in official example)
                )

                if self.prompt_state.points:
                    points_abs = np.array(
                        [[p.x, p.y] for p in self.prompt_state.points], dtype=np.float32
                    )
                    labels = np.array(
                        [p.label for p in self.prompt_state.points], dtype=np.int32
                    )
                    points_rel = points_abs.copy()
                    points_rel[:, 0] /= width
                    points_rel[:, 1] /= height
                    req["points"] = torch.tensor(points_rel, dtype=torch.float32)
                    req["point_labels"] = torch.tensor(labels, dtype=torch.int32)

                if self.prompt_state.boxes:
                    boxes_abs = np.array(
                        [[b.x1, b.y1, b.x2, b.y2] for b in self.prompt_state.boxes],
                        dtype=np.float32,
                    )
                    labels = np.array(
                        [b.label for b in self.prompt_state.boxes], dtype=np.int32
                    )
                    boxes_rel = boxes_abs.copy()
                    boxes_rel[:, 0] /= width
                    boxes_rel[:, 1] /= height
                    boxes_rel[:, 2] /= width
                    boxes_rel[:, 3] /= height
                    req["bounding_boxes"] = torch.tensor(boxes_rel, dtype=torch.float32)
                    req["bounding_box_labels"] = torch.tensor(labels, dtype=torch.int32)

                predictor.handle_request(request=req)

            # Propagate and collect outputs
            masks_per_frame = {}
            obj_ids_set = set()
            scores_per_frame = {}

            for response in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = response["frame_index"]

                # Only collect frames in specified range
                if start_frame <= frame_idx < end_frame:
                    outputs = response["outputs"]

                    obj_ids = outputs.get("obj_ids")
                    if obj_ids is None:
                        logger.warning("Skipping frame %s: obj_ids missing", frame_idx)
                        continue
                    if isinstance(obj_ids, torch.Tensor):
                        obj_ids = obj_ids.cpu().numpy().tolist()
                    elif isinstance(obj_ids, np.ndarray):
                        obj_ids = obj_ids.tolist()
                    obj_ids_set.update(obj_ids)

                    try:
                        masks, scores = self._extract_masks_and_scores(
                            outputs, width, height
                        )
                    except RuntimeError as e:
                        logger.warning("Skipping frame %s: %s", frame_idx, e)
                        continue

                    if masks.shape[0] != len(obj_ids):
                        logger.warning(
                            "Skipping frame %s: masks count %s != obj_ids count %s",
                            frame_idx,
                            masks.shape[0],
                            len(obj_ids),
                        )
                        continue
                    masks_per_frame[frame_idx] = masks

                    if scores is not None:
                        if scores.shape[0] != len(obj_ids):
                            logger.warning(
                                "Skipping scores for frame %s: scores count %s != obj_ids count %s",
                                frame_idx,
                                scores.shape[0],
                                len(obj_ids),
                            )
                        else:
                            scores_per_frame[frame_idx] = scores
                    else:
                        scores_per_frame[frame_idx] = np.zeros(
                            len(obj_ids), dtype=np.float32
                        )

            if len(masks_per_frame) == 0:
                raise RuntimeError("No masks were produced during tracking.")

            # Create result
            result = VideoInferenceResult(
                masks_per_frame=masks_per_frame,
                obj_ids=sorted(list(obj_ids_set)),
                scores_per_frame=scores_per_frame,
                meta={
                    "video_path": self.video_path,
                    "prompt_frame_idx": prompt_frame_idx,
                    "frame_range": (start_frame, end_frame),
                    "direction": direction,
                },
            )

            self.current_result = result
            logger.info(
                f"Tracking completed ({direction}): {len(masks_per_frame)} frames "
                f"({start_frame} to {end_frame - 1})"
            )

            return result

        finally:
            # Always close session to free resources
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

    def run_tracking_stream(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Run tracking with streaming progress updates.

        Yields progress updates as frames are processed, then yields the final result.

        Args:
            start_frame: Start frame index (default: 0).
            end_frame: End frame index (default: last frame).

        Yields:
            Dict with either:
                - {"progress": frame_idx, "total": total_frames} for progress updates
                - {"result": VideoInferenceResult} for final result

        Raises:
            ValueError: If no video loaded, no prompt frame set, or no prompts added.
        """
        # Validate state
        if self.video_path is None or self.video_metadata is None:
            raise ValueError("No video loaded. Call set_video() first.")

        if self.prompt_state.prompt_frame_idx is None:
            raise ValueError("No prompt frame set. Call set_prompt_frame() first.")

        if not self.prompt_state.points and not self.prompt_state.boxes:
            raise ValueError("No prompts added. Call add_point() or add_box() first.")

        # Set default frame range and clamp
        total_frames_all = self.video_metadata["num_frames"]
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames_all
        start_frame = max(0, min(start_frame, total_frames_all - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames_all))
        if max_frames is not None and max_frames > 0:
            end_frame = min(end_frame, start_frame + max_frames)
        total_frames = end_frame - start_frame

        # Get video predictor
        predictor = self.model_manager.get_video_predictor()

        # Start session
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=self.video_path,
            )
        )
        session_id = response["session_id"]

        try:
            # Convert prompts to SAM3 format
            prompt_frame_idx = int(self.prompt_state.prompt_frame_idx)
            prompt_frame_idx = max(0, min(prompt_frame_idx, total_frames_all - 1))
            width = self.video_metadata["width"]
            height = self.video_metadata["height"]

            # Add prompts (points and boxes)
            if self.prompt_state.points or self.prompt_state.boxes:
                req = dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame_idx,
                    obj_id=1,  # Use obj_id=1 for the first object (as in official example)
                )

                if self.prompt_state.points:
                    points_abs = np.array(
                        [[p.x, p.y] for p in self.prompt_state.points], dtype=np.float32
                    )
                    labels = np.array(
                        [p.label for p in self.prompt_state.points], dtype=np.int32
                    )
                    points_rel = points_abs.copy()
                    points_rel[:, 0] /= width
                    points_rel[:, 1] /= height
                    req["points"] = torch.tensor(points_rel, dtype=torch.float32)
                    req["point_labels"] = torch.tensor(labels, dtype=torch.int32)

                if self.prompt_state.boxes:
                    boxes_abs = np.array(
                        [[b.x1, b.y1, b.x2, b.y2] for b in self.prompt_state.boxes],
                        dtype=np.float32,
                    )
                    labels = np.array(
                        [b.label for b in self.prompt_state.boxes], dtype=np.int32
                    )
                    boxes_rel = boxes_abs.copy()
                    boxes_rel[:, 0] /= width
                    boxes_rel[:, 1] /= height
                    boxes_rel[:, 2] /= width
                    boxes_rel[:, 3] /= height
                    req["bounding_boxes"] = torch.tensor(boxes_rel, dtype=torch.float32)
                    req["bounding_box_labels"] = torch.tensor(labels, dtype=torch.int32)

                predictor.handle_request(request=req)

            # Propagate and collect outputs
            masks_per_frame: Dict[int, np.ndarray] = {}
            obj_ids_set: set[int] = set()
            scores_per_frame: Dict[int, np.ndarray] = {}
            processed_count = 0

            try:
                for response in predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                    )
                ):
                    frame_idx = response["frame_index"]

                    # Only collect frames in specified range
                    if start_frame <= frame_idx < end_frame:
                        outputs = response["outputs"]

                        obj_ids = outputs.get("obj_ids")
                        if obj_ids is None:
                            logger.warning("Skipping frame %s: obj_ids missing", frame_idx)
                            continue
                        if isinstance(obj_ids, torch.Tensor):
                            obj_ids = obj_ids.cpu().numpy().tolist()
                        elif isinstance(obj_ids, np.ndarray):
                            obj_ids = obj_ids.tolist()
                        obj_ids_set.update(obj_ids)

                        try:
                            masks, scores = self._extract_masks_and_scores(
                                outputs, width, height
                            )
                        except RuntimeError as e:
                            logger.warning("Skipping frame %s: %s", frame_idx, e)
                            continue

                        if masks.shape[0] != len(obj_ids):
                            logger.warning(
                                "Skipping frame %s: masks count %s != obj_ids count %s",
                                frame_idx,
                                masks.shape[0],
                                len(obj_ids),
                            )
                            continue
                        masks_per_frame[frame_idx] = masks

                        if scores is not None:
                            if scores.shape[0] != len(obj_ids):
                                logger.warning(
                                    "Skipping scores for frame %s: scores count %s != obj_ids count %s",
                                    frame_idx,
                                    scores.shape[0],
                                    len(obj_ids),
                                )
                            else:
                                scores_per_frame[frame_idx] = scores
                        else:
                            scores_per_frame[frame_idx] = np.zeros(
                                len(obj_ids), dtype=np.float32
                            )

                        # Yield progress update
                        processed_count += 1
                        yield {
                            "type": "progress",
                            "progress": processed_count,
                            "total": total_frames,
                            "frame_idx": frame_idx,
                            "desc": f"Frame {frame_idx}/{end_frame - 1}",
                        }

                if len(masks_per_frame) == 0:
                    raise RuntimeError("No masks were produced during tracking.")

                # Create result
                result = VideoInferenceResult(
                    masks_per_frame=masks_per_frame,
                    obj_ids=sorted(list(obj_ids_set)),
                    scores_per_frame=scores_per_frame,
                    meta={
                        "video_path": self.video_path,
                        "prompt_frame_idx": prompt_frame_idx,
                        "frame_range": (start_frame, end_frame),
                    },
                )

                self.current_result = result
                logger.info(
                    f"Streaming tracking completed: {len(masks_per_frame)} frames "
                    f"({start_frame} to {end_frame - 1})"
                )

                # Yield final result
                yield {
                    "type": "result",
                    "result": result,
                    "video_path": result.meta.get("video_path", self.video_path),
                }

            except Exception as e:
                logger.error("Tracking stream failed: %s", e)
                yield {"type": "error", "message": str(e)}

            if len(masks_per_frame) == 0:
                raise RuntimeError("No masks were produced during tracking.")

            # Create result
            result = VideoInferenceResult(
                masks_per_frame=masks_per_frame,
                obj_ids=sorted(list(obj_ids_set)),
                scores_per_frame=scores_per_frame,
                meta={
                    "video_path": self.video_path,
                    "prompt_frame_idx": prompt_frame_idx,
                    "frame_range": (start_frame, end_frame),
                },
            )

            self.current_result = result
            logger.info(
                f"Streaming tracking completed: {len(masks_per_frame)} frames "
                f"({start_frame} to {end_frame - 1})"
            )

            # Yield final result
            yield {"result": result}

        finally:
            # Always close session to free resources
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
