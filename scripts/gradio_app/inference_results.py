"""Inference result data structures for SAM3 Gradio application.

This module provides data classes for storing and managing inference results
from both image segmentation and video tracking.
"""

import logging
import os
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .temp_file_manager import TempFileManager

logger = logging.getLogger(__name__)


# Color palette for visualization (RGB format)
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 255, 128),  # Spring Green
    (255, 0, 128),  # Rose
]


class ImageInferenceResult:
    """Result of image segmentation inference.

    This class stores the masks, boxes, scores, and labels produced by
    SAM3 image segmentation, and provides methods for visualization and export.
    """

    def __init__(
        self,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: Optional[List[str]] = None,
        meta: Optional[dict] = None,
    ):
        """Initialize an image inference result.

        Args:
            masks: Binary masks array of shape (N, H, W), uint8, 0-255.
            boxes: Bounding boxes array of shape (N, 4), float32, XYXY format.
            scores: Confidence scores array of shape (N,), float32.
            labels: Optional list of N label strings.
            meta: Optional metadata dictionary.

        Raises:
            ValueError: If inputs have inconsistent shapes or are empty.
        """
        if masks.shape[0] == 0:
            raise ValueError("masks must have at least one object")

        num_objects = masks.shape[0]

        if boxes.shape[0] != num_objects or scores.shape[0] != num_objects:
            raise ValueError(
                f"masks, boxes, and scores must have the same number of objects. "
                f"Got masks: {masks.shape[0]}, boxes: {boxes.shape[0]}, "
                f"scores: {scores.shape[0]}"
            )

        if labels is not None and len(labels) != num_objects:
            raise ValueError(
                f"labels must have {num_objects} elements, got {len(labels)}"
            )

        self.masks = masks
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.meta = meta if meta is not None else {}
        self.overlay_image: Optional[Image.Image] = None

        logger.debug(
            f"ImageInferenceResult created with {num_objects} objects, "
            f"mask shape: {masks.shape}"
        )

    def get_count(self) -> int:
        """Get the number of detected objects.

        Returns:
            Number of objects.
        """
        return self.masks.shape[0]

    def build_overlay_image(self, original_image: Image.Image) -> Image.Image:
        """Build an overlay image with masks and bounding boxes.

        Args:
            original_image: The original input image (PIL Image).

        Returns:
            PIL Image with masks and boxes overlaid.
        """
        # Convert to RGBA for alpha blending
        if original_image.mode != "RGBA":
            overlay = original_image.convert("RGBA")
        else:
            overlay = original_image.copy()

        width, height = original_image.size

        # Create a drawing context
        draw = ImageDraw.Draw(overlay)

        num_objects = self.get_count()

        for i in range(num_objects):
            color = COLORS[i % len(COLORS)]

            # Draw mask with alpha blending
            mask = self.masks[i]  # (H, W)

            # Create a colored mask overlay
            mask_rgba = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            mask_pixels = mask_rgba.load()

            for y in range(height):
                for x in range(width):
                    if mask[y, x] > 0:
                        # Apply semi-transparent color
                        mask_pixels[x, y] = (*color, 100)  # 100/255 alpha

            # Composite the mask onto the overlay
            overlay = Image.alpha_composite(overlay, mask_rgba)

            # Draw bounding box
            x1, y1, x2, y2 = self.boxes[i]
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label and score
            score = self.scores[i]
            if self.labels is not None and i < len(self.labels):
                label_text = f"{self.labels[i]}: {score:.2f}"
            else:
                label_text = f"obj{i}: {score:.2f}"

            # Position text above the box
            text_position = (x1, max(0, y1 - 20))

            # Draw text background
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except Exception:
                font = None

            # Get text bounding box
            bbox = draw.textbbox(text_position, label_text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 180))

            # Draw text
            draw.text(text_position, label_text, fill=(*color, 255), font=font)

        # Convert back to RGB for display
        self.overlay_image = overlay.convert("RGB")

        logger.info(f"Built overlay image with {num_objects} objects")
        return self.overlay_image

    def ensure_overlay(self, original_image: Image.Image) -> Image.Image:
        """Ensure overlay image is built, building it if necessary.

        Args:
            original_image: The original input image (PIL Image).

        Returns:
            PIL Image with masks and boxes overlaid (guaranteed non-None).
        """
        if self.overlay_image is None:
            self.overlay_image = self.build_overlay_image(original_image)
        return self.overlay_image

    def to_download_zip(self, temp_manager: TempFileManager) -> str:
        """Create a ZIP file for download containing all masks.

        Args:
            temp_manager: TempFileManager instance for creating the ZIP.

        Returns:
            Path to the created ZIP file.
        """
        # Convert masks to list of uint8 arrays
        mask_list = [self.masks[i] for i in range(self.get_count())]

        zip_path = temp_manager.create_mask_zip(mask_list, prefix="segmentation_masks")

        logger.info(f"Created download ZIP: {zip_path}")
        return zip_path


class VideoInferenceResult:
    """Result of video tracking inference.

    This class stores the per-frame masks, object IDs, and scores produced by
    SAM3 video tracking, and provides methods for export and visualization.
    """

    def __init__(
        self,
        masks_per_frame: dict,
        obj_ids: List[int],
        scores_per_frame: dict,
        labels: Optional[List[str]] = None,
        meta: Optional[dict] = None,
    ):
        """Initialize a video inference result.

        Args:
            masks_per_frame: Dict mapping frame_idx -> masks array of shape (N, H, W), uint8, 0-255.
            obj_ids: List of N object IDs.
            scores_per_frame: Dict mapping frame_idx -> scores array of shape (N,), float32.
            labels: Optional list of N label strings.
            meta: Optional metadata dictionary.

        Raises:
            ValueError: If inputs have inconsistent shapes or are empty.
        """
        if len(masks_per_frame) == 0:
            raise ValueError("masks_per_frame must have at least one frame")

        # Validate frame indices consistency
        mask_frames = set(masks_per_frame.keys())
        score_frames = set(scores_per_frame.keys())
        if mask_frames != score_frames:
            raise ValueError(
                f"masks_per_frame and scores_per_frame must have the same frame indices. "
                f"Got masks: {sorted(mask_frames)}, scores: {sorted(score_frames)}"
            )

        num_objects = len(obj_ids)

        # Validate number of objects per frame
        for frame_idx, masks in masks_per_frame.items():
            if masks.shape[0] != num_objects:
                raise ValueError(
                    f"Frame {frame_idx}: Expected {num_objects} objects, got {masks.shape[0]} masks"
                )

        for frame_idx, scores in scores_per_frame.items():
            if scores.shape[0] != num_objects:
                raise ValueError(
                    f"Frame {frame_idx}: Expected {num_objects} objects, got {scores.shape[0]} scores"
                )

        if labels is not None and len(labels) != num_objects:
            raise ValueError(
                f"labels must have {num_objects} elements, got {len(labels)}"
            )

        self.masks_per_frame = masks_per_frame
        self.obj_ids = obj_ids
        self.scores_per_frame = scores_per_frame
        self.labels = labels
        self.meta = meta if meta is not None else {}
        self.overlay_video_path: Optional[str] = None

        logger.debug(
            f"VideoInferenceResult created with {num_objects} objects, "
            f"{len(masks_per_frame)} frames"
        )

    def get_num_frames(self) -> int:
        """Get the number of frames.

        Returns:
            Number of frames.
        """
        return len(self.masks_per_frame)

    def get_num_objects(self) -> int:
        """Get the number of tracked objects.

        Returns:
            Number of objects.
        """
        return len(self.obj_ids)

    def export_tracks(self, temp_manager: TempFileManager) -> str:
        """Export tracking information to a JSON file.

        Args:
            temp_manager: TempFileManager instance for creating the JSON file.

        Returns:
            Path to the created JSON file.
        """
        import json
        import tempfile

        # Build tracking data structure
        tracks = []
        for i, obj_id in enumerate(self.obj_ids):
            track = {
                "obj_id": obj_id,
                "label": self.labels[i] if self.labels else f"obj{obj_id}",
                "frames": [],
            }

            # Add per-frame information
            for frame_idx in sorted(self.masks_per_frame.keys()):
                track["frames"].append(
                    {
                        "frame_idx": frame_idx,
                        "score": float(self.scores_per_frame[frame_idx][i]),
                    }
                )

            tracks.append(track)

        data = {
            "meta": self.meta,
            "num_objects": self.get_num_objects(),
            "num_frames": self.get_num_frames(),
            "tracks": tracks,
        }

        # Create JSON file
        fd, json_path = tempfile.mkstemp(
            suffix=".json", prefix="tracking_results_", dir=temp_manager.temp_dir
        )

        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported tracks to JSON: {json_path}")
        return json_path

    def to_download_zip(self, temp_manager: TempFileManager) -> str:
        """Create a ZIP file for download containing all frame masks.

        Args:
            temp_manager: TempFileManager instance for creating the ZIP.

        Returns:
            Path to the created ZIP file.
        """
        from zipfile import ZipFile
        import tempfile

        # Collect all masks with names
        mask_items = []

        for frame_idx in sorted(self.masks_per_frame.keys()):
            masks = self.masks_per_frame[frame_idx]
            for obj_idx, obj_id in enumerate(self.obj_ids):
                mask = masks[obj_idx]
                name = f"frame_{frame_idx:04d}_obj{obj_id}.png"
                mask_items.append((mask, name))

        # Create temporary mask files
        mask_files = []
        for mask, name in mask_items:
            fd, mask_path = tempfile.mkstemp(
                suffix=".png",
                prefix=f"{name.replace('.png', '_')}",
                dir=temp_manager.temp_dir,
            )
            os.close(fd)  # Close the file descriptor, cv2.imwrite will reopen
            cv2.imwrite(mask_path, mask)
            mask_files.append((mask_path, name))

        # Create ZIP file
        fd, zip_path = tempfile.mkstemp(
            suffix=".zip", prefix="video_masks_", dir=temp_manager.temp_dir
        )
        os.close(fd)

        with ZipFile(zip_path, "w") as zf:
            for file_path, arc_name in mask_files:
                zf.write(file_path, arcname=arc_name)

        logger.info(f"Created download ZIP with {len(mask_files)} masks: {zip_path}")
        return zip_path
