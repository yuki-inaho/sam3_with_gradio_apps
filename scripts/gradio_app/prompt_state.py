# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Prompt state management for SAM3 Gradio application.

This module manages the state of prompts (text, points, boxes) and associated
data (images, videos) for the Gradio UI.
"""

from typing import List, Optional

from PIL import Image

from .prompt_types import BoxPrompt, PointPrompt, PromptConfig, PromptMode


class PromptState:
    """Base class for managing prompt state.

    This class handles the core prompt data: text, points, and boxes,
    along with the current prompt configuration mode.
    """

    def __init__(self):
        """Initialize an empty prompt state."""
        self.text_prompt: str = ""
        self.points: List[PointPrompt] = []
        self.boxes: List[BoxPrompt] = []
        self.config = PromptConfig(PromptMode.AUTO)

    def reset(self) -> None:
        """Reset all prompts to empty state."""
        self.text_prompt = ""
        self.points = []
        self.boxes = []
        # Keep the current mode, just clear prompts

    def set_text_prompt(self, text: str) -> None:
        """Set the text prompt.

        Args:
            text: The text prompt string.
        """
        self.text_prompt = text

    def add_point(self, x: float, y: float, label: int) -> None:
        """Add a point prompt.

        Args:
            x: X coordinate of the point.
            y: Y coordinate of the point.
            label: Prompt label (1 for include, 0 for exclude).

        Raises:
            ValueError: If label is not 0 or 1.
        """
        point = PointPrompt(x=x, y=y, label=label)
        self.points.append(point)

    def add_box(self, x1: float, y1: float, x2: float, y2: float, label: int) -> None:
        """Add a box prompt.

        Args:
            x1: X coordinate of top-left corner.
            y1: Y coordinate of top-left corner.
            x2: X coordinate of bottom-right corner.
            y2: Y coordinate of bottom-right corner.
            label: Prompt label (1 for include, 0 for exclude).

        Raises:
            ValueError: If coordinates are invalid or label is not 0 or 1.
        """
        box = BoxPrompt(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
        self.boxes.append(box)

    def set_prompt_mode(self, mode: PromptMode) -> None:
        """Set the prompt mode.

        Args:
            mode: The new prompt mode.
        """
        self.config.set_mode(mode)

    def get_prompt_mode(self) -> PromptMode:
        """Get the current prompt mode.

        Returns:
            The current PromptMode.
        """
        return self.config.get_mode()

    def get_effective_text(self) -> Optional[str]:
        """Get the text prompt if enabled in current mode.

        Returns:
            The text prompt if text is enabled, None otherwise.
        """
        if self.config.is_text_enabled() and self.text_prompt:
            return self.text_prompt
        return None

    def get_effective_points(self) -> List[PointPrompt]:
        """Get point prompts if enabled in current mode.

        Returns:
            List of PointPrompt objects if points are enabled, empty list otherwise.
        """
        if self.config.is_points_enabled():
            return self.points
        return []

    def get_effective_boxes(self) -> List[BoxPrompt]:
        """Get box prompts if enabled in current mode.

        Returns:
            List of BoxPrompt objects if boxes are enabled, empty list otherwise.
        """
        if self.config.is_boxes_enabled():
            return self.boxes
        return []


class ImagePromptState(PromptState):
    """Prompt state for image segmentation.

    Extends PromptState with image-specific data: the image itself,
    original size, scaled size, and scale factor for coordinate transformations.
    """

    MAX_DIMENSION = 2048
    """Maximum dimension (width or height) for input images."""

    def __init__(self):
        """Initialize an empty image prompt state."""
        super().__init__()
        self.image: Optional[Image.Image] = None
        self.orig_size: tuple[int, int] = (0, 0)  # (width, height)
        self.scaled_size: tuple[int, int] = (0, 0)  # (width, height)
        self.scale_factor: float = 1.0

    def set_image(self, image: Image.Image) -> None:
        """Set the image and compute scaling information.

        Args:
            image: PIL Image object (RGB).

        Raises:
            ValueError: If image is not a PIL Image or has invalid dimensions.
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"image must be a PIL Image, got {type(image)}")

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        self.orig_size = (width, height)

        # Compute scale factor if resizing is needed
        max_dim = max(width, height)
        if max_dim > self.MAX_DIMENSION:
            self.scale_factor = self.MAX_DIMENSION / max_dim
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.scaled_size = (new_width, new_height)
            # Resize image with LANCZOS for high quality
            self.image = image.resize((new_width, new_height), resample=Image.LANCZOS)
        else:
            self.scale_factor = 1.0
            self.scaled_size = (width, height)
            self.image = image

    def reset(self) -> None:
        """Reset all prompts and clear the image."""
        super().reset()
        self.image = None
        self.orig_size = (0, 0)
        self.scaled_size = (0, 0)
        self.scale_factor = 1.0


class VideoPromptState(PromptState):
    """Prompt state for video tracking.

    Extends PromptState with video-specific data: video path, frame metadata,
    and the current prompt frame index.
    """

    def __init__(self):
        """Initialize an empty video prompt state."""
        super().__init__()
        self.video_path: str = ""
        self.frames: Optional[List] = None
        self.fps: float = 0.0
        self.num_frames: int = 0
        self.prompt_frame_idx: int = 0
        self.frame_range: tuple[int, int] = (0, 0)  # (start, end)
        self.stride: int = 1
        self.direction: str = "forward"  # "forward", "backward", "both"
        self.current_frame: Optional[Image.Image] = None

    def set_video(self, video_path: str) -> None:
        """Set the video path.

        This method stores the path but does not load the video.
        Video loading is deferred to when it's needed.

        Args:
            video_path: Path to the video file or frame directory.

        Raises:
            ValueError: If video_path is empty.
        """
        if not video_path:
            raise ValueError("video_path cannot be empty")

        self.video_path = video_path
        # Video metadata will be loaded when needed

    def set_frames(self, frames: List, fps: float) -> None:
        """Set video frames and FPS metadata.

        Args:
            frames: List of frame data (paths or arrays).
            fps: Frames per second of the video.

        Raises:
            ValueError: If frames is empty or fps is invalid.
        """
        if not frames:
            raise ValueError("frames cannot be empty")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        self.frames = frames
        self.fps = fps
        self.num_frames = len(frames)
        self.frame_range = (0, self.num_frames - 1)

    def set_prompt_frame(self, frame_idx: int) -> None:
        """Set the frame index where prompts will be applied.

        Args:
            frame_idx: The frame index (0-based).

        Raises:
            ValueError: If frame_idx is out of range.
        """
        if self.num_frames == 0:
            raise ValueError("No frames loaded, call set_frames first")
        if not (0 <= frame_idx < self.num_frames):
            raise ValueError(
                f"frame_idx {frame_idx} out of range [0, {self.num_frames - 1}]"
            )

        self.prompt_frame_idx = frame_idx

    def set_range(
        self, start: int, end: int, stride: int = 1, direction: str = "forward"
    ) -> None:
        """Set the frame range and tracking parameters.

        Args:
            start: Start frame index.
            end: End frame index (inclusive).
            stride: Frame stride (skip frames).
            direction: Tracking direction ("forward", "backward", "both").

        Raises:
            ValueError: If parameters are invalid.
        """
        if not (0 <= start < self.num_frames):
            raise ValueError(f"start {start} out of range [0, {self.num_frames - 1}]")
        if not (0 <= end < self.num_frames):
            raise ValueError(f"end {end} out of range [0, {self.num_frames - 1}]")
        if start > end:
            raise ValueError(f"start {start} must be <= end {end}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if direction not in ("forward", "backward", "both"):
            raise ValueError(f"Invalid direction: {direction}")

        self.frame_range = (start, end)
        self.stride = stride
        self.direction = direction

    def reset(self) -> None:
        """Reset all prompts but keep video metadata."""
        super().reset()
        # Keep video metadata, only reset prompts and prompt frame
        self.prompt_frame_idx = 0

    def set_current_frame(self, frame: Image.Image) -> None:
        """Set the current prompt frame image.
        
        Args:
            frame: PIL Image of the current prompt frame.
        """
        self.current_frame = frame

    def get_frame_with_overlay(self) -> Optional[Image.Image]:
        """Get the current frame with prompts (points/boxes) overlaid.

        Returns:
            PIL Image with prompts visualized, or None if no frame is set.
        """
        if self.current_frame is None:
            return None

        from PIL import ImageDraw

        # Copy the frame to avoid modifying the original
        overlay = self.current_frame.copy()
        draw = ImageDraw.Draw(overlay)

        # Draw points
        for point in self.points:
            color = (0, 255, 0) if point.label == 1 else (255, 0, 0)  # Green for include, Red for exclude
            radius = 5
            draw.ellipse(
                [(point.x - radius, point.y - radius), (point.x + radius, point.y + radius)],
                fill=color,
                outline="white",
                width=2,
            )

        # Draw boxes
        for box in self.boxes:
            color = (0, 255, 0) if box.label == 1 else (255, 0, 0)  # Green for include, Red for exclude
            draw.rectangle(
                [(box.x1, box.y1), (box.x2, box.y2)],
                outline=color,
                width=3,
            )

        return overlay
