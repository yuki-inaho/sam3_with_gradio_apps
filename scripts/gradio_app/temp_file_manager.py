# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Temporary file management for SAM3 Gradio application.

This module provides utilities for creating and managing temporary files,
including mask ZIPs, overlay videos, and automatic cleanup of old files.
"""

import logging
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TempFileManager:
    """Manager for temporary file operations.

    This class handles creation of temporary files (mask ZIPs, videos)
    and automatic cleanup of old files to prevent disk space issues.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize the temporary file manager.

        Args:
            temp_dir: Optional custom temp directory. If None, uses
                     'temp/gradio_outputs' in the project root.
        """
        if temp_dir is None:
            # Default to temp/gradio_outputs
            project_root = Path(__file__).parent.parent.parent
            temp_dir = str(project_root / "temp" / "gradio_outputs")

        self.temp_dir = temp_dir
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"TempFileManager initialized with temp_dir: {self.temp_dir}")

    def create_mask_zip(self, masks: List[np.ndarray], prefix: str = "masks") -> str:
        """Create a ZIP file containing mask images.

        Args:
            masks: List of mask arrays (HxW, uint8, 0-255).
            prefix: Prefix for the ZIP filename.

        Returns:
            Path to the created ZIP file.

        Raises:
            ValueError: If masks is empty or contains invalid data.
        """
        if not masks:
            raise ValueError("masks cannot be empty")

        # Create temporary ZIP file
        timestamp = int(time.time())
        zip_path = Path(self.temp_dir) / f"{prefix}_{timestamp}.zip"

        logger.info(f"Creating mask ZIP: {zip_path} ({len(masks)} masks)")

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, mask in enumerate(masks):
                    # Validate mask
                    if not isinstance(mask, np.ndarray):
                        raise ValueError(
                            f"mask {idx} must be a numpy array, got {type(mask)}"
                        )

                    # Convert mask to PIL Image and save to ZIP
                    mask_img = Image.fromarray(mask)
                    mask_filename = f"mask_{idx}.png"

                    # Write to ZIP using BytesIO
                    from io import BytesIO

                    buffer = BytesIO()
                    mask_img.save(buffer, format="PNG")
                    zf.writestr(mask_filename, buffer.getvalue())

            logger.info(f"Successfully created mask ZIP: {zip_path}")
            return str(zip_path)

        except Exception as e:
            logger.error(f"Failed to create mask ZIP: {e}")
            # Clean up partial file if it exists
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(f"Failed to create mask ZIP: {e}") from e

    def create_temp_video_from_frames(
        self, frames: List[np.ndarray], fps: float, prefix: str = "video"
    ) -> str:
        """Create a temporary video file from frame arrays.

        Args:
            frames: List of frame arrays (HxWx3, uint8, RGB).
            fps: Frames per second for the output video.
            prefix: Prefix for the video filename.

        Returns:
            Path to the created video file.

        Raises:
            ValueError: If frames is empty, fps is invalid, or frame data is invalid.
            RuntimeError: If video creation fails.
        """
        if not frames:
            raise ValueError("frames cannot be empty")

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        # Create temporary video file
        timestamp = int(time.time())
        video_path = Path(self.temp_dir) / f"{prefix}_{timestamp}.mp4"

        logger.info(f"Creating video: {video_path} ({len(frames)} frames @ {fps} fps)")

        # Get video dimensions from first frame
        # Convert PIL Images to numpy arrays if needed
        converted_frames = []
        for idx, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                # Convert PIL Image to RGB numpy array
                frame = np.array(frame.convert("RGB"))
            elif not isinstance(frame, np.ndarray):
                raise ValueError(
                    f"frame {idx} must be a PIL Image or numpy array, got {type(frame)}"
                )
            converted_frames.append(frame)
        
        frames = converted_frames
        if not frames:
            raise ValueError("No valid frames after conversion")


        height, width = frames[0].shape[:2]

        try:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            if not writer.isOpened():
                raise RuntimeError(
                    "Failed to initialize cv2.VideoWriter. "
                    "Check that OpenCV is properly installed."
                )

            # Write frames
            for idx, frame in enumerate(frames):

                if frame.shape[:2] != (height, width):
                    raise ValueError(
                        f"frame {idx} has inconsistent dimensions: "
                        f"expected {(height, width)}, got {frame.shape[:2]}"
                    )

                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame

                writer.write(frame_bgr)

            writer.release()

            if not video_path.exists():
                raise RuntimeError(f"Video file was not created: {video_path}")

            logger.info(f"Successfully created video: {video_path}")
            return str(video_path)

        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            # Clean up partial file if it exists
            if video_path.exists():
                video_path.unlink()
            raise RuntimeError(f"Failed to create video: {e}") from e

    def cleanup_old_files(self, max_age_seconds: int = 3600) -> int:
        """Clean up old temporary files.

        Args:
            max_age_seconds: Maximum age of files to keep (in seconds).
                            Files older than this will be deleted.

        Returns:
            Number of files deleted.
        """
        current_time = time.time()
        deleted_count = 0

        logger.info(
            f"Starting cleanup of files older than {max_age_seconds}s "
            f"in {self.temp_dir}"
        )

        try:
            for file_path in Path(self.temp_dir).iterdir():
                if not file_path.is_file():
                    continue

                # Check file age
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(
                            f"Deleted old file: {file_path.name} "
                            f"(age: {file_age:.0f}s)"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path.name}: {e}")

            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return deleted_count
