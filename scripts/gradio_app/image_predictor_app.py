"""Image predictor application for SAM3 Gradio interface.

This module provides the ImagePredictorApp class, which manages the state
and operations for image segmentation with SAM3.
"""

import logging
from typing import Optional

import numpy as np
from PIL import Image

from .inference_results import ImageInferenceResult
from .model_manager import Sam3ModelManager
from .prompt_state import ImagePromptState
from .prompt_types import PromptMode
from .temp_file_manager import TempFileManager

logger = logging.getLogger(__name__)


class ImagePredictorApp:
    """Application class for SAM3 image segmentation.

    This class manages the state and operations for interactive image
    segmentation, including prompt handling, model inference, and result
    visualization.
    """

    def __init__(
        self,
        model_manager: Optional[Sam3ModelManager] = None,
        temp_manager: Optional[TempFileManager] = None,
    ):
        """Initialize the image predictor app.

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
        self.prompt_state = ImagePromptState()
        self.current_result: Optional[ImageInferenceResult] = None

        logger.info("ImagePredictorApp initialized")

    def set_image(self, image: Image.Image) -> None:
        """Set the input image for segmentation.

        Args:
            image: PIL Image object (RGB).

        Raises:
            ValueError: If image is invalid.
        """
        self.prompt_state.set_image(image)
        logger.info(
            f"Image set: {self.prompt_state.orig_size}, "
            f"scaled: {self.prompt_state.scaled_size}, "
            f"scale_factor: {self.prompt_state.scale_factor}"
        )

    def set_prompt_mode(self, mode: PromptMode) -> None:
        """Set the prompt mode.

        Args:
            mode: The prompt mode to use.
        """
        self.prompt_state.set_prompt_mode(mode)
        logger.info(f"Prompt mode set to: {mode.value}")

    def add_point(self, x: float, y: float, label: int) -> None:
        """Add a point prompt.

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
        """Add a box prompt.

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

    def clear_prompts(self) -> None:
        """Clear all prompts while keeping the image.

        This clears text, points, and boxes, but keeps the loaded image.
        """
        self.prompt_state.text_prompt = ""
        self.prompt_state.points = []
        self.prompt_state.boxes = []
        logger.info("Prompts cleared")

    def reset_session(self) -> None:
        """Reset the entire session.

        This clears all prompts, the image, and inference results.
        """
        self.prompt_state.reset()
        self.current_result = None
        logger.info("Session reset")

    def _create_processor(self, model):
        """Create a Sam3Processor instance (helper for testing)."""
        from sam3.model.sam3_image_processor import Sam3Processor

        device = self.model_manager.get_device()
        return Sam3Processor(model, device=device, confidence_threshold=0.5)

    def _has_any_prompts(self) -> bool:
        """Check if any prompts are set."""
        effective_text = self.prompt_state.get_effective_text()
        effective_points = self.prompt_state.get_effective_points()
        effective_boxes = self.prompt_state.get_effective_boxes()

        return bool(
            effective_text or len(effective_points) > 0 or len(effective_boxes) > 0
        )

    def run_segmentation(
        self, checkpoint_path: str = "models/sam3.pt"
    ) -> ImageInferenceResult:
        """Run segmentation with current image and prompts.

        Args:
            checkpoint_path: Path to the SAM3 model checkpoint.

        Returns:
            ImageInferenceResult containing masks, boxes, scores, and labels.

        Raises:
            ValueError: If no image or no prompts are set.
            RuntimeError: If segmentation fails.
        """
        # Validate state
        if self.prompt_state.image is None:
            raise ValueError(
                "No image set. Please call set_image() before running segmentation."
            )

        if not self._has_any_prompts():
            raise ValueError(
                "No prompts set. Please set text, points, or box prompts before running."
            )

        logger.info("Starting segmentation...")

        try:
            # Load model
            model = self.model_manager.get_image_model(checkpoint_path)

            # Create processor
            processor = self._create_processor(model)

            # Set image
            state = processor.set_image(self.prompt_state.image)

            # Get effective prompts based on current mode
            effective_text = self.prompt_state.get_effective_text()
            effective_points = self.prompt_state.get_effective_points()
            effective_boxes = self.prompt_state.get_effective_boxes()

            logger.info(
                f"Running with prompts: text={bool(effective_text)}, "
                f"points={len(effective_points)}, boxes={len(effective_boxes)}"
            )

            # Run inference based on prompts
            if effective_text:
                # Text prompt
                state = processor.set_text_prompt(effective_text, state)

            # Add geometric prompts (points/boxes)
            # Note: SAM3 uses normalized box coordinates in cxcywh format
            if len(effective_boxes) > 0 or len(effective_points) > 0:
                if "language_features" not in state.get("backbone_out", {}):
                    # No text prompt, need to set dummy prompt

                    dummy_text = processor.model.backbone.forward_text(
                        ["visual"], device=processor.device
                    )
                    state["backbone_out"].update(dummy_text)

                # Add boxes
                width, height = self.prompt_state.orig_size
                for box in effective_boxes:
                    # Convert from XYXY to normalized CXCYWH
                    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
                    cx = (x1 + x2) / 2 / width
                    cy = (y1 + y2) / 2 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    norm_box = [cx, cy, w, h]

                    state = processor.add_geometric_prompt(
                        norm_box, label=bool(box.label), state=state
                    )

            # Extract results
            masks_tensor = state["masks"]  # (N, 1, H, W) bool
            boxes_tensor = state["boxes"]  # (N, 4) float, XYXY
            scores_tensor = state["scores"]  # (N,) float

            # Convert to numpy
            masks_np = (
                masks_tensor.squeeze(1).cpu().numpy().astype(np.uint8) * 255
            )  # (N, H, W)
            boxes_np = boxes_tensor.cpu().numpy()  # (N, 4)
            scores_np = scores_tensor.cpu().numpy()  # (N,)

            # Create labels
            labels = None
            if effective_text:
                labels = [effective_text] * len(masks_np)

            # Create result
            result = ImageInferenceResult(
                masks=masks_np,
                boxes=boxes_np,
                scores=scores_np,
                labels=labels,
                meta={
                    "image_size": self.prompt_state.orig_size,
                    "prompt_mode": self.prompt_state.get_prompt_mode().value,
                },
            )

            # Ensure overlay image is built before returning
            result.ensure_overlay(self.prompt_state.image)

            self.current_result = result

            logger.info(
                f"Segmentation completed: {result.get_count()} objects detected"
            )
            return result

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise RuntimeError(f"Segmentation failed: {e}") from e

    def export_masks_zip(self) -> str:
        """Export masks from the latest inference result to a ZIP file.

        Returns:
            Path to the created ZIP file.

        Raises:
            ValueError: If no segmentation has been run yet.
        """
        if self.current_result is None:
            raise ValueError(
                "No segmentation results available. "
                "Please run run_segmentation() first."
            )

        zip_path = self.current_result.to_download_zip(self.temp_manager)
        logger.info(f"Exported masks to ZIP: {zip_path}")
        return zip_path
