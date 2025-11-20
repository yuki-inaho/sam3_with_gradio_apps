# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Model management for SAM3 Gradio application.

This module provides a singleton ModelManager for lazy-loading and sharing
SAM3 models across the application.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class Sam3ModelManager:
    """Singleton manager for SAM3 models.

    This class provides lazy-loading of image and video models,
    with automatic device selection (CUDA/CPU) and explicit error handling.
    """

    _instance: Optional["Sam3ModelManager"] = None
    _image_model: Optional[torch.nn.Module] = None
    _video_predictor: Optional[object] = None
    _device: Optional[str] = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model manager.

        Note: Due to singleton pattern, this only runs once.
        """
        if self._device is None:
            self._device = self._select_device()
            logger.info(f"Sam3ModelManager initialized with device: {self._device}")

    @staticmethod
    def _select_device() -> str:
        """Select the appropriate device (cuda or cpu).

        Returns:
            "cuda" if CUDA is available, "cpu" otherwise.
        """
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            return "cuda"
        else:
            logger.warning(
                "CUDA is not available, falling back to CPU. "
                "Performance may be significantly slower."
            )
            return "cpu"

    def get_device(self) -> str:
        """Get the current device.

        Returns:
            The device string ("cuda" or "cpu").
        """
        return self._device

    def load_models(
        self,
        checkpoint_path: str = "models/sam3.pt",
        bpe_path: Optional[str] = None,
    ) -> None:
        """Pre-load both image and video models.

        This method is optional; models can be loaded on-demand via
        get_image_model() and get_video_predictor().

        Args:
            checkpoint_path: Path to the SAM3 checkpoint file.
            bpe_path: Path to the BPE vocabulary file (optional).

        Raises:
            FileNotFoundError: If checkpoint file is not found.
            RuntimeError: If model loading fails.
        """
        self.get_image_model(checkpoint_path, bpe_path)
        self.get_video_predictor(checkpoint_path, bpe_path)
        logger.info("All models loaded successfully")

    def get_image_model(
        self,
        checkpoint_path: str = "models/sam3.pt",
        bpe_path: Optional[str] = None,
    ) -> torch.nn.Module:
        """Get the image model, loading it if necessary.

        Args:
            checkpoint_path: Path to the SAM3 checkpoint file.
            bpe_path: Path to the BPE vocabulary file (optional).

        Returns:
            The loaded SAM3 image model.

        Raises:
            FileNotFoundError: If checkpoint file is not found.
            RuntimeError: If model loading fails.
        """
        if self._image_model is not None:
            return self._image_model

        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                f"Please ensure the model file is downloaded and placed correctly."
            )

        logger.info(f"Loading image model from {checkpoint_path}...")

        try:
            from sam3.model_builder import build_sam3_image_model

            self._image_model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                load_from_HF=False,
                device=self._device,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )

            logger.info("Image model loaded successfully")
            return self._image_model

        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise RuntimeError(
                f"Failed to load SAM3 image model: {e}. "
                f"Please check the checkpoint file and dependencies."
            ) from e

    def get_video_predictor(
        self,
        checkpoint_path: str = "models/sam3.pt",
        bpe_path: Optional[str] = None,
    ) -> object:
        """Get the video predictor, loading it if necessary.

        Args:
            checkpoint_path: Path to the SAM3 checkpoint file.
            bpe_path: Path to the BPE vocabulary file (optional).

        Returns:
            The loaded SAM3 video predictor.

        Raises:
            FileNotFoundError: If checkpoint file is not found.
            RuntimeError: If model loading fails.
        """
        if self._video_predictor is not None:
            return self._video_predictor

        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                f"Please ensure the model file is downloaded and placed correctly."
            )

        logger.info(f"Loading video predictor from {checkpoint_path}...")

        try:
            from sam3.model_builder import build_sam3_video_predictor

            # Determine GPU IDs for multi-GPU support
            gpus_to_use = None
            if self._device == "cuda":
                gpus_to_use = [torch.cuda.current_device()]

            # Note: build_sam3_video_predictor does not accept load_from_HF; only
            # pass supported arguments to avoid unexpected keyword errors.
            self._video_predictor = build_sam3_video_predictor(
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                gpus_to_use=gpus_to_use,
            )

            logger.info("Video predictor loaded successfully")
            return self._video_predictor

        except Exception as e:
            logger.error(f"Failed to load video predictor: {e}")
            raise RuntimeError(
                f"Failed to load SAM3 video predictor: {e}. "
                f"Please check the checkpoint file and dependencies."
            ) from e

    def reset(self) -> None:
        """Reset the model manager, clearing all loaded models.

        This is useful for testing or when you want to reload models
        with different parameters.
        """
        self._image_model = None
        self._video_predictor = None
        logger.info("Model manager reset, all models cleared")
