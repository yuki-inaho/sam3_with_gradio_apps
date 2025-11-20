# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Prompt type definitions for SAM3 Gradio application.

This module defines the core types for managing different prompt modes
and their configurations in the Gradio UI.
"""

from dataclasses import dataclass
from enum import Enum


class PromptMode(Enum):
    """Enumeration of available prompt modes.

    These modes determine which types of prompts are active and how they
    should be processed by the model.
    """

    TEXT_ONLY = "text_only"
    """Only text prompts are active (points and boxes disabled)."""

    POINTS_ONLY = "points_only"
    """Only point prompts are active (text and boxes disabled)."""

    BOXES_ONLY = "boxes_only"
    """Only box prompts are active (text and points disabled)."""

    TEXT_PLUS_POINTS = "text_plus_points"
    """Text and point prompts are both active (boxes disabled)."""

    TEXT_PLUS_BOXES = "text_plus_boxes"
    """Text and box prompts are both active (points disabled)."""

    AUTO = "auto"
    """Automatically determine which prompts to use based on user input."""


class PromptConfig:
    """Configuration class for managing prompt mode settings.

    This class encapsulates the logic for determining which prompt channels
    (text, points, boxes) are enabled based on the selected mode.
    """

    def __init__(self, mode: PromptMode = PromptMode.AUTO):
        """Initialize the prompt configuration.

        Args:
            mode: The initial prompt mode. Defaults to AUTO.
        """
        self._mode = mode

    def set_mode(self, mode: PromptMode) -> None:
        """Set the current prompt mode.

        Args:
            mode: The new prompt mode to set.

        Raises:
            ValueError: If mode is not a valid PromptMode.
        """
        if not isinstance(mode, PromptMode):
            raise ValueError(f"mode must be a PromptMode, got {type(mode)}")
        self._mode = mode

    def get_mode(self) -> PromptMode:
        """Get the current prompt mode.

        Returns:
            The current PromptMode.
        """
        return self._mode

    def is_text_enabled(self) -> bool:
        """Check if text prompts are enabled in the current mode.

        Returns:
            True if text prompts should be processed, False otherwise.
        """
        return self._mode in (
            PromptMode.TEXT_ONLY,
            PromptMode.TEXT_PLUS_POINTS,
            PromptMode.TEXT_PLUS_BOXES,
            PromptMode.AUTO,
        )

    def is_points_enabled(self) -> bool:
        """Check if point prompts are enabled in the current mode.

        Returns:
            True if point prompts should be processed, False otherwise.
        """
        return self._mode in (
            PromptMode.POINTS_ONLY,
            PromptMode.TEXT_PLUS_POINTS,
            PromptMode.AUTO,
        )

    def is_boxes_enabled(self) -> bool:
        """Check if box prompts are enabled in the current mode.

        Returns:
            True if box prompts should be processed, False otherwise.
        """
        return self._mode in (
            PromptMode.BOXES_ONLY,
            PromptMode.TEXT_PLUS_BOXES,
            PromptMode.AUTO,
        )


@dataclass
class PointPrompt:
    """Data class representing a single point prompt.

    Attributes:
        x: X coordinate of the point (absolute or relative depending on context).
        y: Y coordinate of the point (absolute or relative depending on context).
        label: Prompt label (1 for include/positive, 0 for exclude/negative).
    """

    x: float
    y: float
    label: int

    def __post_init__(self):
        """Validate the point prompt after initialization."""
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label}")


@dataclass
class BoxPrompt:
    """Data class representing a single box prompt.

    Attributes:
        x1: X coordinate of the top-left corner.
        y1: Y coordinate of the top-left corner.
        x2: X coordinate of the bottom-right corner.
        y2: Y coordinate of the bottom-right corner.
        label: Prompt label (1 for include/positive, 0 for exclude/negative).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    label: int

    def __post_init__(self):
        """Validate the box prompt after initialization."""
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label}")
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be greater than y1 ({self.y1})")
