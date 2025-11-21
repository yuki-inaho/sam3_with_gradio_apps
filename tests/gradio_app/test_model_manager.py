# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for model manager module."""

import pytest
import torch

from scripts.gradio_app.model_manager import Sam3ModelManager


def test_model_manager_singleton():
    """Test that Sam3ModelManager implements singleton pattern."""
    manager1 = Sam3ModelManager()
    manager2 = Sam3ModelManager()

    # Both should be the same instance
    assert manager1 is manager2


def test_model_manager_device_selection():
    """Test device selection logic."""
    manager = Sam3ModelManager()
    device = manager.get_device()

    # Device should be either cuda or cpu
    assert device in ("cuda", "cpu")

    # If CUDA is available, device should be cuda
    if torch.cuda.is_available():
        assert device == "cuda"
    else:
        assert device == "cpu"


def test_model_manager_get_image_model_nonexistent_checkpoint():
    """Test that FileNotFoundError is raised for nonexistent checkpoint."""
    manager = Sam3ModelManager()
    # Reset to ensure clean state (singleton pattern)
    manager.reset()

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        manager.get_image_model(checkpoint_path="/nonexistent/path.pt")


def test_model_manager_get_video_predictor_nonexistent_checkpoint():
    """Test that FileNotFoundError is raised for nonexistent checkpoint."""
    manager = Sam3ModelManager()
    # Reset to ensure clean state (singleton pattern)
    manager.reset()

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        manager.get_video_predictor(checkpoint_path="/nonexistent/path.pt")


@pytest.mark.slow
def test_model_manager_load_real_models():
    """Test loading real models (slow test, requires models/sam3.pt)."""
    from pathlib import Path

    checkpoint_path = Path("models/sam3.pt")
    if not checkpoint_path.exists():
        pytest.skip("models/sam3.pt not found, skipping real model test")

    manager = Sam3ModelManager()
    # Reset to ensure clean state
    manager.reset()

    # Load image model
    image_model = manager.get_image_model(str(checkpoint_path))
    assert image_model is not None

    # Calling again should return the same instance
    image_model2 = manager.get_image_model(str(checkpoint_path))
    assert image_model is image_model2

    # Clean up
    manager.reset()


def test_model_manager_reset():
    """Test that reset clears loaded models."""
    manager = Sam3ModelManager()
    manager.reset()

    # After reset, internal models should be None
    # We can't directly access private attributes in tests,
    # but we can verify behavior by checking that get_image_model
    # will attempt to load again
    assert manager._image_model is None
    assert manager._video_predictor is None
