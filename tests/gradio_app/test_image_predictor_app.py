"""Tests for image predictor app module."""

import pytest
from PIL import Image
import numpy as np
from unittest.mock import Mock
import torch

from scripts.gradio_app.image_predictor_app import ImagePredictorApp
from scripts.gradio_app.prompt_types import PromptMode


def test_image_predictor_app_init():
    """Test that ImagePredictorApp initializes correctly."""
    app = ImagePredictorApp()

    # Verify managers are initialized
    assert app.model_manager is not None
    assert app.temp_manager is not None
    assert app.prompt_state is not None

    # Verify initial state
    assert app.current_result is None


def test_set_image():
    """Test setting an image."""
    app = ImagePredictorApp()

    # Create a test image
    img = Image.new("RGB", (800, 600), color=(128, 128, 128))
    app.set_image(img)

    # Verify image is set in prompt_state
    assert app.prompt_state.image is not None
    assert app.prompt_state.orig_size == (800, 600)


def test_set_image_with_resize():
    """Test setting a large image that requires resize."""
    app = ImagePredictorApp()

    # Create a large image > 2048px
    img = Image.new("RGB", (3000, 2500), color=(255, 0, 0))
    app.set_image(img)

    # Verify image is resized
    assert app.prompt_state.image is not None
    assert app.prompt_state.orig_size == (3000, 2500)
    assert app.prompt_state.scale_factor < 1.0
    assert max(app.prompt_state.scaled_size) <= 2048


def test_set_prompt_mode():
    """Test setting prompt mode."""
    app = ImagePredictorApp()

    # Set to TEXT_ONLY
    app.set_prompt_mode(PromptMode.TEXT_ONLY)
    assert app.prompt_state.get_prompt_mode() == PromptMode.TEXT_ONLY

    # Set to POINTS_ONLY
    app.set_prompt_mode(PromptMode.POINTS_ONLY)
    assert app.prompt_state.get_prompt_mode() == PromptMode.POINTS_ONLY


def test_add_point():
    """Test adding a point prompt."""
    app = ImagePredictorApp()

    # Set an image first
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    app.set_image(img)

    # Add a point
    app.add_point(50, 40, label=1)

    # Verify point is added
    assert len(app.prompt_state.points) == 1
    assert app.prompt_state.points[0].x == 50
    assert app.prompt_state.points[0].y == 40
    assert app.prompt_state.points[0].label == 1


def test_add_box():
    """Test adding a box prompt."""
    app = ImagePredictorApp()

    # Set an image first
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    app.set_image(img)

    # Add a box
    app.add_box(10, 20, 50, 60, label=1)

    # Verify box is added
    assert len(app.prompt_state.boxes) == 1
    assert app.prompt_state.boxes[0].x1 == 10
    assert app.prompt_state.boxes[0].y1 == 20
    assert app.prompt_state.boxes[0].x2 == 50
    assert app.prompt_state.boxes[0].y2 == 60
    assert app.prompt_state.boxes[0].label == 1


def test_clear_prompts():
    """Test clearing prompts."""
    app = ImagePredictorApp()

    # Set image and add prompts
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    app.set_image(img)
    app.prompt_state.set_text_prompt("test")
    app.add_point(50, 40, 1)
    app.add_box(10, 20, 50, 60, 1)

    # Clear prompts
    app.clear_prompts()

    # Verify prompts are cleared but image remains
    assert app.prompt_state.text_prompt == ""
    assert len(app.prompt_state.points) == 0
    assert len(app.prompt_state.boxes) == 0
    assert app.prompt_state.image is not None  # Image should remain


def test_reset_session():
    """Test resetting the session."""
    app = ImagePredictorApp()

    # Set image and add prompts
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    app.set_image(img)
    app.prompt_state.set_text_prompt("test")
    app.add_point(50, 40, 1)

    # Create a fake result
    app.current_result = "fake_result"

    # Reset session
    app.reset_session()

    # Verify everything is cleared
    assert app.prompt_state.text_prompt == ""
    assert len(app.prompt_state.points) == 0
    assert app.prompt_state.image is None
    assert app.current_result is None


def test_run_segmentation_text_only_mocked():
    """Test run_segmentation with text-only prompt (mocked model)."""
    app = ImagePredictorApp()

    # Set image
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    app.set_image(img)

    # Set text prompt
    app.prompt_state.set_text_prompt("test object")
    app.set_prompt_mode(PromptMode.TEXT_ONLY)

    # Mock the model and processor
    mock_model = Mock()
    mock_processor = Mock()

    # Create fake results
    fake_masks = torch.rand(2, 1, 80, 100) > 0.5  # 2 objects
    fake_boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0], [30.0, 40.0, 70.0, 80.0]])
    fake_scores = torch.tensor([0.95, 0.87])

    fake_state = {
        "masks": fake_masks,
        "boxes": fake_boxes,
        "scores": fake_scores,
    }

    mock_processor.set_image.return_value = fake_state
    mock_processor.set_text_prompt.return_value = fake_state

    # Mock model manager to return our mock model
    app.model_manager.get_image_model = Mock(return_value=mock_model)

    # Inject mock processor
    app._create_processor = Mock(return_value=mock_processor)

    result = app.run_segmentation()

    # Verify result
    assert result is not None
    assert result.get_count() == 2
    assert result.scores[0] == pytest.approx(0.95)
    assert app.current_result is result


def test_run_segmentation_no_image():
    """Test that run_segmentation raises error when no image is set."""
    app = ImagePredictorApp()

    # Set prompt but no image
    app.prompt_state.set_text_prompt("test")

    with pytest.raises(ValueError, match="No image set"):
        app.run_segmentation()


def test_run_segmentation_no_prompts():
    """Test that run_segmentation raises error when no prompts are set."""
    app = ImagePredictorApp()

    # Set image but no prompts
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    app.set_image(img)

    with pytest.raises(ValueError, match="No prompts"):
        app.run_segmentation()


def test_export_masks_zip():
    """Test exporting masks to ZIP file."""
    from pathlib import Path

    app = ImagePredictorApp()

    # Set image
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    app.set_image(img)
    app.prompt_state.set_text_prompt("test object")

    # Mock run_segmentation to create fake result
    from scripts.gradio_app.inference_results import ImageInferenceResult

    fake_masks = np.random.randint(0, 2, (2, 80, 100), dtype=np.uint8) * 255
    fake_boxes = np.array([[10.0, 20.0, 50.0, 60.0], [30.0, 40.0, 70.0, 80.0]])
    fake_scores = np.array([0.95, 0.87])

    app.current_result = ImageInferenceResult(
        masks=fake_masks,
        boxes=fake_boxes,
        scores=fake_scores,
        labels=["obj1", "obj2"],
        meta={},
    )

    # Export masks
    zip_path = app.export_masks_zip()

    # Verify ZIP file exists
    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")

    # Clean up
    Path(zip_path).unlink()


def test_export_masks_zip_no_result():
    """Test that export_masks_zip raises error when no result exists."""
    app = ImagePredictorApp()

    with pytest.raises(ValueError, match="No segmentation results"):
        app.export_masks_zip()
