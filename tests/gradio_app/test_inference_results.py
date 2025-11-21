"""Tests for inference results module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.gradio_app.inference_results import ImageInferenceResult


def test_image_inference_result_creation():
    """Test basic ImageInferenceResult creation."""
    masks = np.random.randint(0, 2, (3, 100, 80), dtype=np.uint8) * 255
    boxes = np.array(
        [[10, 20, 50, 60], [30, 40, 70, 90], [15, 25, 55, 65]], dtype=np.float32
    )
    scores = np.array([0.95, 0.87, 0.92], dtype=np.float32)
    labels = ["object1", "object2", "object3"]
    meta = {"image_size": (80, 100), "prompt": "test objects"}

    result = ImageInferenceResult(
        masks=masks,
        boxes=boxes,
        scores=scores,
        labels=labels,
        meta=meta,
    )

    assert result.masks.shape == (3, 100, 80)
    assert result.boxes.shape == (3, 4)
    assert result.scores.shape == (3,)
    assert result.labels == labels
    assert result.meta == meta
    assert result.overlay_image is None


def test_image_inference_result_invalid_shapes():
    """Test that ImageInferenceResult validates input shapes."""
    masks = np.random.randint(0, 2, (3, 100, 80), dtype=np.uint8) * 255
    boxes = np.array(
        [[10, 20, 50, 60], [30, 40, 70, 90]], dtype=np.float32
    )  # Wrong count
    scores = np.array([0.95, 0.87, 0.92], dtype=np.float32)

    with pytest.raises(ValueError, match="must have the same number"):
        ImageInferenceResult(masks=masks, boxes=boxes, scores=scores)


def test_image_inference_result_build_overlay():
    """Test building overlay image from inference results."""
    # Create a simple test image
    orig_image = Image.new("RGB", (80, 100), color=(128, 128, 128))

    # Create sample masks (3 objects)
    masks = np.zeros((3, 100, 80), dtype=np.uint8)
    masks[0, 20:40, 10:30] = 255  # Object 1
    masks[1, 50:70, 40:60] = 255  # Object 2
    masks[2, 30:50, 50:70] = 255  # Object 3

    boxes = np.array(
        [[10, 20, 30, 40], [40, 50, 60, 70], [50, 30, 70, 50]], dtype=np.float32
    )
    scores = np.array([0.95, 0.87, 0.92], dtype=np.float32)
    labels = ["obj1", "obj2", "obj3"]

    result = ImageInferenceResult(
        masks=masks,
        boxes=boxes,
        scores=scores,
        labels=labels,
        meta={"image_size": (80, 100)},
    )

    overlay_image = result.build_overlay_image(orig_image)

    # Verify overlay image has same size as original
    assert overlay_image.size == orig_image.size
    # Verify it's a PIL Image
    assert isinstance(overlay_image, Image.Image)


def test_image_inference_result_build_overlay_no_labels():
    """Test building overlay image without labels."""
    orig_image = Image.new("RGB", (80, 100), color=(255, 255, 255))

    masks = np.zeros((2, 100, 80), dtype=np.uint8)
    masks[0, 10:30, 10:30] = 255
    masks[1, 40:60, 40:60] = 255

    boxes = np.array([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)

    result = ImageInferenceResult(
        masks=masks,
        boxes=boxes,
        scores=scores,
        labels=None,
        meta={},
    )

    overlay_image = result.build_overlay_image(orig_image)
    assert isinstance(overlay_image, Image.Image)
    assert overlay_image.size == orig_image.size


def test_image_inference_result_to_download_zip():
    """Test creating download ZIP file."""
    masks = np.random.randint(0, 2, (3, 100, 80), dtype=np.uint8) * 255
    boxes = np.array(
        [[10, 20, 50, 60], [30, 40, 70, 90], [15, 25, 55, 65]], dtype=np.float32
    )
    scores = np.array([0.95, 0.87, 0.92], dtype=np.float32)

    result = ImageInferenceResult(
        masks=masks,
        boxes=boxes,
        scores=scores,
        labels=["a", "b", "c"],
        meta={},
    )

    # Need to provide temp_manager
    from scripts.gradio_app.temp_file_manager import TempFileManager

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_manager = TempFileManager(temp_dir=tmpdir)
        zip_path = result.to_download_zip(temp_manager)

        # Verify ZIP file exists
        assert Path(zip_path).exists()
        assert zip_path.endswith(".zip")

        # Verify it's in the temp directory
        assert Path(zip_path).parent == Path(tmpdir)


def test_image_inference_result_empty_masks():
    """Test that empty masks raise ValueError."""
    masks = np.array([], dtype=np.uint8).reshape(0, 10, 10)
    boxes = np.array([], dtype=np.float32).reshape(0, 4)
    scores = np.array([], dtype=np.float32)

    with pytest.raises(ValueError, match="must have at least one"):
        ImageInferenceResult(masks=masks, boxes=boxes, scores=scores)


def test_image_inference_result_get_count():
    """Test getting the count of detected objects."""
    masks = np.random.randint(0, 2, (5, 50, 40), dtype=np.uint8) * 255
    boxes = np.random.rand(5, 4).astype(np.float32) * 50
    scores = np.array([0.9, 0.85, 0.92, 0.88, 0.91], dtype=np.float32)

    result = ImageInferenceResult(
        masks=masks,
        boxes=boxes,
        scores=scores,
    )

    assert result.get_count() == 5
