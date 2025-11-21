"""Tests for video inference result module."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.gradio_app.inference_results import VideoInferenceResult
from scripts.gradio_app.temp_file_manager import TempFileManager


def test_video_inference_result_creation():
    """Test basic creation of VideoInferenceResult."""
    # Create fake tracking results for 3 frames, 2 objects
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        1: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        2: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }

    obj_ids = [1, 2]
    scores_per_frame = {
        0: np.array([0.95, 0.87]),
        1: np.array([0.94, 0.86]),
        2: np.array([0.93, 0.85]),
    }

    meta = {"video_path": "test.mp4", "fps": 30, "num_frames": 3}

    result = VideoInferenceResult(
        masks_per_frame=masks_per_frame,
        obj_ids=obj_ids,
        scores_per_frame=scores_per_frame,
        meta=meta,
    )

    assert result.masks_per_frame == masks_per_frame
    assert result.obj_ids == obj_ids
    assert result.scores_per_frame == scores_per_frame
    assert result.meta == meta
    assert result.overlay_video_path is None


def test_video_inference_result_empty_frames():
    """Test that ValueError is raised for empty frames."""
    with pytest.raises(
        ValueError, match="masks_per_frame must have at least one frame"
    ):
        VideoInferenceResult(
            masks_per_frame={},
            obj_ids=[1],
            scores_per_frame={},
            meta={},
        )


def test_video_inference_result_inconsistent_frames():
    """Test that ValueError is raised for inconsistent frame indices."""
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        1: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }
    scores_per_frame = {
        0: np.array([0.95, 0.87]),
        # Missing frame 1
    }

    with pytest.raises(ValueError, match="must have the same frame indices"):
        VideoInferenceResult(
            masks_per_frame=masks_per_frame,
            obj_ids=[1, 2],
            scores_per_frame=scores_per_frame,
            meta={},
        )


def test_video_inference_result_get_num_frames():
    """Test get_num_frames method."""
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        1: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        2: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }

    result = VideoInferenceResult(
        masks_per_frame=masks_per_frame,
        obj_ids=[1, 2],
        scores_per_frame={
            0: np.array([0.95, 0.87]),
            1: np.array([0.94, 0.86]),
            2: np.array([0.93, 0.85]),
        },
        meta={},
    )

    assert result.get_num_frames() == 3


def test_video_inference_result_get_num_objects():
    """Test get_num_objects method."""
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }

    result = VideoInferenceResult(
        masks_per_frame=masks_per_frame,
        obj_ids=[1, 2],
        scores_per_frame={0: np.array([0.95, 0.87])},
        meta={},
    )

    assert result.get_num_objects() == 2


def test_video_inference_result_export_tracks_json():
    """Test export_tracks method."""
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        1: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }

    result = VideoInferenceResult(
        masks_per_frame=masks_per_frame,
        obj_ids=[1, 2],
        scores_per_frame={0: np.array([0.95, 0.87]), 1: np.array([0.94, 0.86])},
        labels=["obj1", "obj2"],
        meta={"video_path": "test.mp4"},
    )

    temp_manager = TempFileManager()
    json_path = result.export_tracks(temp_manager)

    # Verify JSON file exists
    assert Path(json_path).exists()
    assert json_path.endswith(".json")

    # Verify JSON content
    with open(json_path, "r") as f:
        data = json.load(f)

    assert "meta" in data
    assert "tracks" in data
    assert len(data["tracks"]) == 2
    assert data["tracks"][0]["obj_id"] == 1
    assert data["tracks"][1]["obj_id"] == 2

    # Clean up
    Path(json_path).unlink()


def test_video_inference_result_to_download_zip():
    """Test to_download_zip method."""
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        1: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }

    result = VideoInferenceResult(
        masks_per_frame=masks_per_frame,
        obj_ids=[1, 2],
        scores_per_frame={0: np.array([0.95, 0.87]), 1: np.array([0.94, 0.86])},
        meta={},
    )

    temp_manager = TempFileManager()
    zip_path = result.to_download_zip(temp_manager)

    # Verify ZIP file exists
    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")

    # Verify ZIP content (should have masks for 2 frames, 2 objects each)
    with ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Each frame has 2 objects, so 2 frames * 2 objects = 4 mask files
        assert len(names) == 4
        assert any("frame_0000" in n for n in names)
        assert any("frame_0001" in n for n in names)

    # Clean up
    Path(zip_path).unlink()
