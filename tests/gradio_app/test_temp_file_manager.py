# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for temporary file manager module."""

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.gradio_app.temp_file_manager import TempFileManager


def test_temp_file_manager_init():
    """Test that TempFileManager initializes correctly."""
    manager = TempFileManager()

    assert manager.temp_dir is not None
    assert Path(manager.temp_dir).exists()


def test_create_mask_zip():
    """Test creating a ZIP file from mask arrays."""
    manager = TempFileManager()

    # Create sample masks (3 masks, 100x80 each)
    masks = [
        np.random.randint(0, 2, (100, 80), dtype=np.uint8) * 255,
        np.random.randint(0, 2, (100, 80), dtype=np.uint8) * 255,
        np.random.randint(0, 2, (100, 80), dtype=np.uint8) * 255,
    ]

    zip_path = manager.create_mask_zip(masks, prefix="test_masks")

    # Verify ZIP file was created
    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")

    # Verify ZIP contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list = zf.namelist()
        assert len(file_list) == 3
        assert "mask_0.png" in file_list
        assert "mask_1.png" in file_list
        assert "mask_2.png" in file_list

    # Clean up
    Path(zip_path).unlink()


def test_create_mask_zip_empty():
    """Test that creating ZIP with empty masks raises ValueError."""
    manager = TempFileManager()

    with pytest.raises(ValueError, match="masks cannot be empty"):
        manager.create_mask_zip([])


def test_create_temp_video_from_frames():
    """Test creating a temporary video from frame arrays."""
    manager = TempFileManager()

    # Create sample frames (5 frames, 64x48 RGB)
    frames = []
    for i in range(5):
        frame = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        frames.append(frame)

    video_path = manager.create_temp_video_from_frames(
        frames, fps=10.0, prefix="test_video"
    )

    # Verify video file was created
    assert Path(video_path).exists()
    assert video_path.endswith(".mp4")

    # Clean up
    Path(video_path).unlink()


def test_create_temp_video_empty_frames():
    """Test that creating video with empty frames raises ValueError."""
    manager = TempFileManager()

    with pytest.raises(ValueError, match="frames cannot be empty"):
        manager.create_temp_video_from_frames([], fps=10.0)


def test_create_temp_video_invalid_fps():
    """Test that creating video with invalid FPS raises ValueError."""
    manager = TempFileManager()
    frames = [np.zeros((48, 64, 3), dtype=np.uint8)]

    with pytest.raises(ValueError, match="fps must be positive"):
        manager.create_temp_video_from_frames(frames, fps=0)


def test_cleanup_old_files():
    """Test cleanup of old temporary files."""
    manager = TempFileManager()

    # Create some temporary files
    temp_file1 = Path(manager.temp_dir) / "test_file_1.txt"
    temp_file2 = Path(manager.temp_dir) / "test_file_2.txt"
    temp_file1.write_text("test content 1")
    temp_file2.write_text("test content 2")

    # Verify files exist
    assert temp_file1.exists()
    assert temp_file2.exists()

    # Cleanup with max_age=0 (should delete all files)
    deleted_count = manager.cleanup_old_files(max_age_seconds=0)

    # At least the 2 test files should be deleted
    assert deleted_count >= 2
    # Our test files should be gone
    assert not temp_file1.exists()
    assert not temp_file2.exists()


def test_cleanup_old_files_respects_age():
    """Test that cleanup respects max_age_seconds parameter."""
    manager = TempFileManager()

    # Create a temporary file
    temp_file = Path(manager.temp_dir) / "test_file_new.txt"
    temp_file.write_text("test content")

    # Cleanup with very large max_age (should not delete newly created file)
    manager.cleanup_old_files(max_age_seconds=3600)

    # File should still exist (it was just created)
    assert temp_file.exists()

    # Clean up
    temp_file.unlink()


def test_temp_file_manager_custom_dir():
    """Test TempFileManager with custom temp directory."""
    with tempfile.TemporaryDirectory() as custom_temp:
        manager = TempFileManager(temp_dir=custom_temp)

        assert manager.temp_dir == custom_temp
        assert Path(manager.temp_dir).exists()

        # Create a mask zip in custom directory
        masks = [np.ones((10, 10), dtype=np.uint8) * 255]
        zip_path = manager.create_mask_zip(masks, prefix="custom")

        # Verify it's in the custom directory
        assert Path(zip_path).parent == Path(custom_temp)

        # Clean up
        Path(zip_path).unlink()
