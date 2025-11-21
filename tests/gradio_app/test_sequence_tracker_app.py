"""Tests for sequence tracker app module."""

import os
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image
from unittest.mock import Mock, MagicMock

from scripts.gradio_app.sequence_tracker_app import SequenceTrackerApp


def create_test_video(num_frames=10, width=320, height=240, fps=30.0):
    """Create a small test video file.

    Args:
        num_frames: Number of frames to generate.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.

    Returns:
        Path to the created video file.
    """
    fd, video_path = tempfile.mkstemp(suffix=".mp4", prefix="test_video_")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a simple gradient frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 255 // num_frames  # Blue channel gradient
        out.write(frame)

    out.release()
    return video_path


def test_sequence_tracker_app_init():
    """Test that SequenceTrackerApp initializes correctly."""
    app = SequenceTrackerApp()

    # Verify managers are initialized
    assert app.model_manager is not None
    assert app.temp_manager is not None
    assert app.prompt_state is not None

    # Verify initial state
    assert app.current_result is None
    assert app.video_path is None
    assert app.video_metadata is None


def test_set_video_metadata():
    """Test set_video method retrieves correct metadata."""
    app = SequenceTrackerApp()

    # Create a test video
    video_path = create_test_video(num_frames=10, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)

        # Verify video path is stored
        assert app.video_path == video_path

        # Verify metadata is extracted
        assert app.video_metadata is not None
        assert "fps" in app.video_metadata
        assert "num_frames" in app.video_metadata
        assert "width" in app.video_metadata
        assert "height" in app.video_metadata

        # Verify metadata values
        assert app.video_metadata["fps"] == 30.0
        assert app.video_metadata["num_frames"] == 10
        assert app.video_metadata["width"] == 320
        assert app.video_metadata["height"] == 240
    finally:
        # Clean up test video
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_set_video_nonexistent():
    """Test that ValueError is raised for nonexistent video file."""
    app = SequenceTrackerApp()

    with pytest.raises(ValueError, match="Video file not found"):
        app.set_video("/nonexistent/video.mp4")


def test_set_prompt_frame():
    """Test set_prompt_frame method extracts correct frame."""
    app = SequenceTrackerApp()

    # Create a test video
    video_path = create_test_video(num_frames=10, width=320, height=240, fps=30.0)

    try:
        # Set video first
        app.set_video(video_path)

        # Extract frame 5
        frame_image = app.set_prompt_frame(frame_idx=5)

        # Verify frame is returned as PIL Image
        assert isinstance(frame_image, Image.Image)
        assert frame_image.size == (320, 240)

        # Verify prompt_state is updated
        assert app.prompt_state.prompt_frame_idx == 5
    finally:
        # Clean up test video
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_set_prompt_frame_no_video():
    """Test that ValueError is raised when no video is loaded."""
    app = SequenceTrackerApp()

    with pytest.raises(ValueError, match="No video loaded"):
        app.set_prompt_frame(frame_idx=0)


def test_set_prompt_frame_invalid_index():
    """Test that ValueError is raised for invalid frame index."""
    app = SequenceTrackerApp()

    # Create a test video with 10 frames
    video_path = create_test_video(num_frames=10, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)

        # Try to extract frame beyond range
        with pytest.raises(ValueError, match="Frame index out of range"):
            app.set_prompt_frame(frame_idx=15)
    finally:
        # Clean up test video
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_add_point():
    """Test add_point method."""
    app = SequenceTrackerApp()

    # Create and load video
    video_path = create_test_video(num_frames=10, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        app.set_prompt_frame(frame_idx=5)

        # Add a point
        app.add_point(x=100.0, y=150.0, label=1)

        # Verify point is added to prompt_state
        assert len(app.prompt_state.points) == 1
        assert app.prompt_state.points[0].x == 100.0
        assert app.prompt_state.points[0].y == 150.0
        assert app.prompt_state.points[0].label == 1
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_add_box():
    """Test add_box method."""
    app = SequenceTrackerApp()

    # Create and load video
    video_path = create_test_video(num_frames=10, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        app.set_prompt_frame(frame_idx=5)

        # Add a box
        app.add_box(x1=50.0, y1=50.0, x2=150.0, y2=150.0, label=1)

        # Verify box is added to prompt_state
        assert len(app.prompt_state.boxes) == 1
        assert app.prompt_state.boxes[0].x1 == 50.0
        assert app.prompt_state.boxes[0].y1 == 50.0
        assert app.prompt_state.boxes[0].x2 == 150.0
        assert app.prompt_state.boxes[0].y2 == 150.0
        assert app.prompt_state.boxes[0].label == 1
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_export_masks_zip():
    """Test export_masks_zip method."""
    app = SequenceTrackerApp()

    # Create a mock result
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        1: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }
    from scripts.gradio_app.inference_results import VideoInferenceResult

    result = VideoInferenceResult(
        masks_per_frame=masks_per_frame,
        obj_ids=[1, 2],
        scores_per_frame={0: np.array([0.95, 0.87]), 1: np.array([0.94, 0.86])},
        meta={},
    )
    app.current_result = result

    # Export masks
    zip_path = app.export_masks_zip()

    # Verify ZIP file exists
    from pathlib import Path

    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")

    # Clean up
    Path(zip_path).unlink()


def test_export_tracks_json():
    """Test export_tracks_json method."""
    app = SequenceTrackerApp()

    # Create a mock result
    masks_per_frame = {
        0: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
        1: np.random.randint(0, 2, (2, 100, 80), dtype=np.uint8) * 255,
    }
    from scripts.gradio_app.inference_results import VideoInferenceResult

    result = VideoInferenceResult(
        masks_per_frame=masks_per_frame,
        obj_ids=[1, 2],
        scores_per_frame={0: np.array([0.95, 0.87]), 1: np.array([0.94, 0.86])},
        labels=["obj1", "obj2"],
        meta={},
    )
    app.current_result = result

    # Export tracks
    json_path = app.export_tracks_json()

    # Verify JSON file exists
    from pathlib import Path

    assert Path(json_path).exists()
    assert json_path.endswith(".json")

    # Clean up
    Path(json_path).unlink()


def test_export_no_result():
    """Test that ValueError is raised when no result is available."""
    app = SequenceTrackerApp()

    with pytest.raises(ValueError, match="No tracking result available"):
        app.export_masks_zip()

    with pytest.raises(ValueError, match="No tracking result available"):
        app.export_tracks_json()


def test_preview_tracking():
    """Test preview_tracking method with mocked video predictor."""
    app = SequenceTrackerApp()

    # Create and load video
    video_path = create_test_video(num_frames=50, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        app.set_prompt_frame(frame_idx=10)
        app.add_point(x=160.0, y=120.0, label=1)

        # Mock the video predictor to avoid actual inference
        mock_predictor = MagicMock()
        mock_session_id = "test_session_123"

        # Mock start_session response
        mock_predictor.handle_request.return_value = {"session_id": mock_session_id}

        # Mock propagate_in_video streaming response
        mock_outputs = {
            "masks": np.ones((1, 240, 320), dtype=bool),
            "obj_ids": [1],
            "scores": np.array([0.95]),
        }

        def mock_stream_generator():
            for frame_idx in range(10, 20):  # Preview 10 frames
                yield {
                    "frame_index": frame_idx,
                    "outputs": mock_outputs,
                }

        mock_predictor.handle_stream_request.return_value = mock_stream_generator()

        # Replace model_manager's get_video_predictor with mock
        app.model_manager.get_video_predictor = Mock(return_value=mock_predictor)

        # Run preview tracking (10 frames)
        result = app.preview_tracking(max_frames=10)

        # Verify result is VideoInferenceResult
        from scripts.gradio_app.inference_results import VideoInferenceResult

        assert isinstance(result, VideoInferenceResult)

        # Verify current_result is updated
        assert app.current_result is not None

    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_preview_tracking_no_prompts():
    """Test that ValueError is raised when no prompts are added."""
    app = SequenceTrackerApp()

    # Create and load video
    video_path = create_test_video(num_frames=50, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        app.set_prompt_frame(frame_idx=10)
        # Don't add any prompts

        with pytest.raises(ValueError, match="No prompts added"):
            app.preview_tracking(max_frames=10)
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_run_tracking():
    """Test run_tracking method with mocked video predictor."""
    app = SequenceTrackerApp()

    # Create and load video
    video_path = create_test_video(num_frames=30, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        app.set_prompt_frame(frame_idx=10)
        app.add_point(x=160.0, y=120.0, label=1)

        # Mock the video predictor to avoid actual inference
        mock_predictor = MagicMock()
        mock_session_id = "test_session_456"

        # Mock start_session response
        mock_predictor.handle_request.return_value = {"session_id": mock_session_id}

        # Mock propagate_in_video streaming response
        mock_outputs = {
            "masks": np.ones((1, 240, 320), dtype=bool),
            "obj_ids": [1],
            "scores": np.array([0.95]),
        }

        def mock_stream_generator():
            for frame_idx in range(30):  # All frames
                yield {
                    "frame_index": frame_idx,
                    "outputs": mock_outputs,
                }

        mock_predictor.handle_stream_request.return_value = mock_stream_generator()

        # Replace model_manager's get_video_predictor with mock
        app.model_manager.get_video_predictor = Mock(return_value=mock_predictor)

        # Run full tracking
        result = app.run_tracking()

        # Verify result is VideoInferenceResult
        from scripts.gradio_app.inference_results import VideoInferenceResult

        assert isinstance(result, VideoInferenceResult)

        # Verify all frames are processed
        assert len(result.masks_per_frame) == 30

        # Verify current_result is updated
        assert app.current_result is not None

    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


@pytest.mark.slow
def test_run_tracking_with_real_model():
    """Test run_tracking method with real SAM3 model (slow test)."""
    app = SequenceTrackerApp()

    # Create a small test video
    video_path = create_test_video(num_frames=5, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        app.set_prompt_frame(frame_idx=2)
        app.add_point(x=160.0, y=120.0, label=1)

        # Run tracking with real model
        result = app.run_tracking()

        # Verify result is VideoInferenceResult
        from scripts.gradio_app.inference_results import VideoInferenceResult

        assert isinstance(result, VideoInferenceResult)

        # Verify all frames are processed
        assert len(result.masks_per_frame) >= 1

    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_run_tracking_stream():
    """Test run_tracking_stream method with mocked video predictor."""
    app = SequenceTrackerApp()

    # Create and load video
    video_path = create_test_video(num_frames=20, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        app.set_prompt_frame(frame_idx=5)
        app.add_point(x=160.0, y=120.0, label=1)

        # Mock the video predictor to avoid actual inference
        mock_predictor = MagicMock()
        mock_session_id = "test_session_789"

        # Mock start_session response
        mock_predictor.handle_request.return_value = {"session_id": mock_session_id}

        # Mock propagate_in_video streaming response
        mock_outputs = {
            "masks": np.ones((1, 240, 320), dtype=bool),
            "obj_ids": [1],
            "scores": np.array([0.95]),
        }

        def mock_stream_generator():
            for frame_idx in range(20):  # All frames
                yield {
                    "frame_index": frame_idx,
                    "outputs": mock_outputs,
                }

        mock_predictor.handle_stream_request.return_value = mock_stream_generator()

        # Replace model_manager's get_video_predictor with mock
        app.model_manager.get_video_predictor = Mock(return_value=mock_predictor)

        # Run streaming tracking
        progress_updates = []
        result = None
        for update in app.run_tracking_stream():
            if "progress" in update:
                progress_updates.append(update)
            elif "result" in update:
                result = update["result"]

        # Verify progress updates were received
        assert len(progress_updates) > 0

        # Verify result is VideoInferenceResult
        from scripts.gradio_app.inference_results import VideoInferenceResult

        assert isinstance(result, VideoInferenceResult)
        assert len(result.masks_per_frame) == 20

        # Verify current_result is updated
        assert app.current_result is not None

    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_bidirectional_tracking():
    """Test bidirectional tracking with direction parameter."""
    app = SequenceTrackerApp()

    # Create and load video (30 frames)
    video_path = create_test_video(num_frames=30, width=320, height=240, fps=30.0)

    try:
        app.set_video(video_path)
        # Set prompt frame in the middle (frame 15)
        app.set_prompt_frame(frame_idx=15)
        app.add_point(x=160.0, y=120.0, label=1)

        # Mock the video predictor
        mock_predictor = MagicMock()
        mock_session_id = "test_session_bidirectional"

        # Mock start_session response
        mock_predictor.handle_request.return_value = {"session_id": mock_session_id}

        # Mock propagate_in_video streaming response
        mock_outputs = {
            "masks": np.ones((1, 240, 320), dtype=bool),
            "obj_ids": [1],
            "scores": np.array([0.95]),
        }

        def mock_stream_generator():
            for frame_idx in range(30):  # All frames
                yield {
                    "frame_index": frame_idx,
                    "outputs": mock_outputs,
                }

        mock_predictor.handle_stream_request.return_value = mock_stream_generator()

        # Replace model_manager's get_video_predictor with mock
        app.model_manager.get_video_predictor = Mock(return_value=mock_predictor)

        # Test bidirectional tracking
        result = app.run_tracking(direction="both")

        # Verify result is VideoInferenceResult
        from scripts.gradio_app.inference_results import VideoInferenceResult

        assert isinstance(result, VideoInferenceResult)

        # Verify all frames are processed (both forward and backward from frame 15)
        assert len(result.masks_per_frame) == 30

        # Verify current_result is updated
        assert app.current_result is not None

    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
