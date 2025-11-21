"""Integration tests for SAM3 Gradio Application.

Tests the complete workflow of Image Predictor and Sequence Tracker tabs
with mocked model inference to ensure UI and event handlers work together.
"""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from scripts.gradio_app.sam3_gradio_app import Sam3GradioApp


def create_test_image(width: int = 100, height: int = 100) -> Image.Image:
    """Create a test image for integration testing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        PIL Image instance.
    """
    return Image.new("RGB", (width, height), color="red")


def create_test_video(num_frames: int = 10, width: int = 100, height: int = 100) -> str:
    """Create a test video file for integration testing.

    Args:
        num_frames: Number of frames in the video.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Path to the created video file.
    """
    import cv2

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, 10.0, (width, height))

    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)

    out.release()
    return temp_path


def test_image_predictor_workflow():
    """Test complete Image Predictor workflow: upload -> segmentation -> clear -> reset."""
    app = Sam3GradioApp()

    # Step 1: Upload image
    test_image = create_test_image(256, 256)
    result = app._on_image_upload(test_image)

    # Verify image is set
    assert result is not None
    assert app.image_predictor.prompt_state.image is not None

    # Step 2: Add point directly to image predictor (bypassing UI event handler)
    app.image_predictor.add_point(x=128, y=128, label=True)
    assert len(app.image_predictor.prompt_state.points) == 1

    # Step 3: Run segmentation (with mocked model)
    with patch.object(app.image_predictor, "run_segmentation") as mock_run:
        # Mock return: overlay_image, masks
        mock_result = Mock()
        mock_result.overlay_image = create_test_image(256, 256)
        mock_result.masks = [np.ones((256, 256), dtype=bool)]
        mock_run.return_value = mock_result

        seg_result = app._on_run_segmentation()

        # Verify segmentation was called
        mock_run.assert_called_once()

        # Verify result contains gallery images
        assert seg_result is not None
        assert isinstance(seg_result, list)
        assert len(seg_result) > 0

    # Step 4: Clear prompts
    clear_result = app._on_clear_prompts()

    # Verify prompts are cleared
    assert clear_result is not None
    assert len(app.image_predictor.prompt_state.points) == 0

    # Step 5: Reset session
    reset_result = app._on_reset_session()

    # Verify session is reset
    assert reset_result is not None
    assert app.image_predictor.prompt_state.image is None


def test_sequence_tracker_workflow():
    """Test complete Sequence Tracker workflow: upload -> extract -> prompt -> track -> result."""
    import os

    app = Sam3GradioApp()

    # Step 1: Upload video
    test_video_path = create_test_video(num_frames=10, width=256, height=256)

    try:
        video_result = app._on_video_upload(test_video_path)

        # Verify video is set and metadata is returned
        assert video_result is not None
        assert app.sequence_tracker.video_path is not None
        assert app.sequence_tracker.video_metadata is not None

        # Step 2: Extract frame
        frame_result = app._on_extract_frame(frame_index=0)

        # Verify frame is extracted
        assert frame_result is not None
        assert isinstance(frame_result, Image.Image)

        # Step 3: Add point directly to sequence tracker (bypassing UI event handler)
        app.sequence_tracker.add_point(x=128, y=128, label=1)
        assert len(app.sequence_tracker.prompt_state.points) == 1

        # Step 4: Preview tracking (with mocked model)
        with patch.object(app.sequence_tracker, "preview_tracking") as mock_preview:
            # Mock return: video path
            temp_preview_video = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            )
            mock_preview.return_value = temp_preview_video.name

            preview_result = app._on_preview_tracking()

            # Verify preview was called
            mock_preview.assert_called_once_with(num_frames=10)

            # Verify result contains video path
            assert preview_result is not None

            # Clean up preview video
            os.remove(temp_preview_video.name)

        # Step 5: Track video (with mocked model)
        with patch.object(app.sequence_tracker, "run_tracking_stream") as mock_track:
            # Mock generator for tracking
            def mock_tracking_generator():
                yield {
                    "type": "progress",
                    "progress": 0.5,
                    "desc": "Processing frame 5/10",
                }
                temp_track_video = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                )
                yield {"type": "result", "video_path": temp_track_video.name}

            mock_track.return_value = mock_tracking_generator()

            track_result = app._on_track_video()

            # Verify tracking was called
            mock_track.assert_called_once()

            # Verify result contains video path and download button visibility
            assert track_result is not None
            assert isinstance(track_result, tuple)
            assert len(track_result) == 3

            # Clean up track video if it exists
            if isinstance(track_result[0], str) and os.path.exists(track_result[0]):
                os.remove(track_result[0])

    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def test_cross_tab_independence():
    """Test that Image Predictor and Sequence Tracker tabs are independent."""
    app = Sam3GradioApp()

    # Upload image to Image Predictor
    test_image = create_test_image(128, 128)
    app._on_image_upload(test_image)

    # Verify Image Predictor has image
    assert app.image_predictor.prompt_state.image is not None

    # Verify Sequence Tracker is still empty
    assert app.sequence_tracker.video_path is None

    # Upload video to Sequence Tracker
    import os

    test_video_path = create_test_video(num_frames=5, width=128, height=128)

    try:
        app._on_video_upload(test_video_path)

        # Verify Sequence Tracker has video
        assert app.sequence_tracker.video_path is not None

        # Verify Image Predictor still has image (not affected)
        assert app.image_predictor.prompt_state.image is not None

    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
