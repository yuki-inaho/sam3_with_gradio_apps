"""Tests for Sam3GradioApp."""

from unittest.mock import Mock, patch

import gradio as gr
import pytest
from PIL import Image as PILImage

from scripts.gradio_app.sam3_gradio_app import Sam3GradioApp


def test_sam3_gradio_app_init():
    """Test Sam3GradioApp initialization."""
    app = Sam3GradioApp()

    # Verify that ImagePredictorApp and SequenceTrackerApp are initialized
    assert app.image_predictor is not None
    assert app.sequence_tracker is not None

    # Verify shared model_manager
    assert app.image_predictor.model_manager is app.sequence_tracker.model_manager


def test_build_image_tab():
    """Test build_image_tab method creates UI components."""
    app = Sam3GradioApp()

    # Build the Image Predictor tab within gr.Blocks context
    with gr.Blocks():
        components = app.build_image_tab()

        # Verify that components dictionary is returned
        assert isinstance(components, dict)

        # Verify essential components exist
        assert "input_image" in components
        assert "text_prompt" in components
        assert "prompt_mode" in components
        assert "run_button" in components
        assert "clear_button" in components
        assert "reset_button" in components
        assert "output_gallery" in components

        # Verify component types
        assert isinstance(components["input_image"], gr.Image)
        assert isinstance(components["text_prompt"], gr.Textbox)
        assert isinstance(components["prompt_mode"], gr.Radio)
        assert isinstance(components["run_button"], gr.Button)
        assert isinstance(components["clear_button"], gr.Button)
        assert isinstance(components["reset_button"], gr.Button)
        assert isinstance(components["output_gallery"], gr.Gallery)


def test_build_sequence_tab():
    """Test build_sequence_tab method creates UI components."""
    app = Sam3GradioApp()

    # Build the Sequence Tracker tab within gr.Blocks context
    with gr.Blocks():
        components = app.build_sequence_tab()

        # Verify that components dictionary is returned
        assert isinstance(components, dict)

        # Verify essential components exist
        assert "input_type" in components
        assert "input_video" in components
        assert "input_images" in components
        assert "frame_slider" in components
        assert "extract_frame_button" in components
        assert "prompt_frame_image" in components
        assert "text_prompt" in components
        assert "prompt_mode" in components
        assert "preview_button" in components
        assert "track_button" in components
        assert "clear_button" in components
        assert "reset_button" in components
        assert "output_video" in components

        # Verify component types
        assert isinstance(components["input_type"], gr.Radio)
        assert isinstance(components["input_video"], gr.Video)
        assert isinstance(components["input_images"], gr.File)
        assert isinstance(components["frame_slider"], gr.Slider)
        assert isinstance(components["extract_frame_button"], gr.Button)
        assert isinstance(components["prompt_frame_image"], gr.Image)
        assert isinstance(components["text_prompt"], gr.Textbox)
        assert isinstance(components["prompt_mode"], gr.Radio)
        assert isinstance(components["preview_button"], gr.Button)
        assert isinstance(components["track_button"], gr.Button)
        assert isinstance(components["clear_button"], gr.Button)
        assert isinstance(components["reset_button"], gr.Button)
        assert isinstance(components["output_video"], gr.Video)


def test_build_blocks():
    """Test build_blocks method creates complete UI with 2 tabs."""
    app = Sam3GradioApp()

    # Build the complete Blocks UI
    blocks = app.build_blocks()

    # Verify that gr.Blocks instance is returned
    assert isinstance(blocks, gr.Blocks)

    # Verify that image_components and sequence_components are stored
    assert hasattr(app, "image_components")
    assert hasattr(app, "sequence_components")

    # Verify that components dictionaries are not empty
    assert len(app.image_components) > 0
    assert len(app.sequence_components) > 0

    # Verify essential components exist in image tab
    assert "input_image" in app.image_components
    assert "run_button" in app.image_components

    # Verify essential components exist in sequence tab
    assert "input_type" in app.sequence_components
    assert "track_button" in app.sequence_components


def test_image_point_click_handling():
    """Test Point mode image click handling with select event."""
    app = Sam3GradioApp()

    # Create a test image
    test_image = PILImage.new("RGB", (100, 100), color="red")

    # Mock the image_predictor methods
    app.image_predictor.set_image = Mock()
    app.image_predictor.add_point = Mock()
    app.image_predictor.get_image_with_overlay = Mock(return_value=test_image)

    # Upload image
    app._on_image_upload(test_image)
    assert app.image_predictor.set_image.called

    # Set prompt mode to POINTS_ONLY
    app._on_prompt_mode_change("POINTS_ONLY")

    # Simulate image click at coordinates (50, 50) with positive label
    # SelectData structure: index is the click coordinates
    select_data = type(
        "SelectData", (), {"index": (50, 50), "value": None, "selected": True}
    )()

    # Call the point click handler with "include" point type (positive)
    result_image = app._on_image_click(select_data, point_type="include")

    # Verify that add_point was called with correct coordinates and positive label
    app.image_predictor.add_point.assert_called_once()
    call_args = app.image_predictor.add_point.call_args
    # Point coordinates should be passed with x=50, y=50, label=True
    assert call_args is not None
    assert call_args.kwargs["x"] == 50
    assert call_args.kwargs["y"] == 50
    assert call_args.kwargs["label"] is True

    # Verify that overlay image is returned
    assert result_image is not None
    assert isinstance(result_image, PILImage.Image)


def test_image_box_annotator_handling():
    """Test Box mode BBox annotation handling with gradio-bbox-annotator."""
    app = Sam3GradioApp()

    # Create a test image
    test_image = PILImage.new("RGB", (100, 100), color="blue")

    # Mock the image_predictor methods
    app.image_predictor.set_image = Mock()
    app.image_predictor.add_box = Mock()
    app.image_predictor.get_image_with_overlay = Mock(return_value=test_image)

    # Upload image
    app._on_image_upload(test_image)

    # Set prompt mode to BOXES_ONLY
    app._on_prompt_mode_change("BOXES_ONLY")

    # Simulate BBox annotation data from gradio-bbox-annotator
    # Format: {"image": image_path, "boxes": [(left, top, right, bottom, label), ...]}
    bbox_data = {
        "image": "test_image.jpg",
        "boxes": [
            (10, 20, 50, 60, "object1"),  # left=10, top=20, right=50, bottom=60
        ],
    }

    # Call the bbox annotation handler (to be implemented)
    result_image = app._on_bbox_change(bbox_data)

    # Verify that add_box was called with converted coordinates
    app.image_predictor.add_box.assert_called_once()
    call_args = app.image_predictor.add_box.call_args
    # BBox should be converted from (left, top, right, bottom) to (x, y, w, h)
    # x=10, y=20, w=40 (50-10), h=40 (60-20)
    assert call_args is not None
    assert call_args.kwargs["x"] == 10
    assert call_args.kwargs["y"] == 20
    assert call_args.kwargs["w"] == 40
    assert call_args.kwargs["h"] == 40

    # Verify that overlay image is returned
    assert result_image is not None
    assert isinstance(result_image, PILImage.Image)


def test_image_run_segmentation():
    """Test Run button segmentation execution with Gallery display."""
    app = Sam3GradioApp()

    # Create test images
    test_image = PILImage.new("RGB", (100, 100), color="green")
    overlay_image = PILImage.new("RGB", (100, 100), color="red")
    mask_image = PILImage.new("L", (100, 100), color=128)

    # Mock ImageInferenceResult
    from scripts.gradio_app.inference_results import ImageInferenceResult

    mock_result = Mock(spec=ImageInferenceResult)
    mock_result.overlay_image = overlay_image
    mock_result.masks = [mask_image]

    # Mock the image_predictor methods
    app.image_predictor.set_image = Mock()
    app.image_predictor.run_segmentation = Mock(return_value=mock_result)

    # Upload image
    app._on_image_upload(test_image)

    # Call the Run button handler with text_prompt parameter
    result_gallery = app._on_run_segmentation("test object")

    # Verify that run_segmentation was called
    app.image_predictor.run_segmentation.assert_called_once()

    # Verify that Gallery output is returned (list of images)
    assert result_gallery is not None
    assert isinstance(result_gallery, list)
    assert len(result_gallery) > 0
    # Gallery should contain overlay image and mask images
    assert overlay_image in result_gallery or mask_image in result_gallery


def test_image_clear_reset():
    """Test Clear and Reset button handlers."""
    app = Sam3GradioApp()

    # Create test image
    test_image = PILImage.new("RGB", (100, 100), color="yellow")

    # Mock the image_predictor methods
    app.image_predictor.set_image = Mock()
    app.image_predictor.clear_prompts = Mock()
    app.image_predictor.reset_session = Mock()

    # Upload image
    app._on_image_upload(test_image)

    # Test Clear button (prompts only)
    result_clear = app._on_clear_prompts()

    # Verify that clear_prompts was called
    app.image_predictor.clear_prompts.assert_called_once()
    # Clear should return updates for UI components (text_prompt cleared, etc.)
    assert result_clear is not None

    # Test Reset button (full reset)
    result_reset = app._on_reset_session()

    # Verify that reset_session was called
    app.image_predictor.reset_session.assert_called_once()
    # Reset should return updates for UI components (image, text_prompt, gallery all cleared)
    assert result_reset is not None


# --- Sequence Tracker Tests ---


def test_sequence_input_type_change():
    """Test input_type Radio change handler for visibility switching."""
    app = Sam3GradioApp()

    # Test switching to "Image Sequence"
    result_images = app._on_input_type_change("Image Sequence")

    # Should return gr.update() for input_video and input_images
    assert result_images is not None
    assert isinstance(result_images, tuple)
    assert len(result_images) == 2

    # Test switching to "Video File"
    result_video = app._on_input_type_change("Video File")

    # Should return gr.update() for input_video and input_images
    assert result_video is not None
    assert isinstance(result_video, tuple)
    assert len(result_video) == 2


def test_sequence_video_upload():
    """Test video upload handler with metadata extraction."""
    import tempfile

    import cv2
    import numpy as np

    app = Sam3GradioApp()

    # Mock sequence_tracker methods
    app.sequence_tracker.set_video = Mock()
    app.sequence_tracker.video_metadata = {
        "fps": 30.0,
        "num_frames": 100,
        "width": 640,
        "height": 480,
    }

    # Create a temporary test video
    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video_file.name
    temp_video_file.close()

    # Write a simple test video (10 frames)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (640, 480))
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    # Call the video upload handler
    result = app._on_video_upload(temp_video_path)

    # Verify that set_video was called
    app.sequence_tracker.set_video.assert_called_once_with(temp_video_path)

    # Verify that metadata and slider update are returned
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2  # (metadata_text, slider_update)

    # Clean up
    import os

    os.remove(temp_video_path)


def test_sequence_images_upload():
    """Test image sequence upload handler with temp video conversion."""
    import tempfile

    app = Sam3GradioApp()

    # Create test images
    test_images = []
    for i in range(5):
        temp_image_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        test_image = PILImage.new("RGB", (640, 480), color=(i * 50, 0, 0))
        test_image.save(temp_image_file.name)
        test_images.append(temp_image_file.name)

    # Mock sequence_tracker and temp_manager methods
    app.sequence_tracker.set_video = Mock()
    app.sequence_tracker.video_metadata = {
        "fps": 10.0,
        "num_frames": 5,
        "width": 640,
        "height": 480,
    }

    # Mock TempFileManager.create_temp_video_from_frames
    with patch.object(
        app.sequence_tracker.temp_manager,
        "create_temp_video_from_frames",
        return_value="/tmp/test_video.mp4",
    ):
        # Call the images upload handler
        result = app._on_images_upload(test_images)

        # Verify that create_temp_video_from_frames was called
        app.sequence_tracker.temp_manager.create_temp_video_from_frames.assert_called_once()

        # Verify that set_video was called with the temp video path
        app.sequence_tracker.set_video.assert_called_once_with("/tmp/test_video.mp4")

        # Verify that metadata and slider update are returned
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # (metadata_text, slider_update)

    # Clean up
    import os

    for img_path in test_images:
        os.remove(img_path)


def test_sequence_extract_frame():
    """Test Extract Frame button handler."""
    app = Sam3GradioApp()

    # Mock sequence_tracker methods
    test_frame = PILImage.new("RGB", (640, 480), color="blue")
    app.sequence_tracker.set_prompt_frame = Mock(return_value=test_frame)

    # Call the extract frame handler
    frame_index = 5
    result = app._on_extract_frame(frame_index)

    # Verify that set_prompt_frame was called with correct index
    app.sequence_tracker.set_prompt_frame.assert_called_once_with(frame_index)

    # Verify that the extracted frame is returned
    assert result is not None
    assert isinstance(result, PILImage.Image)


def test_sequence_prompt_frame_point_click():
    """Test prompt frame image click handler for Point mode."""
    app = Sam3GradioApp()

    # Create mock SelectData for point click
    from gradio import SelectData

    mock_select_data = Mock(spec=SelectData)
    mock_select_data.index = (100, 200)  # (x, y) coordinates

    # Mock sequence_tracker methods
    test_overlay = PILImage.new("RGB", (640, 480), color="green")
    app.sequence_tracker.add_point = Mock()
    app.sequence_tracker.prompt_state = Mock()
    app.sequence_tracker.prompt_state.prompt_frame = PILImage.new("RGB", (640, 480))
    app.sequence_tracker.prompt_state.get_frame_with_overlay = Mock(
        return_value=test_overlay
    )

    # Call the prompt frame point click handler with include type
    result = app._on_prompt_frame_point_click(mock_select_data, "include")

    # Verify that add_point was called with correct coordinates and label
    app.sequence_tracker.add_point.assert_called_once_with(x=100, y=200, label=1)

    # Verify that overlay image is returned
    assert result is not None
    assert isinstance(result, PILImage.Image)


def test_sequence_prompt_frame_bbox_change():
    """Test prompt frame BBox annotation change handler."""
    app = Sam3GradioApp()

    # Create mock bbox data
    bbox_data = {
        "image": "/tmp/test_frame.jpg",
        "boxes": [
            (50, 100, 150, 300, "object1"),  # (left, top, right, bottom, label)
        ],
    }

    # Mock sequence_tracker methods
    test_overlay = PILImage.new("RGB", (640, 480), color="yellow")
    app.sequence_tracker.add_box = Mock()
    app.sequence_tracker.prompt_state = Mock()
    app.sequence_tracker.prompt_state.prompt_frame = PILImage.new("RGB", (640, 480))
    app.sequence_tracker.prompt_state.get_frame_with_overlay = Mock(
        return_value=test_overlay
    )

    # Call the bbox change handler
    result = app._on_prompt_frame_bbox_change(bbox_data)

    # Verify that add_box was called with converted coordinates (x, y, w, h)
    app.sequence_tracker.add_box.assert_called_once_with(x=50, y=100, w=100, h=200)

    # Verify that overlay image is returned
    assert result is not None
    assert isinstance(result, PILImage.Image)


def test_sequence_preview_tracking():
    """Test Preview button handler."""
    import tempfile

    app = Sam3GradioApp()

    # Mock sequence_tracker methods
    temp_preview_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    app.sequence_tracker.preview_tracking = Mock(return_value=temp_preview_video)

    # Call the preview handler
    result = app._on_preview_tracking()

    # Verify that preview_tracking was called with default num_frames=10
    app.sequence_tracker.preview_tracking.assert_called_once_with(num_frames=10)

    # Verify that video path is returned
    assert result is not None
    assert result == temp_preview_video

    # Clean up
    import os

    os.remove(temp_preview_video)


def test_sequence_track_video():
    """Test Track button handler with progress bar."""
    import tempfile

    app = Sam3GradioApp()

    # Mock generator for run_tracking_stream
    def mock_tracking_generator():
        # Yield progress updates
        yield {"type": "progress", "progress": 0.3, "desc": "Processing frame 30/100"}
        yield {"type": "progress", "progress": 0.6, "desc": "Processing frame 60/100"}
        # Yield final result
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        yield {"type": "result", "video_path": temp_video}

    app.sequence_tracker.run_tracking_stream = Mock(
        return_value=mock_tracking_generator()
    )

    # Call the track handler (without actual gr.Progress since it's mocked in tests)
    result = app._on_track_video()

    # Verify that run_tracking_stream was called
    app.sequence_tracker.run_tracking_stream.assert_called_once()

    # Verify that final video path is returned
    assert result is not None

    # Clean up if result is a file path
    import os

    if isinstance(result, str) and os.path.exists(result):
        os.remove(result)


def test_error_handling():
    """Test error handling in event handlers."""
    import gradio as gr

    app = Sam3GradioApp()

    # Test 1: Run segmentation without setting image first
    with pytest.raises(gr.Error):
        app._on_run_segmentation("test")

    # Test 2: Extract frame without setting video first
    with pytest.raises(gr.Error):
        app._on_extract_frame(frame_index=0)

    # Test 3: Preview tracking without setting video first
    with pytest.raises(gr.Error):
        app._on_preview_tracking()

    # Test 4: Track video without setting video first
    with pytest.raises(gr.Error):
        app._on_track_video()


def test_launch():
    """Test app launch method."""
    import gradio as gr
    import socket

    app = Sam3GradioApp()

    # Find an available port
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    free_port = find_free_port()

    # Verify that launch method exists and returns Blocks instance
    blocks = app.launch(share=False, server_port=free_port, prevent_thread_lock=True)

    # Verify that the returned object is a Gradio Blocks instance
    assert blocks is not None
    assert isinstance(blocks, gr.Blocks)

    # Close the server
    blocks.close()
