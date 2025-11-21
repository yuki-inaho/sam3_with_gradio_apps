"""End-to-End tests for SAM3 Gradio Application with real model.

These tests use the actual SAM3 model and are marked as slow.
Run with: pytest -m slow
"""


import pytest
from PIL import Image

from scripts.gradio_app.sam3_gradio_app import Sam3GradioApp


def create_test_image(width: int = 256, height: int = 256) -> Image.Image:
    """Create a test image for E2E testing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        PIL Image instance.
    """
    return Image.new("RGB", (width, height), color="blue")


@pytest.mark.slow
def test_e2e_image_predictor_with_real_model():
    """Test Image Predictor with real SAM3 model (slow).

    This test requires models/sam3.pt to be available.
    """
    app = Sam3GradioApp()

    # Step 1: Upload image
    test_image = create_test_image(512, 512)
    result = app._on_image_upload(test_image)

    # Verify image is set
    assert result is not None
    assert app.image_predictor.prompt_state.image is not None

    # Step 2: Add point prompt
    app.image_predictor.add_point(x=256, y=256, label=True)
    assert len(app.image_predictor.prompt_state.points) == 1

    # Step 3: Run segmentation with real model
    # Note: This will actually load and run the SAM3 model
    try:
        seg_result = app._on_run_segmentation()

        # Verify result contains gallery images
        assert seg_result is not None
        assert isinstance(seg_result, list)
        assert len(seg_result) > 0

        print(f"E2E Image Predictor test passed: {len(seg_result)} results returned")

    except FileNotFoundError as e:
        pytest.skip(f"Model file not found: {e}")
    except Exception as e:
        pytest.fail(f"E2E test failed with real model: {e}")


@pytest.mark.slow
def test_e2e_app_launch():
    """Test that the app can be launched successfully (slow)."""
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

    # Launch the app
    try:
        blocks = app.launch(
            share=False, server_port=free_port, prevent_thread_lock=True
        )

        # Verify that the app launched successfully
        assert blocks is not None

        # Close the server
        blocks.close()

        print(f"E2E app launch test passed on port {free_port}")

    except Exception as e:
        pytest.fail(f"E2E app launch failed: {e}")
