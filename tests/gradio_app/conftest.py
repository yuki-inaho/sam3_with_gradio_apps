"""Pytest configuration for gradio_app tests."""

import pytest

from scripts.gradio_app.model_manager import Sam3ModelManager


@pytest.fixture(autouse=True)
def reset_model_manager_singleton():
    """Reset Sam3ModelManager singleton before and after each test.

    This ensures test isolation despite the singleton pattern.
    Without this, mocking in one test can affect subsequent tests.

    We save and restore the original methods because tests may mock them.
    """
    manager = Sam3ModelManager()

    # Save original methods
    original_get_image_model = manager.__class__.get_image_model
    original_get_video_predictor = manager.__class__.get_video_predictor

    # Reset state
    manager.reset()

    yield

    # Remove instance-level mocks (if tests set them)
    # Tests may do: app.model_manager.get_image_model = Mock(...)
    # This creates an instance attribute that shadows the class method
    if "get_image_model" in manager.__dict__:
        del manager.__dict__["get_image_model"]
    if "get_video_predictor" in manager.__dict__:
        del manager.__dict__["get_video_predictor"]

    # Restore original class methods (in case they were patched at class level)
    manager.__class__.get_image_model = original_get_image_model
    manager.__class__.get_video_predictor = original_get_video_predictor

    # Reset state again
    manager.reset()
