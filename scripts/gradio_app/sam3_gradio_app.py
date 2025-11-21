"""SAM3 Gradio Application with Image Predictor and Sequence Tracker tabs."""

import logging
from typing import Dict, Any, Optional

import gradio as gr
from gradio_bbox_annotator import BBoxAnnotator
from PIL import Image

from scripts.gradio_app.image_predictor_app import ImagePredictorApp
from scripts.gradio_app.model_manager import Sam3ModelManager
from scripts.gradio_app.sequence_tracker_app import SequenceTrackerApp
from scripts.gradio_app.temp_file_manager import TempFileManager

logger = logging.getLogger(__name__)


class Sam3GradioApp:
    """Main SAM3 Gradio application with two tabs.

    This class integrates ImagePredictorApp (image segmentation) and
    SequenceTrackerApp (video tracking) into a unified Gradio interface.
    """

    def __init__(
        self,
        model_manager: Optional[Sam3ModelManager] = None,
        temp_manager: Optional[TempFileManager] = None,
    ):
        """Initialize the SAM3 Gradio application.

        Args:
            model_manager: Model manager instance (optional, creates new if None).
            temp_manager: Temp file manager instance (optional, creates new if None).
        """
        # Create shared managers
        self.model_manager = model_manager or Sam3ModelManager()
        self.temp_manager = temp_manager or TempFileManager()

        # Initialize sub-applications
        self.image_predictor = ImagePredictorApp(
            model_manager=self.model_manager,
            temp_manager=self.temp_manager,
        )
        self.sequence_tracker = SequenceTrackerApp(
            model_manager=self.model_manager,
            temp_manager=self.temp_manager,
        )

        logger.info("Sam3GradioApp initialized with shared managers")

    def build_image_tab(self) -> Dict[str, Any]:
        """Build the Image Predictor tab UI.

        Returns:
            Dictionary of UI components for the Image Predictor tab.
        """
        with gr.Column():
            gr.Markdown("## Image Segmentation")
            gr.Markdown(
                "Upload an image and provide prompts (text, points, or boxes) "
                "to segment objects."
            )

            with gr.Row():
                # Left column: Input
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )

                    image_example = gr.Dropdown(
                        label="Load Example",
                        choices=["Groceries", "Test Image", "Truck"],
                        value=None,
                        multiselect=False,
                    )

                    bbox_annotator = BBoxAnnotator(
                        label="Draw Bounding Boxes",
                        visible=False,
                    )

                    text_prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder="e.g., 'person', 'cat', 'car'",
                        lines=2,
                    )

                    prompt_mode = gr.Radio(
                        label="Prompt Mode",
                        choices=[
                            "TEXT_ONLY",       # Text-only prompting (no points/boxes)
                            "POINTS_ONLY",
                            "BOXES_ONLY",
                            "TEXT_AND_POINTS",
                            "TEXT_AND_BOXES",
                        ],
                        value="TEXT_ONLY"  # Default to simplest mode,
                    )

                    point_type = gr.Radio(
                        label="Point Type (for Point mode)",
                        choices=["include", "exclude"],
                        value="include",
                    )

                    with gr.Row():
                        run_button = gr.Button("Run Segmentation", variant="primary")
                        clear_button = gr.Button("Clear Prompts")
                        reset_button = gr.Button("Reset Session")

                # Right column: Output
                with gr.Column(scale=1):
                    output_gallery = gr.Gallery(
                        label="Segmentation Results",
                        columns=2,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                    )

                    download_button = gr.File(
                        label="Download Masks (ZIP)",
                        visible=False,
                    )

        components = {
            "input_image": input_image,
            "image_example": image_example,
            "bbox_annotator": bbox_annotator,
            "text_prompt": text_prompt,
            "prompt_mode": prompt_mode,
            "point_type": point_type,
            "run_button": run_button,
            "clear_button": clear_button,
            "reset_button": reset_button,
            "output_gallery": output_gallery,
            "download_button": download_button,
        }

        logger.info("Image Predictor tab UI built")
        return components

    def build_sequence_tab(self) -> Dict[str, Any]:
        """Build the Sequence Tracker tab UI.

        Returns:
            Dictionary of UI components for the Sequence Tracker tab.
        """
        with gr.Column():
            gr.Markdown("## Video Tracking")
            gr.Markdown(
                "Upload a video or image sequence, select a frame, add prompts, "
                "and track objects across frames."
            )

            with gr.Row():
                # Left column: Input
                with gr.Column(scale=1):
                    input_type = gr.Radio(
                        label="Input Type",
                        choices=["Video File", "Image Sequence"],
                        value="Video File",
                    )

                    video_example = gr.Dropdown(
                        label="Load Video Example",
                        choices=["Bedroom MP4", "Frames 0001"],
                        value=None,
                        multiselect=False,
                    )

                    input_video = gr.Video(
                        label="Input Video",
                        sources=["upload"],
                        visible=True,
                    )

                    input_images = gr.File(
                        label="Input Images (Sequential)",
                        file_count="multiple",
                        file_types=["image"],
                        visible=False,
                    )

                    video_metadata = gr.Textbox(
                        label="Video Metadata",
                        placeholder="Upload a video or images to see metadata",
                        lines=2,
                        interactive=False,
                    )

                    gr.Markdown("### Frame Selection")
                    frame_slider = gr.Slider(
                        label="Frame Index",
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=0,
                        interactive=False,
                    )

                    max_frames = gr.Slider(
                        label="Max Frames (tracking cap)",
                        minimum=1,
                        maximum=500,
                        step=1,
                        value=120,
                        interactive=True,
                        info="Limits number of frames processed to avoid OOM",
                    )

                    extract_frame_button = gr.Button("Extract Frame")

                    prompt_frame_image = gr.Image(
                        label="Prompt Frame",
                        type="pil",
                        interactive=True,
                    )

                    # BBox Annotator for Box mode (initially hidden)
                    prompt_frame_bbox = BBoxAnnotator(
                        label="Prompt Frame (Box Mode)",
                        visible=False,
                    )

                    text_prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder="e.g., 'person', 'cat', 'car'",
                        lines=2,
                    )

                    prompt_mode = gr.Radio(
                        label="Prompt Mode",
                        choices=[
                            # SequenceではTEXT_ONLYはサポートしない
                            "POINTS_ONLY",
                            "BOXES_ONLY",
                            "TEXT_AND_POINTS",
                            "TEXT_AND_BOXES",
                        ],
                        value="POINTS_ONLY",
                    )

                    point_type = gr.Radio(
                        label="Point Type (for Point mode)",
                        choices=["include", "exclude"],
                        value="include",
                    )

                    with gr.Row():
                        preview_button = gr.Button("Preview (10 frames)")
                        track_button = gr.Button("Track Video", variant="primary")

                    with gr.Row():
                        clear_button = gr.Button("Clear Prompts")
                        reset_button = gr.Button("Reset Session")

                # Right column: Output
                with gr.Column(scale=1):
                    output_video = gr.Video(
                        label="Tracking Results",
                        autoplay=False,
                    )

                    download_masks_button = gr.File(
                        label="Download Masks (ZIP)",
                        visible=False,
                    )

                    download_tracks_button = gr.File(
                        label="Download Tracks (JSON)",
                        visible=False,
                    )

        components = {
            "input_type": input_type,
            "video_example": video_example,
            "input_video": input_video,
            "input_images": input_images,
            "video_metadata": video_metadata,
            "frame_slider": frame_slider,
            "max_frames": max_frames,
            "extract_frame_button": extract_frame_button,
            "prompt_frame_image": prompt_frame_image,
            "prompt_frame_bbox": prompt_frame_bbox,
            "text_prompt": text_prompt,
            "prompt_mode": prompt_mode,
            "point_type": point_type,
            "preview_button": preview_button,
            "track_button": track_button,
            "clear_button": clear_button,
            "reset_button": reset_button,
            "output_video": output_video,
            "download_masks_button": download_masks_button,
            "download_tracks_button": download_tracks_button,
        }

        logger.info("Sequence Tracker tab UI built")
        return components

    def build_blocks(self) -> gr.Blocks:
        """Build the complete Gradio Blocks UI with 2 tabs.

        Returns:
            gr.Blocks instance with Image Predictor and Sequence Tracker tabs.
        """
        with gr.Blocks(title="SAM3 - Segment Anything Model 3") as blocks:
            gr.Markdown("# SAM3: Segment Anything Model 3")
            gr.Markdown(
                "Interactive demo for image segmentation and video object tracking."
            )

            with gr.Tabs():
                with gr.Tab("Image Predictor"):
                    self.image_components = self.build_image_tab()
                    self._bind_image_predictor_events()

                with gr.Tab("Sequence Tracker"):
                    self.sequence_components = self.build_sequence_tab()
                    self._bind_sequence_tracker_events()

        logger.info("Complete Blocks UI built with 2 tabs")
        return blocks

    def _bind_image_predictor_events(self) -> None:
        """Bind event handlers for Image Predictor tab."""
        comp = self.image_components

        # Image upload event
        comp["input_image"].change(
            fn=self._on_image_upload,
            inputs=[comp["input_image"]],
            outputs=[comp["input_image"]],
        )

        # Example selection (load preset image + text)
        comp["image_example"].change(
            fn=self._on_image_example_select,
            inputs=[comp["image_example"]],
            outputs=[comp["input_image"], comp["text_prompt"], comp["output_gallery"]],
        )

        # Prompt mode change event (with visibility control)
        comp["prompt_mode"].change(
            fn=self._on_prompt_mode_change,
            inputs=[comp["prompt_mode"]],
            outputs=[comp["input_image"], comp["bbox_annotator"], comp["point_type"]],
            queue=False,  # Disable queueing for immediate processing
        )

        # Text prompt change event
        comp["text_prompt"].change(
            fn=self._on_text_prompt_change,
            inputs=[comp["text_prompt"]],
            outputs=[],
        )

        # Image click event for Point mode
        # NOTE: gr.Image select event in Gradio 5.x may not provide click coordinates
        # This binding is experimental and may need alternative solution
        # SelectData is passed automatically as first argument, then inputs follow
        try:
            comp["input_image"].select(
                fn=self._on_image_click,
                inputs=[comp["point_type"]],
                outputs=[comp["input_image"]],
            )
            logger.info("Image click event bound (experimental)")
        except AttributeError:
            logger.warning(
                "gr.Image does not support select event - Point mode click not available"
            )

        # BBox annotation change event for Box mode
        comp["bbox_annotator"].change(
            fn=self._on_bbox_change,
            inputs=[comp["bbox_annotator"]],
            outputs=[comp["bbox_annotator"]],
        )

        # Run button click event
        comp["run_button"].click(
            fn=self._on_run_segmentation,
            inputs=[comp["text_prompt"]],
            outputs=[comp["output_gallery"]],
        )

        # Clear button click event (prompts only)
        comp["clear_button"].click(
            fn=self._on_clear_prompts,
            inputs=[],
            outputs=[comp["text_prompt"], comp["output_gallery"]],
        )

        # Reset button click event (full reset)
        comp["reset_button"].click(
            fn=self._on_reset_session,
            inputs=[],
            outputs=[comp["input_image"], comp["text_prompt"], comp["output_gallery"]],
        )

        logger.info("Image Predictor events bound")

    def _bind_sequence_tracker_events(self) -> None:
        """Bind event handlers for Sequence Tracker tab."""
        comp = self.sequence_components

        # Input type change event (visibility switching)
        comp["input_type"].change(
            fn=self._on_input_type_change,
            inputs=[comp["input_type"]],
            outputs=[comp["input_video"], comp["input_images"]],
        )

        # Video example selection
        comp["video_example"].change(
            fn=self._on_sequence_example_select,
            inputs=[comp["video_example"]],
            outputs=[
                comp["input_type"],
                comp["input_video"],
                comp["video_metadata"],
                comp["frame_slider"],
            ],
        )

        # Video upload event
        comp["input_video"].upload(
            fn=self._on_video_upload,
            inputs=[comp["input_video"]],
            outputs=[comp["video_metadata"], comp["frame_slider"]],
        )

        # Image sequence upload event
        comp["input_images"].upload(
            fn=self._on_images_upload,
            inputs=[comp["input_images"]],
            outputs=[comp["video_metadata"], comp["frame_slider"]],
        )

        # Extract Frame button click event
        comp["extract_frame_button"].click(
            fn=self._on_extract_frame,
            inputs=[comp["frame_slider"]],
            outputs=[comp["prompt_frame_image"]],
        )

        # Prompt frame image click event for Point mode
        try:
            comp["prompt_frame_image"].select(
                fn=self._on_prompt_frame_point_click,
                inputs=[comp["point_type"]],
                outputs=[comp["prompt_frame_image"]],
            )
            logger.info("Prompt frame click event bound (experimental)")
        except AttributeError:
            logger.warning(
                "gr.Image does not support select event - Point mode click not available for prompt frame"
            )

        # Prompt frame BBox annotation change event for Box mode
        comp["prompt_frame_bbox"].change(
            fn=self._on_prompt_frame_bbox_change,
            inputs=[comp["prompt_frame_bbox"]],
            outputs=[comp["prompt_frame_bbox"]],
        )

        # Preview button click event
        comp["preview_button"].click(
            fn=self._on_preview_tracking,
            inputs=[comp["max_frames"]],
            outputs=[comp["output_video"]],
        )

        # Track button click event (with queue and progress)
        comp["track_button"].click(
            fn=self._on_track_video,
            inputs=[comp["max_frames"]],
            outputs=[
                comp["output_video"],
                comp["download_masks_button"],
                comp["download_tracks_button"],
            ],
            queue=True,  # 重要: 長時間処理のためqueue=True
        )

        logger.info(
            "Sequence Tracker events bound (手順4-6-1, 4-6-2, 4-6-3, 4-6-4, 4-6-5)"
        )

    # Image Predictor Event Handlers

    def _on_image_upload(self, image: Image.Image) -> Optional[Image.Image]:
        """Handle image upload event.

        Args:
            image: Uploaded PIL Image.

        Returns:
            The uploaded image (for display).
        """
        if image is None:
            return None

        try:
            self.image_predictor.set_image(image)
            logger.info(f"Image uploaded: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to set image: {e}")
            raise gr.Error(f"Failed to load image: {str(e)}")

    def _on_image_example_select(
        self, example_name: Optional[str]
    ) -> tuple:
        """Load preset example image and text."""
        if not example_name:
            return (gr.update(), gr.update(), gr.update())

        example_map = {
            "Groceries": ("assets/images/groceries.jpg", "groceries"),
            "Test Image": ("assets/images/test_image.jpg", "person"),
            "Truck": ("assets/images/truck.jpg", "truck"),
        }
        path, text = example_map.get(example_name, (None, ""))
        if path is None:
            return (gr.update(), gr.update(), gr.update())

        try:
            image = Image.open(path)
            # Clear gallery when loading a preset
            return (gr.update(value=image), gr.update(value=text), gr.update(value=[]))
        except Exception as e:
            logger.error(f"Failed to load example {example_name}: {e}")
            raise gr.Error(f"Failed to load example: {str(e)}")

    def _on_sequence_example_select(
        self, example_name: Optional[str]
    ) -> tuple:
        """Load preset video or image-sequence example.

        Returns updates for (input_type, input_video, video_metadata, frame_slider).
        """
        if not example_name:
            return (gr.update(), gr.update(), gr.update(), gr.update())

        try:
            # Map example name to source and type
            example_map = {
                "Bedroom MP4": {
                    "type": "Video File",
                    "path": "assets/videos/bedroom.mp4",
                },
                "Frames 0001": {
                    "type": "Image Sequence",
                    "path": "assets/videos/0001",
                },
            }

            example = example_map.get(example_name)
            if example is None:
                return (gr.update(), gr.update(), gr.update(), gr.update())

            input_type_value = example["type"]

            # When example is a video file, load directly
            if input_type_value == "Video File":
                video_path = example["path"]
                self.sequence_tracker.set_video(video_path)
            else:
                # Image sequence: load frames, convert to a temp video, then set
                from pathlib import Path
                from PIL import Image as PILImage

                frames_dir = Path(example["path"])
                if not frames_dir.exists():
                    raise ValueError(f"Example frames not found: {frames_dir}")

                # Sort numerically by filename
                frames = []
                for img_path in sorted(
                    frames_dir.glob("*"), key=lambda p: p.stem.zfill(8)
                ):
                    if not img_path.is_file():
                        continue
                    try:
                        frames.append(PILImage.open(img_path))
                    except Exception:
                        continue

                if not frames:
                    raise ValueError(f"No frames found in {frames_dir}")

                temp_video_path = (
                    self.sequence_tracker.temp_manager.create_temp_video_from_frames(
                        frames=frames,
                        fps=10.0,
                    )
                )
                video_path = temp_video_path
                # Switch input type to Video File since we now have a temp video path
                input_type_value = "Video File"
                self.sequence_tracker.set_video(video_path)

            metadata = self.sequence_tracker.video_metadata or {}
            fps = metadata.get("fps", 0)
            num_frames = metadata.get("num_frames", 0)
            width = metadata.get("width", 0)
            height = metadata.get("height", 0)

            metadata_text = (
                f"FPS: {fps:.2f}, Frames: {num_frames}, Resolution: {width}x{height}"
            )

            slider_update = gr.update(
                maximum=max(0, num_frames - 1),
                value=0,
                interactive=True,
            )

            return (
                gr.update(value=input_type_value),
                gr.update(value=video_path),
                gr.update(value=metadata_text),
                slider_update,
            )

        except Exception as e:
            logger.error(f"Failed to load example {example_name}: {e}")
            raise gr.Error(f"Failed to load example: {str(e)}")

    def _on_prompt_mode_change(self, mode: str) -> tuple:
        """Handle prompt mode change event and update component visibility.

        Args:
            mode: Selected prompt mode string.

        Returns:
            Tuple of gr.update() for (input_image, bbox_annotator, point_type) visibility.
        """
        try:
            from scripts.gradio_app.prompt_types import PromptMode

            prompt_mode = PromptMode[mode]
            self.image_predictor.set_prompt_mode(prompt_mode)
            logger.info(f"Prompt mode changed to: {mode}")

            # Determine which component to show based on mode
            # Box modes: show bbox_annotator, hide input_image
            # Other modes: show input_image, hide bbox_annotator
            is_text_only = mode == "TEXT_ONLY"  # Hide point_type for text-only mode
            is_box_mode = mode in ["BOXES_ONLY", "TEXT_AND_BOXES"]

            return (
                gr.update(visible=not is_box_mode),  # input_image
                gr.update(visible=is_box_mode),  # bbox_annotator
                gr.update(visible=not is_text_only),  # point_type
            )

        except Exception as e:
            logger.error(f"Failed to change prompt mode: {e}")
            raise gr.Error(f"Failed to change prompt mode: {str(e)}")

    def _on_text_prompt_change(self, text: str) -> None:
        """Handle text prompt change event.

        Args:
            text: Text prompt entered by user.
        """
        try:
            if text:
                self.image_predictor.prompt_state.text_prompt = text.strip()
                logger.info(f"Text prompt set: {text[:50]}...")
            else:
                self.image_predictor.prompt_state.text_prompt = ""
                logger.info("Text prompt cleared")
        except Exception as e:
            logger.error(f"Failed to set text prompt: {e}")
            raise gr.Error(f"Failed to set text prompt: {str(e)}")

    def _on_image_click(
        self, evt: gr.SelectData, point_type: str
    ) -> Optional[Image.Image]:
        """Handle image click event for Point mode.

        Args:
            evt: Gradio SelectData object containing click coordinates.
                 Expected format: evt.index = (x, y)
            point_type: Point type selection ("include" or "exclude").
                       "include" -> positive label (1), "exclude" -> negative label (0)

        Returns:
            Image with overlay showing added points, or None if error.
        """
        if evt is None or not hasattr(evt, "index"):
            logger.warning("Invalid SelectData received")
            return None

        try:
            # Get click coordinates (x, y)
            x, y = evt.index

            # Determine label based on point_type
            is_positive = point_type == "include"

            # Add point to image predictor
            self.image_predictor.add_point(x=x, y=y, label=is_positive)
            logger.info(
                f"Point added at ({x}, {y}), positive={is_positive} (type={point_type})"
            )

            # Get image with overlay
            overlay_image = self.image_predictor.get_image_with_overlay()
            return overlay_image

        except Exception as e:
            logger.error(f"Failed to handle image click: {e}")
            raise gr.Error(f"Failed to add point: {str(e)}")

    def _on_bbox_change(
        self, bbox_data: Optional[Dict[str, Any]]
    ) -> Optional[Image.Image]:
        """Handle BBox annotation change event from gradio-bbox-annotator.

        Args:
            bbox_data: BBox annotation data from BBoxAnnotator.
                      Format: {"image": image_path, "boxes": [(left, top, right, bottom, label), ...]}

        Returns:
            Image with overlay showing added boxes, or None if error.
        """
        if bbox_data is None or "boxes" not in bbox_data:
            logger.warning("Invalid bbox_data received")
            return None

        try:
            boxes = bbox_data.get("boxes", [])

            # Process each box
            for box in boxes:
                if len(box) == 5:
                    left, top, right, bottom, label = box

                    # Convert from (left, top, right, bottom) to (x, y, w, h)
                    x = left
                    y = top
                    w = right - left
                    h = bottom - top

                    # Add box to image predictor
                    self.image_predictor.add_box(x=x, y=y, w=w, h=h)
                    logger.info(f"Box added: ({x}, {y}, {w}, {h}), label={label}")
                else:
                    logger.warning(f"Invalid box format: {box}")

            # Get image with overlay
            overlay_image = self.image_predictor.get_image_with_overlay()
            return overlay_image

        except Exception as e:
            logger.error(f"Failed to handle bbox annotation: {e}")
            raise gr.Error(f"Failed to add box: {str(e)}")

    def _on_run_segmentation(self, text_prompt: str) -> list[Image.Image]:
        """Handle Run button click event for image segmentation.

        Args:
            text_prompt: Text prompt from the textbox (may be empty).

        Returns:
            List of images for Gallery display (overlay and masks).

        Raises:
            gr.Error: If segmentation fails or no image is set.
        """
        try:
            # Debug: Print directly to stderr for debugging (bypasses logger)
            import sys

            print(
                f"DEBUG: _on_run_segmentation called with text_prompt='{text_prompt}' (type={type(text_prompt).__name__}, len={len(text_prompt) if text_prompt else 0})",
                file=sys.stderr,
                flush=True,
            )

            # Debug: Log the received text_prompt value
            logger.info(
                f"_on_run_segmentation called with text_prompt='{text_prompt}' (type={type(text_prompt).__name__}, len={len(text_prompt) if text_prompt else 0})"
            )

            # Set text prompt if provided
            if text_prompt:
                self.image_predictor.prompt_state.text_prompt = text_prompt.strip()
                print(
                    f"DEBUG: Text prompt set to: '{text_prompt}'",
                    file=sys.stderr,
                    flush=True,
                )
                logger.info(f"Text prompt set from Run button: {text_prompt[:50]}...")
            else:
                print(f"DEBUG: Text prompt is empty!", file=sys.stderr, flush=True)
                logger.warning(
                    "Text prompt is empty - no text will be used for segmentation"
                )

            # Run segmentation
            result = self.image_predictor.run_segmentation()

            # Build gallery output: [overlay_image, mask1, mask2, ...]
            # overlay_image is guaranteed to be non-None by ensure_overlay() in run_segmentation()
            gallery_images = [result.overlay_image]

            # Convert masks to PIL Images for Gallery
            num_masks = result.masks.shape[0] if result.masks is not None else 0
            if num_masks > 0:
                mask_images = [Image.fromarray(m) for m in result.masks]
                gallery_images.extend(mask_images)

            logger.info(f"Segmentation completed: {num_masks} masks generated")
            return gallery_images

        except ValueError as e:
            # Image or prompt not set
            logger.error(f"Segmentation failed (validation): {e}")
            raise gr.Error(f"Segmentation failed: {str(e)}")
        except RuntimeError as e:
            # Model inference failed
            logger.error(f"Segmentation failed (inference): {e}")
            raise gr.Error(f"Segmentation failed: {str(e)}")
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error during segmentation: {e}")
            raise gr.Error(f"Unexpected error: {str(e)}")

    def _on_clear_prompts(self) -> tuple:
        """Handle Clear button click event to clear prompts only.

        Returns:
            Tuple of gr.update() for (text_prompt, output_gallery) components.
        """
        try:
            # Clear prompts in image predictor
            self.image_predictor.clear_prompts()
            logger.info("Prompts cleared (image retained)")

            # Return updates to clear UI components
            return (
                gr.update(value=""),  # text_prompt
                gr.update(value=[]),  # output_gallery
            )

        except Exception as e:
            logger.error(f"Failed to clear prompts: {e}")
            raise gr.Error(f"Failed to clear prompts: {str(e)}")

    def _on_reset_session(self) -> tuple:
        """Handle Reset button click event for full session reset.

        Returns:
            Tuple of gr.update() for (input_image, text_prompt, output_gallery) components.
        """
        try:
            # Reset session in image predictor
            self.image_predictor.reset_session()
            logger.info("Session reset (all data cleared)")

            # Return updates to clear all UI components
            return (
                gr.update(value=None),  # input_image
                gr.update(value=""),  # text_prompt
                gr.update(value=[]),  # output_gallery
            )

        except Exception as e:
            logger.error(f"Failed to reset session: {e}")
            raise gr.Error(f"Failed to reset session: {str(e)}")

    # --- Sequence Tracker Event Handlers ---

    def _on_input_type_change(self, input_type: str) -> tuple:
        """Handle input_type Radio change event for visibility switching.

        Args:
            input_type: Selected input type ("Video File" or "Image Sequence").

        Returns:
            Tuple of gr.update() for (input_video, input_images) components.
        """
        if input_type == "Video File":
            # Show video input, hide images input
            return (
                gr.update(visible=True),  # input_video
                gr.update(visible=False),  # input_images
            )
        else:  # "Image Sequence"
            # Hide video input, show images input
            return (
                gr.update(visible=False),  # input_video
                gr.update(visible=True),  # input_images
            )

    def _on_video_upload(self, video_path: str) -> tuple:
        """Handle video upload event with metadata extraction.

        Args:
            video_path: Path to uploaded video file.

        Returns:
            Tuple of (metadata_text, slider_update).

        Raises:
            gr.Error: If video upload or metadata extraction fails.
        """
        try:
            # Set video in sequence tracker
            self.sequence_tracker.set_video(video_path)
            logger.info(f"Video uploaded: {video_path}")

            # Get metadata
            metadata = self.sequence_tracker.video_metadata
            if metadata:
                fps = metadata.get("fps", 0)
                num_frames = metadata.get("num_frames", 0)
                width = metadata.get("width", 0)
                height = metadata.get("height", 0)

                metadata_text = f"FPS: {fps:.2f}, Frames: {num_frames}, Resolution: {width}x{height}"

                # Update frame slider
                slider_update = gr.update(
                    maximum=max(0, num_frames - 1),
                    value=0,
                    interactive=True,
                )

                return (gr.update(value=metadata_text), slider_update)
            else:
                raise ValueError("Failed to extract video metadata")

        except ValueError as e:
            logger.error(f"Video upload failed (validation): {e}")
            raise gr.Error(f"Video upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during video upload: {e}")
            raise gr.Error(f"Unexpected error: {str(e)}")

    def _on_images_upload(self, image_paths: list[str]) -> tuple:
        """Handle image sequence upload event with temp video conversion.

        Args:
            image_paths: List of paths to uploaded image files.

        Returns:
            Tuple of (metadata_text, slider_update).

        Raises:
            gr.Error: If image sequence upload or conversion fails.
        """
        try:
            if not image_paths:
                raise ValueError("No images uploaded")

            # Load images as PIL Images
            from PIL import Image as PILImage

            images = []
            for img_path in image_paths:
                try:
                    img = PILImage.open(img_path)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")

            if not images:
                raise ValueError("No valid images found")

            # Convert images to temporary video
            temp_video_path = (
                self.sequence_tracker.temp_manager.create_temp_video_from_frames(
                    frames=images,
                    fps=10.0,  # Default FPS for image sequences
                )
            )
            logger.info(f"Image sequence converted to temp video: {temp_video_path}")

            # Set video in sequence tracker
            self.sequence_tracker.set_video(temp_video_path)

            # Get metadata
            metadata = self.sequence_tracker.video_metadata
            if metadata:
                fps = metadata.get("fps", 0)
                num_frames = metadata.get("num_frames", 0)
                width = metadata.get("width", 0)
                height = metadata.get("height", 0)

                metadata_text = f"FPS: {fps:.2f}, Frames: {num_frames}, Resolution: {width}x{height}"

                # Update frame slider
                slider_update = gr.update(
                    maximum=max(0, num_frames - 1),
                    value=0,
                    interactive=True,
                )

                return (gr.update(value=metadata_text), slider_update)
            else:
                raise ValueError("Failed to extract video metadata")

        except ValueError as e:
            logger.error(f"Image sequence upload failed (validation): {e}")
            raise gr.Error(f"Image sequence upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during image sequence upload: {e}")
            raise gr.Error(f"Unexpected error: {str(e)}")

    def _on_extract_frame(self, frame_index: int) -> Optional[Image.Image]:
        """Handle Extract Frame button click event.

        Args:
            frame_index: Frame index to extract.

        Returns:
            Extracted frame as PIL Image.

        Raises:
            gr.Error: If frame extraction fails.
        """
        try:
            # Extract frame at specified index
            frame = self.sequence_tracker.set_prompt_frame(frame_index)
            logger.info(f"Frame extracted at index {frame_index}")
            return frame

        except ValueError as e:
            logger.error(f"Frame extraction failed (validation): {e}")
            raise gr.Error(f"Frame extraction failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during frame extraction: {e}")
            raise gr.Error(f"Unexpected error: {str(e)}")

    def _on_prompt_frame_point_click(
        self, evt: gr.SelectData, point_type: str
    ) -> Optional[Image.Image]:
        """Handle prompt frame image click event for Point mode.

        Args:
            evt: SelectData containing click coordinates.
            point_type: "include" or "exclude".

        Returns:
            Frame with overlay showing added point.

        Raises:
            gr.Error: If point addition fails.
        """
        try:
            x, y = evt.index
            # Convert point_type to label (1 for include, 0 for exclude)
            label = 1 if point_type == "include" else 0

            # Add point to sequence tracker
            self.sequence_tracker.add_point(x=x, y=y, label=label)
            logger.info(f"Point added to prompt frame: ({x}, {y}), label={label}")

            # Get frame with overlay
            overlay_image = self.sequence_tracker.prompt_state.get_frame_with_overlay()
            return overlay_image

        except Exception as e:
            logger.error(f"Failed to add point to prompt frame: {e}")
            raise gr.Error(f"Failed to add point: {str(e)}")

    def _on_prompt_frame_bbox_change(
        self, bbox_data: Optional[dict]
    ) -> Optional[Image.Image]:
        """Handle prompt frame BBox annotation change event for Box mode.

        Args:
            bbox_data: Dictionary containing image path and boxes list.

        Returns:
            Frame with overlay showing added boxes.

        Raises:
            gr.Error: If box addition fails.
        """
        if bbox_data is None or "boxes" not in bbox_data:
            logger.warning("Invalid bbox_data received for prompt frame")
            return None

        try:
            boxes = bbox_data.get("boxes", [])

            # Process each box
            for box in boxes:
                if len(box) == 5:
                    left, top, right, bottom, label = box

                    # Convert from (left, top, right, bottom) to (x, y, w, h)
                    x = left
                    y = top
                    w = right - left
                    h = bottom - top

                    # Add box to sequence tracker
                    self.sequence_tracker.add_box(x=x, y=y, w=w, h=h)
                    logger.info(
                        f"Box added to prompt frame: ({x}, {y}, {w}, {h}), label={label}"
                    )
                else:
                    logger.warning(f"Invalid box format: {box}")

            # Get frame with overlay
            overlay_image = self.sequence_tracker.prompt_state.get_frame_with_overlay()
            return overlay_image

        except Exception as e:
            logger.error(f"Failed to add box to prompt frame: {e}")
            raise gr.Error(f"Failed to add box: {str(e)}")

    def _on_preview_tracking(self, max_frames: Optional[int] = None) -> Optional[str]:
        """Handle Preview button click event for short tracking preview.

        Returns:
            Path to preview video file.

        Raises:
            gr.Error: If preview tracking fails.
        """
        try:
            # Run preview tracking (10 frames by default)
            video_path = self.sequence_tracker.preview_tracking(
                max_frames=max_frames or 10
            )
            logger.info(f"Preview tracking completed: {video_path}")
            return video_path

        except ValueError as e:
            logger.error(f"Preview tracking failed (validation): {e}")
            raise gr.Error(f"Preview tracking failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during preview tracking: {e}")
            raise gr.Error(f"Unexpected error: {str(e)}")

    def _on_track_video(
        self, max_frames: Optional[int] = None, progress=gr.Progress()
    ) -> tuple:
        """Handle Track button click event with progress bar for full video tracking.

        Args:
            progress: Gradio Progress object for progress bar display.

        Returns:
            Tuple of (video_path, download_masks_visibility, download_tracks_visibility).

        Raises:
            gr.Error: If video tracking fails.
        """
        try:
            # Run tracking with progress updates
            video_path = None
            for update in self.sequence_tracker.run_tracking_stream(
                max_frames=max_frames
            ):
                if update["type"] == "progress":
                    # Update progress bar
                    progress(
                        update.get("progress", 0),
                        desc=update.get("desc", "Processing"),
                    )
                elif update["type"] == "result":
                    # Get final result
                    video_path = update["video_path"]
                elif update["type"] == "error":
                    raise gr.Error(update.get("message", "Tracking failed"))

            if video_path is None:
                raise ValueError("Tracking failed: no result returned")

            logger.info(f"Video tracking completed: {video_path}")

            # Return video path and make download buttons visible
            return (
                video_path,  # output_video
                gr.update(visible=True),  # download_masks_button
                gr.update(visible=True),  # download_tracks_button
            )

        except ValueError as e:
            logger.error(f"Video tracking failed (validation): {e}")
            raise gr.Error(f"Video tracking failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during video tracking: {e}")
            raise gr.Error(f"Unexpected error: {str(e)}")

    # Public Methods

    def launch(
        self,
        share: bool = False,
        server_port: int = 7860,
        prevent_thread_lock: bool = False,
        **kwargs,
    ) -> gr.Blocks:
        """Launch the Gradio application.

        Args:
            share: Whether to create a public link (default: False).
            server_port: Port number for the server (default: 7860).
            prevent_thread_lock: Whether to prevent thread lock for testing (default: False).
            **kwargs: Additional arguments to pass to blocks.launch().

        Returns:
            The launched Gradio Blocks instance.

        Raises:
            RuntimeError: If launch fails.
        """
        try:
            blocks = self.build_blocks()
            blocks.launch(
                share=share,
                server_port=server_port,
                prevent_thread_lock=prevent_thread_lock,
                **kwargs,
            )
            logger.info(f"Application launched on port {server_port}")
            return blocks
        except Exception as e:
            logger.error(f"Failed to launch application: {e}")
            raise RuntimeError(f"Failed to launch application: {str(e)}")


if __name__ == "__main__":
    app = Sam3GradioApp()
    app.launch(share=False, server_port=7860)
