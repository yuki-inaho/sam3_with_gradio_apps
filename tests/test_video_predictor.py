import pytest
import torch
from PIL import Image

from sam3.model.sam3_video_predictor import Sam3VideoPredictor


class DummyVideoModel:
    def __init__(self):
        self.reset_called = False

    def cuda(self):
        return self

    def eval(self):
        return self

    def init_state(self, resource_path, async_loading_frames=False, video_loader_type="cv2"):
        return {"resource_path": resource_path, "num_frames": 4}

    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        obj_id=None,
    ):
        outputs = {
            "frame": frame_idx,
            "text": text_str,
            "points": points,
            "boxes": boxes_xywh,
        }
        return frame_idx, outputs

    def remove_object(self, inference_state, obj_id, is_user_action=True):
        returns = inference_state
        returns["last_removed"] = obj_id
        return returns

    def reset_state(self, inference_state):
        self.reset_called = True

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        steps = 2 if max_frame_num_to_track is None else max_frame_num_to_track
        direction = -1 if reverse else 1
        for idx in range(steps):
            frame = start_frame_idx + idx * direction
            yield frame, {"mask": frame}


def test_video_predictor_session_lifecycle(monkeypatch, tmp_path):
    dummy_model = DummyVideoModel()

    def _build_model(*args, **kwargs):
        return dummy_model

    monkeypatch.setattr(
        "sam3.model_builder.build_sam3_video_model",
        _build_model,
    )

    predictor = Sam3VideoPredictor(checkpoint_path=None)

    resource = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(resource)

    session_info = predictor.start_session(str(resource))
    session_id = session_info["session_id"]

    add_response = predictor.add_prompt(
        session_id=session_id,
        frame_idx=0,
        text="object",
        points=[[0.1, 0.2]],
        point_labels=[1],
        bounding_boxes=[[0.2, 0.2, 0.3, 0.3]],
        bounding_box_labels=[1],
        obj_id=1,
    )
    assert add_response["frame_index"] == 0

    forward = list(
        predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="forward",
            start_frame_idx=0,
            max_frame_num_to_track=2,
        )
    )
    assert len(forward) == 2

    predictor.remove_object(session_id=session_id, obj_id=1)
    predictor.reset_session(session_id)
    assert dummy_model.reset_called is True

    predictor.close_session(session_id)
    predictor.shutdown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="CUDA 環境が利用できないためスキップ",
)
def test_video_predictor_cuda118(monkeypatch, tmp_path):
    cuda_version = torch.version.cuda or ""
    if cuda_version and not cuda_version.startswith("11.8"):
        pytest.skip("CUDA 11.8 以外の環境のためスキップ")

    class CudaVideoModel(DummyVideoModel):
        def cuda(self):
            self.cuda_called = True
            return self

    dummy_model = CudaVideoModel()

    def _build_model(*args, **kwargs):
        return dummy_model

    monkeypatch.setattr(
        "sam3.model_builder.build_sam3_video_model",
        _build_model,
    )

    predictor = Sam3VideoPredictor(checkpoint_path=None)

    resource = tmp_path / "frame_cuda.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(resource)
    session_id = predictor.start_session(str(resource))["session_id"]

    list(
        predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="forward",
            start_frame_idx=0,
            max_frame_num_to_track=1,
        )
    )

    predictor.close_session(session_id)
    predictor.shutdown()

    assert getattr(dummy_model, "cuda_called", False)
