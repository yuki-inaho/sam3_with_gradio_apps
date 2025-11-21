import itertools
from pathlib import Path

import pytest
import torch

from sam3.model.sam3_video_predictor import Sam3VideoPredictor


@pytest.mark.slow
def test_bedroom_video_single_step():
    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "models" / "sam3.pt"
    video_path = project_root / "assets" / "videos" / "bedroom.mp4"

    if not ckpt_path.exists():
        pytest.skip("models/sam3.pt が無いためスキップ")
    if not video_path.exists():
        pytest.skip("assets/videos/bedroom.mp4 が無いためスキップ")

    # CUDA利用不可の場合はスキップ（CPU版はメモリ・時間コストが高いためここでは実行しない）
    if not torch.cuda.is_available():
        pytest.skip("CUDA環境が利用できないためスキップ")

    predictor = Sam3VideoPredictor(checkpoint_path=str(ckpt_path))

    session_info = predictor.start_session(str(video_path))
    session_id = session_info["session_id"]

    add_resp = predictor.add_prompt(
        session_id=session_id,
        frame_idx=0,
        text="person",
        points=None,
        point_labels=None,
        bounding_boxes=None,
        bounding_box_labels=None,
        obj_id=None,
    )
    assert add_resp["frame_index"] == 0

    gen = predictor.propagate_in_video(
        session_id=session_id,
        propagation_direction="forward",
        start_frame_idx=0,
        max_frame_num_to_track=5,
    )
    outputs = list(itertools.islice(gen, 5))
    assert outputs, "propagate_in_videoが出力を返しませんでした"
    out0 = outputs[0]["outputs"]
    assert "out_binary_masks" in out0
    masks = out0["out_binary_masks"]
    assert masks.shape[0] >= 0

    predictor.close_session(session_id)
    predictor.shutdown()
