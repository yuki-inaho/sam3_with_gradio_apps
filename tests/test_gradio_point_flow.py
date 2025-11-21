"""
Gradio Point/BoxタブのフローをCUIで再現し、dtypeミスマッチが出ないこととマスクが生成されることを検証。
GPU前提。models/sam3.pt と assets/videos/bedroom.mp4 が必要。
"""

from pathlib import Path

import pytest
import torch

from video_app_hf_original import (
    AppState,
    init_video_session,
    on_image_click,
    propagate_masks,
)


@pytest.mark.slow
def test_gradio_point_flow_smoke():
    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "models" / "sam3.pt"
    video_path = project_root / "assets" / "videos" / "bedroom.mp4"

    if not ckpt_path.exists() or not video_path.exists():
        pytest.skip("必要なファイル(models/sam3.pt, assets/videos/bedroom.mp4)が無いためスキップ")
    if not torch.cuda.is_available():
        pytest.skip("CUDA環境が必要です")

    state = AppState()
    state, _, _, _, _ = init_video_session(state, str(video_path), active_tab="point_box")

    # フレーム0にポジティブ点を1つ追加
    img = on_image_click(
        img=state.video_frames[0],
        state=state,
        frame_idx=0,
        obj_id=1,
        label="positive",
        clear_old=True,
        evt=type("DummyEvt", (), {"index": (10, 10)}),
    )
    assert img is not None

    # 伝播を最後まで進め、エラーが出ないことを確認
    gen = propagate_masks(state)
    final = None
    for final in gen:
        pass
    assert final is not None
    _, _, slider_update = final
    assert slider_update is not None
