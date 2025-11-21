"""
Gradio TextタブのフローをCUIで再現し、dtypeミスマッチが出ないこととマスクが生成されることを検証。
GPU前提。models/sam3.pt と assets/videos/bedroom.mp4 が必要。
"""

from pathlib import Path

import pytest
import torch

from video_app_hf_original import (
    AppState,
    init_video_session,
    on_text_prompt,
    propagate_masks,
)


@pytest.mark.slow
def test_gradio_text_flow_smoke():
    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "models" / "sam3.pt"
    video_path = project_root / "assets" / "videos" / "bedroom.mp4"

    if not ckpt_path.exists() or not video_path.exists():
        pytest.skip("必要なファイル(models/sam3.pt, assets/videos/bedroom.mp4)が無いためスキップ")
    if not torch.cuda.is_available():
        pytest.skip("CUDA環境が必要です")

    # init
    state = AppState()
    state, _, _, _, _ = init_video_session(state, str(video_path), active_tab="text")

    # 1フレームにテキストプロンプトを適用
    img, status = on_text_prompt(state, 0, "person")
    assert img is not None
    assert isinstance(status, str)

    # 1ステップだけ伝播（中でタイプミスマッチが出ないことを確認）
    # 全フレーム伝播は時間がかかるため省略
    gen = propagate_masks(state)
    final = next(gen)
    assert final is not None
    _, status_text, slider_update = final
    assert isinstance(status_text, str)
    assert slider_update is not None
