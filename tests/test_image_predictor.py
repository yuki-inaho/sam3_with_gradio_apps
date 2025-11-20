import torch
from PIL import Image
import pytest

from sam3.model.sam3_image_processor import Sam3Processor


class DummyPrompt:
    def __init__(self, device):
        self.device = device
        self.appended = 0

    def append_boxes(self, boxes, labels):
        assert boxes.shape[-1] == 4
        self.appended += 1


class DummyBackbone:
    def __init__(self, device):
        self.device = torch.device(device)

    def forward_image(self, images):
        batch = images.shape[0]
        feature = torch.ones((batch, 4, 4, 4), device=images.device)
        return {
            "backbone_fpn": [feature],
            "vision_pos_enc": [feature],
        }

    def forward_text(self, prompts, device):
        seq_len = len(prompts)
        features = torch.zeros((1, seq_len, 4), device=device)
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        embeds = torch.zeros_like(features)
        return {
            "language_features": features,
            "language_mask": mask,
            "language_embeds": embeds,
        }


class DummyImageModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self._device = torch.device(device)
        self.backbone = DummyBackbone(device)
        self.inst_interactive_predictor = None

    def _get_dummy_prompt(self):
        return DummyPrompt(self._device)

    def forward_grounding(self, *args, **kwargs):
        boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=self._device)
        logits = torch.tensor([[10.0]], device=self._device)
        masks = torch.ones((1, 4, 4), device=self._device)
        presence = torch.tensor([10.0], device=self._device)
        return {
            "pred_boxes": boxes,
            "pred_logits": logits,
            "pred_masks": masks,
            "presence_logit_dec": presence,
        }


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def processor(device):
    model = DummyImageModel(device)
    return Sam3Processor(model, resolution=32, device=device, confidence_threshold=0.1)


def test_processor_sets_image_and_text_prompt(processor):
    image = Image.new("RGB", (32, 24), color=(128, 128, 128))
    state = processor.set_image(image, state={})
    assert state["original_height"] == 24
    assert state["original_width"] == 32

    state = processor.set_text_prompt("an object", state)
    assert "boxes" in state and state["boxes"].shape == (1, 4)
    assert state["masks"].shape[-2:] == (24, 32)
    assert torch.all(state["scores"] <= 1)


def test_add_geometric_prompt_without_text(processor):
    image = Image.new("RGB", (16, 16), color=(64, 64, 64))
    state = processor.set_image(image, state={})
    state = processor.add_geometric_prompt([0.5, 0.5, 0.2, 0.2], True, state)
    assert state["boxes"].shape[-1] == 4
    assert state["masks"].dtype == torch.bool

    processor.reset_all_prompts(state)
    assert "boxes" not in state and "masks" not in state


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="CUDA 環境が利用できないためスキップ",
)
def test_processor_cuda118_compatibility():
    cuda_version = torch.version.cuda or ""
    if cuda_version and not cuda_version.startswith("11.8"):
        pytest.skip("CUDA 11.8 以外の環境のためスキップ")

    device = "cuda"
    model = DummyImageModel(device)
    processor = Sam3Processor(model, resolution=16, device=device)

    image = Image.new("RGB", (12, 10), color=(32, 32, 32))
    state = processor.set_image(image, state={})
    state = processor.set_text_prompt("cuda test", state)

    assert state["boxes"].is_cuda
    assert state["masks_logits"].is_cuda
