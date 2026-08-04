"""Focused tests for the LLaVA weight/activation calibration orchestration."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from config.default_config import SureQuantConfig
from model.sure_quant_linear import SureQuantLinear
from model.sure_quantizer import SureQuantizer
from scripts.llava_quant_calib_wa import (
    build_parser,
    calibrate_all_quantizers,
    generate_assistant_outputs,
    loss_grid,
    run_grid_search,
)
from scripts.llava_wa.persistence import (
    INT4_WEIGHTS_NAME,
    _pack_signed_int4,
    _restore_int4_weights,
    _unpack_signed_int4,
    save_quantized_model,
)
from scripts.llava_wa.modeling import quantize_linear_layer


class _TinyModel(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        linear = nn.Linear(12, 8, bias=False, dtype=dtype)
        activation_quantizer = SureQuantizer(
            dim=12,
            block_size=4,
            num_bits=4,
            scale_granularity="per_vector_block",
        )
        weight_quantizer = SureQuantizer(
            dim=8,
            block_size=4,
            num_bits=4,
            scale_granularity="per_vector_block",
        )
        self.layer = SureQuantLinear(
            linear, activation_quantizer, weight_quantizer
        )


class _SaveableTinyModel(_TinyModel):
    def save_pretrained(self, output_dir, *, state_dict, **_kwargs):
        torch.save(state_dict, Path(output_dir) / "pytorch_model.bin")


class _SaveableProcessor:
    def save_pretrained(self, output_dir):
        (Path(output_dir) / "processor_config.json").write_text("{}")


def _config() -> SureQuantConfig:
    return SureQuantConfig(
        block_size=4,
        num_bits=4,
        calibration_steps=2,
        calibration_lr=0.02,
        calibration_batch_size=5,
        dk_sample_size=4,
        lambda_rec=1.0,
        lambda_dk=0.01,
        lambda_bal=0.01,
        lambda_range=0.01,
        device="cpu",
    )


def test_llava_wrapper_uses_fine_grained_scales_and_clip_ratio():
    wrapped = quantize_linear_layer(
        nn.Linear(12, 8, bias=False),
        num_bits=4,
        block_size=4,
        rotation_strategy="rotation",
        quantize_weight=True,
        clip_ratio=0.9,
    )

    assert wrapped.activation_quantizer.quantizer.scale_granularity == (
        "per_vector_block"
    )
    assert wrapped.weight_quantizer.quantizer.scale_granularity == (
        "per_vector_block"
    )
    assert wrapped.activation_quantizer.quantizer.clip_ratio == 0.9
    assert wrapped.weight_quantizer.quantizer.clip_ratio == 0.9


def test_calibrate_all_quantizers_calibrates_and_bakes_rectangular_weight():
    torch.manual_seed(11)
    model = _TinyModel(torch.float32)
    weight_before = model.layer.linear.weight.detach().clone()
    theta_before = (
        model.layer.weight_quantizer.rotation.givens.theta.detach().clone()
    )

    logs = calibrate_all_quantizers(
        model, {"layer": torch.randn(13, 12)}, _config()
    )

    weight_logs = logs["layer"]["weight"]
    theta_after = model.layer.weight_quantizer.rotation.givens.theta.detach()
    assert len(weight_logs) == 2
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in weight_logs)
    assert not torch.equal(theta_before, theta_after)
    assert not torch.equal(weight_before, model.layer.linear.weight)
    assert model.layer.linear.weight.shape == (8, 12)
    assert model.layer.linear.weight.grad is None


def test_calibrate_all_quantizers_preserves_half_weight_dtype():
    torch.manual_seed(12)
    model = _TinyModel(torch.float16)

    calibrate_all_quantizers(
        model, {"layer": torch.randn(8, 12, dtype=torch.float16)}, _config()
    )

    assert model.layer.linear.weight.dtype == torch.float16


def test_signed_int4_pack_round_trip_including_odd_length():
    values = torch.arange(-8, 8, dtype=torch.int8)
    values = torch.cat((values, torch.tensor([-3], dtype=torch.int8)))

    packed = _pack_signed_int4(values)
    restored = _unpack_signed_int4(packed, values.numel())

    assert packed.dtype == torch.uint8
    assert packed.numel() == (values.numel() + 1) // 2
    assert torch.equal(restored, values)


def test_save_quantized_model_stores_weights_as_packed_int4(tmp_path: Path):
    torch.manual_seed(13)
    model = _SaveableTinyModel(torch.float32)
    model.layer.quantize_weight()
    baked_weight = model.layer.linear.weight.detach().clone()
    metadata = {
        "base_checkpoint": "tiny",
        "surequant": {"num_bits": 4},
        "model_quantization": {"quantize_weight": True},
    }

    save_quantized_model(
        model,
        _SaveableProcessor(),
        tmp_path,
        metadata,
        max_shard_size="1GB",
    )

    residual = torch.load(
        tmp_path / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    artifact = torch.load(
        tmp_path / INT4_WEIGHTS_NAME, map_location="cpu", weights_only=True
    )
    saved_metadata = json.loads(
        (tmp_path / "surequant_config.json").read_text(encoding="utf-8")
    )
    layer_state = artifact["layers"]["layer"]

    assert "layer.linear.weight" not in residual
    assert artifact["num_bits"] == 4
    assert layer_state["packed_weight"].dtype == torch.uint8
    assert layer_state["packed_weight"].numel() * 2 == baked_weight.numel()
    assert layer_state["scale"].shape == (12, 2)
    assert layer_state["scale_granularity"] == "per_vector_block"
    assert saved_metadata["weight_storage"]["format"] == "surequant_packed_int4"

    model.layer.linear.weight.data.zero_()
    _restore_int4_weights(model, artifact)
    assert torch.allclose(model.layer.linear.weight, baked_weight, atol=1e-5, rtol=1e-5)


class _ProcessorBatch(dict):
    def to(self, _device):
        return self


class _InferenceProcessor:
    def apply_chat_template(self, _messages, add_generation_prompt):
        assert add_generation_prompt
        return "formatted prompt"

    def __call__(self, *, images, text, return_tensors):
        assert images.mode == "RGB"
        assert text == "formatted prompt"
        assert return_tensors == "pt"
        return _ProcessorBatch(input_ids=torch.tensor([[10, 11, 12]]))

    def batch_decode(self, token_ids, **_kwargs):
        assert token_ids.tolist() == [[21, 22]]
        return [" assistant answer "]


class _InferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, *, max_new_tokens, do_sample, **_kwargs):
        assert max_new_tokens == 16
        assert not do_sample
        suffix = torch.tensor([[21, 22]], device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


def test_generate_assistant_outputs_decodes_only_new_tokens(tmp_path: Path):
    image_paths = [tmp_path / "sample1.jpg", tmp_path / "sample2.jpg"]
    for image_path in image_paths:
        Image.new("RGB", (2, 2), color="white").save(image_path)

    outputs = generate_assistant_outputs(
        _InferenceModel(),
        _InferenceProcessor(),
        image_paths,
        prompt_text="describe",
        max_new_tokens=16,
    )

    assert outputs == [
        {"image": str(image_paths[0]), "assistant": "assistant answer"},
        {"image": str(image_paths[1]), "assistant": "assistant answer"},
    ]


def test_parser_requires_exactly_two_inference_images():
    args = build_parser().parse_args(
        ["--output-dir", "out", "--test-images", "first.jpg", "second.jpg"]
    )
    assert args.test_images == ["first.jpg", "second.jpg"]


def test_loss_grid_expands_steps_lr_and_loss_weights_to_scalar_configs():
    configs = loss_grid(
        _config(),
        {
            "calibration_steps": [2, 3],
            "calibration_lr": [0.01, 0.02],
            "lambda_rec": [1.0],
            "lambda_dk": [0.0, 0.1],
        },
    )

    assert len(configs) == 8
    assert {
        (cfg.calibration_steps, cfg.calibration_lr, cfg.lambda_dk)
        for cfg in configs
    } == {
        (steps, lr, lambda_dk)
        for steps in (2, 3)
        for lr in (0.01, 0.02)
        for lambda_dk in (0.0, 0.1)
    }
    assert all(isinstance(cfg.calibration_steps, int) for cfg in configs)
    assert all(isinstance(cfg.calibration_lr, float) for cfg in configs)


def test_clip_ratio_grid_is_expanded_and_validated():
    configs = loss_grid(_config(), {"clip_ratio": [0.9, 1.0]})

    assert [cfg.clip_ratio for cfg in configs] == [0.9, 1.0]
    with pytest.raises(ValueError, match="no greater than 1"):
        loss_grid(_config(), {"clip_ratio": [1.1]})


def test_grid_search_saves_best_weights_and_assistant_outputs(
    tmp_path: Path, monkeypatch
):
    class FakeLlava:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    processor = object()
    calibration_data = {"layer": torch.randn(4, 4)}
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(LlavaForConditionalGeneration=FakeLlava),
    )
    monkeypatch.setattr(
        "scripts.llava_wa.search.load_calib_data",
        lambda **_kwargs: (processor, calibration_data),
    )
    monkeypatch.setattr(
        "scripts.llava_wa.search.quantize_llava_model",
        lambda model, **_kwargs: model,
    )
    monkeypatch.setattr(
        "scripts.llava_wa.search.calibrate_all_quantizers",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "scripts.llava_wa.search.reconstruction_score",
        lambda *_args, **_kwargs: (0.125, {"layer": 0.125}),
    )

    def fake_save(_model, _processor, output_dir, metadata, **_kwargs):
        output_dir.mkdir(parents=True)
        (output_dir / "pytorch_model.bin").write_bytes(b"weights")
        (output_dir / "surequant_config.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    assistant_outputs = [
        {"image": "sample1.jpg", "assistant": "first answer"},
        {"image": "sample2.jpg", "assistant": "second answer"},
    ]
    monkeypatch.setattr(
        "scripts.llava_wa.search.save_quantized_model", fake_save
    )
    monkeypatch.setattr(
        "scripts.llava_wa.search.generate_assistant_outputs",
        lambda *_args, **_kwargs: assistant_outputs,
    )

    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--calibration-steps",
            "2",
            "--calibration-lr",
            "0.01",
            "--lambda-dk-grid",
            "0",
            "--lambda-bal-grid",
            "0",
            "--lambda-range-grid",
            "0",
        ]
    )
    summary = run_grid_search(args)

    best_model_dir = tmp_path / "best_quantized_model"
    assert (best_model_dir / "pytorch_model.bin").read_bytes() == b"weights"
    metadata = json.loads(
        (best_model_dir / "surequant_config.json").read_text(encoding="utf-8")
    )
    assert metadata["assistant_outputs_file"] == "../best_model_inference.json"
    inference = json.loads(
        (tmp_path / "best_model_inference.json").read_text(encoding="utf-8")
    )
    assert inference["outputs"] == assistant_outputs
    assert summary["best_trial"] == 1
    assert summary["best_assistant_outputs"] == assistant_outputs
    persisted_summary = json.loads(
        (tmp_path / "grid_search_results.json").read_text(encoding="utf-8")
    )
    assert persisted_summary["best_quantized_model_dir"] == str(best_model_dir)
