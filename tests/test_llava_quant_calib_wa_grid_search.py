"""Focused tests for the grid search CLI entry point."""

from unittest.mock import patch

from scripts.llava_quant_calib_wa_grid_search import main
from scripts.llava_wa.search import run_grid_search
from scripts.llava_quant_calib_wa_grid_search import build_parser
from pathlib import Path
import torch
import sys
from types import SimpleNamespace
import json

def test_main_dispatches_to_run_grid_search_and_prints_summary(capsys):
    mock_summary = {"best_trial": 3, "best_score": 0.000123456789}

    with patch("sys.argv", ["llava_quant_calib_wa_grid_search.py"]), patch(
        "scripts.llava_quant_calib_wa_grid_search.run_grid_search",
        return_value=mock_summary,
    ) as mock_run:
        main()

    mock_run.assert_called_once()
    captured = capsys.readouterr()
    assert "Best trial: 3" in captured.out
    assert "validation MSE: 0.00012345679" in captured.out
    assert "elapsed:" in captured.out


def test_main_handles_empty_summary_gracefully(capsys):
    mock_summary = {"best_trial": -1, "best_score": float("inf")}

    with patch("sys.argv", ["llava_quant_calib_wa_grid_search.py"]), patch(
        "scripts.llava_quant_calib_wa_grid_search.run_grid_search",
        return_value=mock_summary,
    ) as mock_run:
        main()

    mock_run.assert_called_once()
    captured = capsys.readouterr()
    assert "Best trial: -1" in captured.out
    assert "validation MSE: inf" in captured.out


def test_main_parses_cli_arguments_before_running():
    """Verify that CLI args are forwarded to the parser inside the function."""
    mock_summary = {"best_trial": 1, "best_score": 0.5}

    with patch(
        "sys.argv",
        [
            "llava_quant_calib_wa_grid_search.py",
            "--mode",
            "grid",
            "--output-dir",
            "runs/test_out",
        ],
    ), patch(
        "scripts.llava_quant_calib_wa_grid_search.run_grid_search",
        return_value=mock_summary,
    ) as mock_run:
        main()

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args.mode == "grid"
    assert args.output_dir == "runs/test_out"
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
