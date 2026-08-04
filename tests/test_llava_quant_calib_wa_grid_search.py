"""Focused tests for the grid search CLI entry point."""

from unittest.mock import patch

from scripts.llava_quant_calib_wa_grid_search import main


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
