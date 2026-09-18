"""SepID CLI entry point (PRD §15).

Minimum command shape (full behavior lands in later epics)::

    sepid separate INPUT.wav --method neural --output-dir RUN_DIR
    sepid evaluate MANIFEST.csv --methods mixture,nmf,neural,oracle --config CONFIG.yaml
    sepid report RUN_ID
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the SepID CLI argument parser.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(prog="sepid", description="SepID speech separation toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_sep = sub.add_parser("separate", help="Separate one mixture into two tracks.")
    p_sep.add_argument("input_wav", help="Input mixture WAV path.")
    p_sep.add_argument("--method", default="neural", help="Separator id (M0-M3).")
    p_sep.add_argument("--output-dir", required=True, help="Run output directory.")
    p_eval = sub.add_parser("evaluate", help="Evaluate methods on a manifest.")
    p_eval.add_argument("manifest", help="Metadata CSV path.")
    p_eval.add_argument("--methods", default="mixture", help="Comma-separated method ids.")
    p_eval.add_argument("--config", default="", help="Experiment YAML config.")
    p_rep = sub.add_parser("report", help="Render a run bundle report.")
    p_rep.add_argument("run_id", help="Run identifier under results/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (skeleton: parsing only).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
