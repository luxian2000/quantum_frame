"""Small helpers shared by visual demo scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "visual_outputs"


def add_common_visual_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where demo figures are saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving them.",
    )


def configure_matplotlib(show: bool):
    """Use a non-interactive backend unless the user asks for windows."""
    mpl_config_dir = Path(tempfile.gettempdir()) / "nexq_mpl_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path
