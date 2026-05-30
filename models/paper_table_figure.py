from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "out" / "ch5_compare" / "paper_external_summary.csv"
DEFAULT_OUTPUT = ROOT / "out" / "figures" / "paper_external_summary_table.png"

METRIC_COLUMNS = [
    ("global_auc", "Global AUC", False),
    ("macro_auc", "Macro AUC", False),
    ("macro_f1", "Macro F1", False),
    ("macro_ece", "Macro ECE↓", True),
    ("gini", "Gini↓", True),
]


def _fmt(mean, std) -> str:
    try:
        return f"{float(mean):.4f} +/- {float(std):.4f}"
    except Exception:
        return ""


def _best_methods(df: pd.DataFrame) -> Dict[str, str]:
    best: Dict[str, str] = {}
    for metric, _, lower in METRIC_COLUMNS:
        col = f"{metric}_mean"
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        idx = vals.idxmin() if lower else vals.idxmax()
        best[metric] = str(df.loc[idx, "method_key"])
    return best


def build_table_figure(input_csv: Path, output_png: Path) -> Path:
    df = pd.read_csv(input_csv)
    if "paper_label" not in df.columns:
        df["paper_label"] = df["method_key"]

    best = _best_methods(df)
    headers = ["Method"] + [label for _, label, _ in METRIC_COLUMNS]
    table_rows: List[List[str]] = []
    bold_cells = set()

    for i, row in df.iterrows():
        display_row = [str(row["paper_label"])]
        for j, (metric, _, _) in enumerate(METRIC_COLUMNS, start=1):
            display_row.append(_fmt(row.get(f"{metric}_mean"), row.get(f"{metric}_std")))
            if str(row.get("method_key")) == best.get(metric):
                bold_cells.add((len(table_rows) + 1, j))
        table_rows.append(display_row)

    fig_h = max(2.8, 0.42 * (len(table_rows) + 2))
    fig, ax = plt.subplots(figsize=(12.8, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.25, 0.16, 0.16, 0.16, 0.16, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.45)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")
        elif c == 0:
            cell.set_text_props(ha="left")
        if (r, c) in bold_cells:
            cell.set_text_props(weight="bold")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper summary CSV as a table figure.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_table_figure(Path(args.input).resolve(), Path(args.output).resolve())
    print(f"[Done] Table figure saved to: {path}")


if __name__ == "__main__":
    main()
