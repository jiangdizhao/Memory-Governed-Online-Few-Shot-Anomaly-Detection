"""Render a versioned, fully keyed Figure 2 from the public A1 summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = [
    "FIXED05",
    "Q90:C32",
    "Q90:C64",
    "Q90:C128",
    "Q90:Cmax-G",
    "Q90:Cmax-V",
    "GAUSSIAN-Q90:C64",
    "GAUSSIAN-2.5:C64",
    "SELF",
    "LOSO",
]
MARKERS = {
    "audiojack": "o",
    "regulator": "s",
    "switch": "^",
    "rolled_strip_base": "D",
}


def render(input_csv: Path, output_png: Path) -> None:
    if output_png.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {output_png}")
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "P0":
                continue
            key = (row["category"], row["arm"])
            if key in rows:
                raise ValueError(f"Duplicate public result row: {key}")
            rows[key] = row

    colors = plt.cm.tab10(range(len(METHODS)))
    fig, ax = plt.subplots(figsize=(10, 8))
    for method, color in zip(METHODS, colors):
        for category, marker in MARKERS.items():
            row = rows.get((category, method))
            if row is None:
                raise ValueError(f"Missing P0 result row: {(category, method)}")
            ax.scatter(
                float(row["fpr"]),
                float(row["fnr"]),
                color=color,
                marker=marker,
                s=68,
                edgecolors="black",
                linewidths=0.25,
            )

    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False positive rate",
        ylabel="False negative rate",
        title="A1 P0 operating points (same frozen score vector per category)",
    )
    ax.grid(True, alpha=0.25)
    method_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color,
            markeredgecolor="black", label=method, markersize=7,
        )
        for method, color in zip(METHODS, colors)
    ]
    category_handles = [
        plt.Line2D(
            [0], [0], marker=marker, color="#444444", linestyle="None",
            label=category, markersize=7,
        )
        for category, marker in MARKERS.items()
    ]
    ax.legend(handles=category_handles, loc="upper right", fontsize=8,
              title="Category marker")
    fig.legend(handles=method_handles, loc="lower center", ncol=5,
               fontsize=8, title="Threshold arm", bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(bottom=0.24, top=0.91)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    render(args.input_csv, args.output_png)


if __name__ == "__main__":
    main()
