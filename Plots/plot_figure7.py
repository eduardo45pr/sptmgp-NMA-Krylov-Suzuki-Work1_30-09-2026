#!/usr/bin/env python
"""
Regenera la Figura 7 del artículo usando únicamente el CSV de telemetría.

Entrada esperada:
    Patulin_Figure7_stage_resolved_wall_time.csv

Uso:
    python plot_figure7_from_outputs.py
    python plot_figure7_from_outputs.py --input-dir ".\jcp_extended_results\small_molecules"

Salidas:
    Patulin_Figure7_stage_resolved_wall_time_REPLOT.png
    Patulin_Figure7_stage_resolved_wall_time_REPLOT.pdf
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


STAGE_ORDER = [
    "prepare optimize 3d",
    "finite difference hessian",
    "dense nma",
    "full lanczos",
    "eigsh low modes",
    "normalize and pad",
    "pauli decomposition",
    "P-recon 100pct",
    "P-recon 10pct",
    "P-recon 1pct",
    "exact matrix exponential 1pct",
    "ST 1pct steps 1",
    "ST 1pct steps 2",
]


def find_csv(root: Path):
    filename = "Patulin_Figure7_stage_resolved_wall_time.csv"

    direct = root / filename
    if direct.exists():
        return direct

    hits = list(root.rglob(filename))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No se encontró {filename} dentro de:\n{root.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directorio que contiene los outputs"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio donde guardar la nueva Figura 7"
    )
    args = parser.parse_args()

    csv_path = find_csv(Path(args.input_dir))
    df = pd.read_csv(csv_path)

    required = ["Stage", "Wall_time_s"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print("\nColumnas disponibles:")
        for c in df.columns:
            print(" ", c)
        raise RuntimeError(
            "Faltan columnas necesarias: " + ", ".join(missing)
        )

    # Reproducir exactamente el orden utilizado por el benchmark.
    d = (
        df.set_index("Stage")
          .reindex(STAGE_ORDER)
          .dropna()
          .reset_index()
    )

    if d.empty:
        raise RuntimeError(
            "El CSV existe, pero no contiene ninguna de las etapas "
            "esperadas para la Figura 7."
        )

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    ax.barh(
        d["Stage"],
        d["Wall_time_s"]
    )

    ax.set_xscale("log")
    ax.set_xlabel("Wall time (s, log scale)")
    ax.set_title(
        "Stage-resolved wall time for the Patulin smoke test"
    )
    ax.grid(
        axis="x",
        which="both",
        alpha=0.2
    )

    fig.tight_layout()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else csv_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = (
        output_dir /
        "Patulin_Figure7_stage_resolved_wall_time_REPLOT.png"
    )

    pdf_path = (
        output_dir /
        "Patulin_Figure7_stage_resolved_wall_time_REPLOT.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nCSV utilizado:")
    print(csv_path.resolve())

    print("\nTelemetría utilizada para Figure 7:")
    print(d.to_string(index=False))

    print("\nFigura 7 generada:")
    print("PNG:", png_path.resolve())
    print("PDF:", pdf_path.resolve())


if __name__ == "__main__":
    main()
