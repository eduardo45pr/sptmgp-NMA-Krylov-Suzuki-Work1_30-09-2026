#!/usr/bin/env python
"""
Regenera la Figura 6 del artículo usando únicamente los outputs CSV.

Entrada preferida:
    Figure6_representation_scaling.csv

Si no existe, intenta reconstruir los datos necesarios desde:
    FINAL_integrated_summary.csv

Uso:
    python plot_figure6_from_outputs.py
    python plot_figure6_from_outputs.py --input-dir ".\jcp_extended_results\small_molecules"

Salidas:
    Figure6_representation_scaling_REPLOT.png
    Figure6_representation_scaling_REPLOT.pdf
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PAPER_ORDER = [
    "Anatoxin-a",
    "Aflatoxin B1",
    "Saxitoxin",
    "Tetrodotoxin",
    "Ricinine",
    "Patulin",
    "Ochratoxin A",
    "Zearalenone",
    "T-2 toxin",
    "Citrinin",
]


def find_file(root: Path, filename: str):
    direct = root / filename
    if direct.exists():
        return direct

    hits = list(root.rglob(filename))
    return hits[0] if hits else None


def build_from_summary(summary_path: Path):
    """
    Reconstruye exactamente las magnitudes analíticas usadas por el benchmark
    para Figure 6 a partir del resumen molecular.
    """
    df = pd.read_csv(summary_path)

    required = ["Molecule", "Atoms", "Cartesian_DOF", "Compact_qubits"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "FINAL_integrated_summary.csv no contiene: " + ", ".join(missing)
        )

    d = df[df["Molecule"].isin(PAPER_ORDER)].copy()
    order = {name: i for i, name in enumerate(PAPER_ORDER)}
    d["_order"] = d["Molecule"].map(order)
    d = d.sort_values("_order").drop(columns="_order")

    q = d["Compact_qubits"].astype(int).to_numpy()
    dof = d["Cartesian_DOF"].astype(int).to_numpy()
    D = 2 ** q

    d["Compact_dimension"] = D

    # Las mismas hipótesis del benchmark:
    # Hessiana densa: float64
    # tabla Pauli completa: 4^q valores float64
    # unitaria densa: D x D complex128
    d["Dense_Hessian_float64_MiB"] = (
        dof.astype(float) ** 2 * 8.0 / (1024.0 ** 2)
    )
    d["Full_structured_Pauli_list_MiB"] = (
        (4.0 ** q) * 8.0 / (1024.0 ** 2)
    )
    d["Dense_complex_unitary_MiB"] = (
        D.astype(float) ** 2 * 16.0 / (1024.0 ** 2)
    )

    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        default=".",
        help="Directorio que contiene los outputs"
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directorio para guardar PNG/PDF"
    )
    args = ap.parse_args()

    root = Path(args.input_dir)

    # 1. Preferir el CSV específico de Figure 6.
    fig6_csv = find_file(root, "Figure6_representation_scaling.csv")

    if fig6_csv is not None:
        print("Usando output directo de Figure 6:")
        print(fig6_csv.resolve())
        d = pd.read_csv(fig6_csv)
        source_dir = fig6_csv.parent

    else:
        # 2. Fallback: reconstrucción desde el resumen ya calculado.
        summary = find_file(root, "FINAL_integrated_summary.csv")
        if summary is None:
            raise FileNotFoundError(
                "No se encontró Figure6_representation_scaling.csv ni "
                "FINAL_integrated_summary.csv dentro de:\n"
                + str(root.resolve())
            )

        print("Figure6_representation_scaling.csv no encontrado.")
        print("Reconstruyendo Figure 6 desde:")
        print(summary.resolve())

        d = build_from_summary(summary)
        source_dir = summary.parent

    required = [
        "Molecule",
        "Cartesian_DOF",
        "Compact_qubits",
        "Dense_Hessian_float64_MiB",
        "Full_structured_Pauli_list_MiB",
        "Dense_complex_unitary_MiB",
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        print("\nColumnas disponibles:")
        for c in d.columns:
            print(" ", c)
        raise RuntimeError(
            "Faltan columnas necesarias: " + ", ".join(missing)
        )

    # Mantener el orden del artículo.
    order = {name: i for i, name in enumerate(PAPER_ORDER)}
    d = d[d["Molecule"].isin(PAPER_ORDER)].copy()
    d["_order"] = d["Molecule"].map(order)
    d = d.sort_values("_order").drop(columns="_order")

    x = np.arange(len(d))

    fig, ax = plt.subplots(
        2, 1,
        figsize=(9.0, 8.0),
        sharex=True
    )

    # ----------------------------------------------------------
    # (a) Cartesian dimension vs compact register size
    # ----------------------------------------------------------
    ax[0].plot(
        x,
        d["Cartesian_DOF"],
        marker="o",
        label="Cartesian DOF"
    )

    ax[0].plot(
        x,
        d["Compact_qubits"],
        marker="s",
        label="Compact qubits"
    )

    ax[0].set_yscale("log")
    ax[0].set_ylabel("Dimension / register size")
    ax[0].set_title(
        "(a) Cartesian dimension versus compact register size"
    )
    ax[0].legend()
    ax[0].grid(alpha=0.2, which="both")

    # ----------------------------------------------------------
    # (b) Classical representation memory
    # ----------------------------------------------------------
    ax[1].plot(
        x,
        d["Dense_Hessian_float64_MiB"],
        marker="o",
        label="Dense Hessian (float64)"
    )

    ax[1].plot(
        x,
        d["Full_structured_Pauli_list_MiB"],
        marker="s",
        label="Full structured Pauli list"
    )

    ax[1].plot(
        x,
        d["Dense_complex_unitary_MiB"],
        marker="^",
        label="Dense complex unitary"
    )

    ax[1].set_yscale("log")
    ax[1].set_ylabel("Classical representation memory (MiB)")
    ax[1].set_title(
        "(b) Classical memory is not reduced by compact qubit count alone"
    )
    ax[1].legend()
    ax[1].grid(alpha=0.2, which="both")

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(
        d["Molecule"],
        rotation=45,
        ha="right"
    )

    fig.tight_layout()

    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else source_dir
    )
    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / "Figure6_representation_scaling_REPLOT.png"
    pdf = outdir / "Figure6_representation_scaling_REPLOT.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nDatos utilizados:")
    print(
        d[
            [
                "Molecule",
                "Cartesian_DOF",
                "Compact_qubits",
                "Dense_Hessian_float64_MiB",
                "Full_structured_Pauli_list_MiB",
                "Dense_complex_unitary_MiB",
            ]
        ].to_string(index=False)
    )

    print("\nFigura 6 generada:")
    print("PNG:", png.resolve())
    print("PDF:", pdf.resolve())


if __name__ == "__main__":
    main()
