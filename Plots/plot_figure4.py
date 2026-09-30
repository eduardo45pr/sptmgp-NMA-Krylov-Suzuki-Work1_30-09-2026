#!/usr/bin/env python
"""
Regenera la Figura 4 del artículo a partir del output CSV del benchmark.

Entrada esperada:
    Patulin_Figure4_same_Hamiltonian_Trotter.csv

Columnas usadas:
    Second_order_ST_steps_r
    Unitary_relative_Frobenius_error
    One_minus_trace_fidelity

Uso:
    python plot_figure4_from_outputs.py
    python plot_figure4_from_outputs.py --input-dir jcp_extended_results/small_molecules

Salidas:
    Patulin_Figure4_same_Hamiltonian_Trotter_REPLOT.png
    Patulin_Figure4_same_Hamiltonian_Trotter_REPLOT.pdf
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def find_csv(input_dir: Path):
    filename = "Patulin_Figure4_same_Hamiltonian_Trotter.csv"

    direct = input_dir / filename
    if direct.exists():
        return direct

    hits = list(input_dir.rglob(filename))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No se encontró {filename} dentro de:\n{input_dir.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Carpeta donde están los outputs del benchmark"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta de salida. Por defecto se usa la carpeta del CSV."
    )
    args = parser.parse_args()

    csv_path = find_csv(Path(args.input_dir))
    df = pd.read_csv(csv_path)

    required = [
        "Second_order_ST_steps_r",
        "Unitary_relative_Frobenius_error",
        "One_minus_trace_fidelity",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print("\nColumnas disponibles en el CSV:")
        for c in df.columns:
            print("  ", c)
        raise RuntimeError(
            "\nFaltan columnas necesarias: " + ", ".join(missing)
        )

    df = df.sort_values("Second_order_ST_steps_r").copy()

    x = df["Second_order_ST_steps_r"].to_numpy(dtype=float)
    frob = df["Unitary_relative_Frobenius_error"].to_numpy(dtype=float)
    trace_error = df["One_minus_trace_fidelity"].to_numpy(dtype=float)

    # Recuperar el porcentaje real retenido si está disponible.
    if "Pauli_retained_percent_actual" in df.columns:
        retained = float(df["Pauli_retained_percent_actual"].iloc[0])
        title = (
            f"Patulin {retained:.2f}% Pauli stress test: "
            "exact compressed vs Trotter"
        )
    else:
        title = "Patulin 1% Pauli stress test: exact compressed vs Trotter"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        x,
        frob,
        marker="o",
        label="Unitary relative Frobenius error"
    )

    ax.plot(
        x,
        trace_error,
        marker="s",
        label="1 - trace fidelity"
    )

    # Exactamente los pasos S2 disponibles en el CSV.
    ax.set_xticks(x)

    ax.set_xlabel("Second-order Suzuki-Trotter steps, r")
    ax.set_ylabel("Same-Hamiltonian error")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else csv_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / (
        "Patulin_Figure4_same_Hamiltonian_Trotter_REPLOT.png"
    )
    pdf_path = output_dir / (
        "Patulin_Figure4_same_Hamiltonian_Trotter_REPLOT.pdf"
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

    print("\nDatos graficados:")
    print(
        df[
            [
                "Second_order_ST_steps_r",
                "Unitary_relative_Frobenius_error",
                "One_minus_trace_fidelity",
            ]
        ].to_string(index=False)
    )

    print("\nFigura 4 generada:")
    print("PNG:", png_path.resolve())
    print("PDF:", pdf_path.resolve())


if __name__ == "__main__":
    main()
