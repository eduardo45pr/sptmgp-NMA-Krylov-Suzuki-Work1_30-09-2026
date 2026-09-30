#!/usr/bin/env python
"""
Regenera la Figura 3 (Pauli compression diagnostics) usando exclusivamente
el CSV generado por jcp_small_molecule_benchmark.py.

Entrada esperada:
    <Molecule>_Figure3_compression_diagnostics.csv

Ejemplos:
    python plot_figure3_from_outputs.py --molecule Aflatoxin_B1
    python plot_figure3_from_outputs.py --molecule Patulin --input-dir jcp_extended_results/small_molecules

Salidas:
    <Molecule>_Figure3_compression_diagnostics_REPLOT.png
    <Molecule>_Figure3_compression_diagnostics_REPLOT.pdf
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_name(name):
    return name.replace(" ", "_").replace("-", "_").replace("/", "_")


def find_csv(input_dir, molecule):
    input_dir = Path(input_dir)
    safe = safe_name(molecule)

    candidates = [
        input_dir / f"{safe}_Figure3_compression_diagnostics.csv",
        input_dir / f"{molecule}_Figure3_compression_diagnostics.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = list(input_dir.rglob(f"{safe}_Figure3_compression_diagnostics.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No se encontró {safe}_Figure3_compression_diagnostics.csv "
        f"dentro de {input_dir.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--molecule",
        default="Patulin",
        help="Ej.: Patulin, Aflatoxin_B1, Anatoxin_a, Citrinin, Curcumin"
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directorio donde están los outputs"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio para guardar la figura; por defecto usa el del CSV"
    )
    args = parser.parse_args()

    csv_path = find_csv(args.input_dir, args.molecule)
    df = pd.read_csv(csv_path)

    required = [
        "Pauli_terms_retained_percent",
        "Spectral_cosine",
        "Frequency_MAE_cm-1",
        "Operator_error_relative_Frobenius",
        "Discarded_coefficient_L2_weight",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Columnas encontradas:")
        for c in df.columns:
            print("  ", c)
        raise RuntimeError(
            "Faltan columnas requeridas en el CSV: " + ", ".join(missing)
        )

    # Igual que la función original del benchmark:
    # ordenar por porcentaje de términos Pauli retenidos.
    d = df.sort_values("Pauli_terms_retained_percent")
    x = d["Pauli_terms_retained_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(3, 1, figsize=(7.8, 9.2), sharex=True)

    # (a) Spectral fidelity
    ax[0].plot(
        x,
        d["Spectral_cosine"].to_numpy(dtype=float),
        marker="o"
    )
    ax[0].axhline(0.95, linestyle="--", linewidth=1.0)
    ax[0].set_title("(a) Spectral fidelity")
    ax[0].set_ylabel("Spectral cosine")
    ax[0].set_ylim(0.0, 1.05)
    ax[0].grid(alpha=0.2)

    # (b) Frequency error
    ax[1].plot(
        x,
        d["Frequency_MAE_cm-1"].to_numpy(dtype=float),
        marker="o"
    )
    ax[1].set_title("(b) Frequency error")
    ax[1].set_ylabel(r"Frequency MAE (cm$^{-1}$)")
    ax[1].grid(alpha=0.2)

    # (c) Operator error vs coefficient loss
    # El CSV mantiene los ceros exactos. Para semilogy se sustituye
    # únicamente durante la representación gráfica.
    tiny = np.finfo(float).tiny

    operator_error = np.maximum(
        d["Operator_error_relative_Frobenius"].to_numpy(dtype=float),
        tiny
    )

    coefficient_loss = np.maximum(
        d["Discarded_coefficient_L2_weight"].to_numpy(dtype=float),
        tiny
    )

    ax[2].semilogy(
        x,
        operator_error,
        marker="o",
        label="Operator error"
    )
    ax[2].semilogy(
        x,
        coefficient_loss,
        marker="s",
        label=r"Discarded coefficient $L_2$ weight"
    )
    ax[2].set_title("(c) Operator error versus coefficient-norm loss")
    ax[2].set_ylabel("Relative measure")
    ax[2].set_xlabel("Pauli terms retained (%)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.2, which="both")

    display_name = args.molecule.replace("_", " ")
    fig.suptitle(
        f"{display_name} Pauli compression diagnostics",
        y=0.995
    )

    fig.tight_layout()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else csv_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_name(args.molecule)
    png_path = output_dir / (
        f"{stem}_Figure3_compression_diagnostics_REPLOT.png"
    )
    pdf_path = output_dir / (
        f"{stem}_Figure3_compression_diagnostics_REPLOT.pdf"
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
    print("\nFigura 3 regenerada:")
    print("PNG:", png_path.resolve())
    print("PDF:", pdf_path.resolve())


if __name__ == "__main__":
    main()
