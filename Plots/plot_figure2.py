#!/usr/bin/env python
"""
Regenera la Figura 2 (Pauli validation) a partir del CSV generado por
jcp_small_molecule_benchmark.py.

Uso:
    python plot_figure2_from_outputs.py --molecule Aflatoxin_B1
    python plot_figure2_from_outputs.py --molecule Anatoxin_a --input-dir jcp_extended_results/small_molecules

El script busca:
    <Molecule>_Figure2_Pauli_validation_data.csv

y genera:
    <Molecule>_Figure2_Pauli_validation_REPLOT.png
    <Molecule>_Figure2_Pauli_validation_REPLOT.pdf
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_name(s):
    return s.replace(" ", "_").replace("-", "_").replace("/", "_")


def find_csv(input_dir: Path, molecule: str):
    safe = normalize_name(molecule)
    candidates = [
        input_dir / f"{safe}_Figure2_Pauli_validation_data.csv",
        input_dir / f"{molecule}_Figure2_Pauli_validation_data.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # búsqueda recursiva por si el CSV está dentro de small_molecules/
    hits = list(input_dir.rglob(f"{safe}_Figure2_Pauli_validation_data.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No encontré {safe}_Figure2_Pauli_validation_data.csv dentro de {input_dir.resolve()}"
    )


def pick_column(df, candidates, contains=None):
    # coincidencia exacta
    for c in candidates:
        if c in df.columns:
            return c

    # coincidencia flexible
    if contains:
        for c in df.columns:
            low = c.lower()
            if all(word.lower() in low for word in contains):
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="Aflatoxin_B1",
                    help="Ej.: Aflatoxin_B1, Anatoxin_a, Citrinin, Curcumin")
    ap.add_argument("--input-dir", default=".",
                    help="Directorio que contiene los outputs")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-cm1", type=float, default=4000.0)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    csv_path = find_csv(input_dir, args.molecule)
    df = pd.read_csv(csv_path)

    print("Leyendo:", csv_path)
    print("Columnas disponibles:")
    for c in df.columns:
        print("  ", c)

    wn_col = pick_column(
        df,
        ["Wavenumber_cm-1", "Wavenumber_cm1", "Wavenumber", "wn_cm1", "wn"],
        contains=["wavenumber"],
    )
    dense_col = pick_column(
        df,
        ["Dense_NMA_reference", "Dense_NMA", "Dense"],
        contains=["dense"],
    )
    lanczos_col = pick_column(
        df,
        ["Full_Lanczos", "Lanczos"],
        contains=["lanczos"],
    )
    full_col = pick_column(
        df,
        ["Pauli_100_percent", "Full_Pauli", "Pauli_reconstruction_100",
         "100%_Pauli_reconstruction"],
        contains=["pauli", "100"],
    )
    comp_col = pick_column(
        df,
        ["Pauli_retained_exact", "Exact_compressed", "Exact_Pauli_compressed",
         "Compressed_Pauli"],
        contains=["pauli", "retained", "exact"],
    )
    trotter_col = pick_column(
        df,
        ["Pauli_retained_Trotter_S2", "Suzuki_Trotter", "Trotter",
         "Suzuki_Trotter_S2"],
        contains=["pauli", "retained", "trotter"],
    )

    required = {
        "wavenumber": wn_col,
        "Dense NMA": dense_col,
        "Lanczos": lanczos_col,
        "100% Pauli": full_col,
        "Exact compressed": comp_col,
        "Suzuki-Trotter": trotter_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(
            "No pude identificar estas columnas: " + ", ".join(missing) +
            "\nRevise arriba la lista real de columnas del CSV."
        )

    wn = df[wn_col].to_numpy(float)
    dense = df[dense_col].to_numpy(float)
    lanczos = df[lanczos_col].to_numpy(float)
    fullp = df[full_col].to_numpy(float)
    comp = df[comp_col].to_numpy(float)
    trotter = df[trotter_col].to_numpy(float)

    # La salida original ya suele estar normalizada. Esto evita diferencias
    # accidentales si se procesa un CSV sin normalizar.
    scale = np.nanmax(np.abs(dense))
    if np.isfinite(scale) and scale > 0 and scale > 1.000001:
        dense /= scale
        lanczos /= scale
        fullp /= scale
        comp /= scale
        trotter /= scale

    fig, ax = plt.subplots(3, 1, figsize=(8.2, 9.5), sharex=True)

    ax[0].plot(wn, dense, label="Dense NMA reference")
    ax[0].plot(wn, lanczos, "--", label="Full Lanczos")
    ax[0].plot(wn, fullp, ":", label="100% Pauli reconstruction")
    ax[0].set_title("(a) Full-Hamiltonian validation")
    ax[0].set_ylabel("Normalized intensity")
    ax[0].legend(fontsize=8)

    ax[1].plot(wn, dense, label="Dense NMA reference")
    ax[1].plot(wn, comp, "-.", label="Exact Pauli retained 1%")
    ax[1].plot(wn, trotter, ":",
               label="Suzuki-Trotter S2, r=4, 1% Pauli")
    ax[1].set_title("(b) Compression and Suzuki-Trotter")
    ax[1].set_ylabel("Normalized intensity")
    ax[1].legend(fontsize=8)

    ax[2].plot(wn, lanczos-dense, "--", label="Lanczos - Dense")
    ax[2].plot(wn, fullp-dense, ":", label="100% Pauli - Dense")
    ax[2].plot(wn, comp-dense, "-.", label="Exact compressed - Dense")
    ax[2].plot(wn, trotter-dense, ":", label="Trotter - Dense")
    ax[2].axhline(0, linewidth=0.8)
    ax[2].set_title("(c) Residual spectra relative to Dense NMA")
    ax[2].set_ylabel("Residual intensity")
    ax[2].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax[2].legend(fontsize=8, ncol=2)

    for a in ax:
        a.set_xlim(0, args.max_cm1)
        a.grid(alpha=0.2)

    fig.tight_layout()

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = normalize_name(args.molecule)

    png = output_dir / f"{safe}_Figure2_Pauli_validation_REPLOT.png"
    pdf = output_dir / f"{safe}_Figure2_Pauli_validation_REPLOT.pdf"

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nFigura 2 regenerada:")
    print(" PNG:", png.resolve())
    print(" PDF:", pdf.resolve())


if __name__ == "__main__":
    main()
