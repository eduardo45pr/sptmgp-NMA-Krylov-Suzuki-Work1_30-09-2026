import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = Path("protein_telemetry_all.csv")

OUTPUT_PNG = "Figure_8_protein_scale_scalability_600dpi.png"
OUTPUT_PDF = "Figure_8_protein_scale_scalability_vector.pdf"

DPI = 600
D_RED = 64
NQ_RED = int(np.ceil(np.log2(D_RED)))   # 6 qubits

# ============================================================
# LOAD AND FILTER PROTEIN TELEMETRY
# ============================================================

df = pd.read_csv(CSV_FILE)

required = [
    "PDB_ID",
    "Primary_modes",
    "Cartesian_DOF",
    "End_to_end_wall_s",
    "Matrixfree_lowmode_solve_s",
    "Dense_Hessian_build_s",
    "Dense_eigensolve_s",
    "Peak_RSS_MB",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        "Missing required columns in protein_telemetry_all.csv: "
        + ", ".join(missing)
    )

# Controlled benchmark: fixed reduced modal space D_red = 64
df = df[df["Primary_modes"] == D_RED].copy()

# Preserve the same exclusion used in the previous protein benchmark figure.
df = df[df["PDB_ID"].astype(str).str.upper() != "1AKE"].copy()

# Keep the most recent run when duplicate proteins exist.
if "Timestamp_UTC" in df.columns:
    df["Timestamp_UTC"] = pd.to_datetime(
        df["Timestamp_UTC"], errors="coerce"
    )
    df = (
        df.sort_values("Timestamp_UTC")
          .drop_duplicates("PDB_ID", keep="last")
    )
else:
    df = df.drop_duplicates("PDB_ID", keep="last")

# Convert required numerical columns safely.
for col in [
    "Cartesian_DOF", "End_to_end_wall_s",
    "Matrixfree_lowmode_solve_s", "Dense_Hessian_build_s",
    "Dense_eigensolve_s", "Peak_RSS_MB"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "PDB_ID", "Cartesian_DOF", "End_to_end_wall_s",
        "Matrixfree_lowmode_solve_s", "Dense_Hessian_build_s",
        "Dense_eigensolve_s", "Peak_RSS_MB"
    ]
).copy()

if df.empty:
    raise ValueError(
        "No valid protein runs remain after filtering Primary_modes == 64."
    )

# Order all panels by physical system size.
df = df.sort_values("Cartesian_DOF").reset_index(drop=True)

protein = df["PDB_ID"].astype(str).to_numpy()
dof = df["Cartesian_DOF"].astype(int).to_numpy()
runtime = df["End_to_end_wall_s"].astype(float).to_numpy()
matrixfree = df["Matrixfree_lowmode_solve_s"].astype(float).to_numpy()
hessian = df["Dense_Hessian_build_s"].astype(float).to_numpy()
eigensolve = df["Dense_eigensolve_s"].astype(float).to_numpy()
rss_gib = df["Peak_RSS_MB"].astype(float).to_numpy() / 1024.0

# Full Cartesian index register and fixed reduced register.
nq_full = np.ceil(np.log2(dof)).astype(int)
nq_red = np.full(len(df), NQ_RED, dtype=int)

x = np.arange(len(protein))

# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 15,
    "axes.linewidth": 0.8,
})

# ============================================================
# FIGURE 8 — PROTEIN-SCALE SCALABILITY ONLY
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(13.2, 9.0),
    constrained_layout=False
)

ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

fig.subplots_adjust(
    left=0.08,
    right=0.97,
    bottom=0.10,
    top=0.89,
    wspace=0.25,
    hspace=0.38
)

fig.suptitle(
    "Protein-Scale Scalability of the Vibrational Hamiltonian Workflow",
    fontweight="bold",
    y=0.965
)

# Common protein labels.
protein_labels = [
    f"{p}\n{d:,} DOF"
    for p, d in zip(protein, dof)
]

# ============================================================
# (a) SYSTEM SIZE / CARTESIAN DOF
# ============================================================

bars_a = ax_a.bar(
    x,
    dof,
    width=0.62
)

for bar, value in zip(bars_a, dof):
    ax_a.annotate(
        f"{value:,}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold"
    )

ax_a.set_xticks(x)
ax_a.set_xticklabels(protein)
ax_a.set_xlabel("Protein")
ax_a.set_ylabel(r"Cartesian degrees of freedom, $d=3N$")
ax_a.set_title(
    "(a) System size / Cartesian DOF",
    fontweight="bold",
    loc="left"
)
ax_a.set_ylim(0, dof.max() * 1.14)
ax_a.grid(axis="y", linestyle="--", alpha=0.25)
ax_a.set_axisbelow(True)

# ============================================================
# (b) COMPUTATIONAL RUNTIME SCALING
# ============================================================

width_b = 0.62

b1 = ax_b.bar(
    x, matrixfree, width=width_b,
    color="tab:blue",
    label="Matrix-free low-mode solve"
)

b2 = ax_b.bar(
    x, hessian, width=width_b,
    bottom=matrixfree,
    color="tab:orange",
    label="Dense Hessian construction"
)

b3 = ax_b.bar(
    x, eigensolve, width=width_b,
    bottom=matrixfree + hessian,
    color="tab:green",
    label="Dense eigensolve"
)

end_to_end = ax_b.scatter(
    x, runtime,
    marker="D", s=92,
    color="tab:red",
    edgecolors="none",
    zorder=6,
    label="End-to-end wall time"
)

for xi, value in zip(x, runtime):
    ax_b.annotate(
        f"{value:.1f} s",
        (xi, value),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=8.2,
        fontweight="bold",
        zorder=7
    )

protein_labels_b = [
    f"{p}\n{d:,} DOF"
    for p, d in zip(protein, dof)
]

ax_b.set_xticks(x)
ax_b.set_xticklabels(protein_labels_b)
ax_b.set_xlabel("Protein system ordered by molecular dimension")
ax_b.set_ylabel("Runtime (s)")
ax_b.set_title(
    "(b) Computational runtime scaling",
    fontweight="bold",
    loc="left"
)

stacked_runtime = matrixfree + hessian + eigensolve
runtime_ymax = max(float(np.max(runtime)), float(np.max(stacked_runtime)))
ax_b.set_ylim(0, runtime_ymax * 1.16)

ax_b.grid(axis="y", linestyle="--", alpha=0.25)
ax_b.set_axisbelow(True)

ax_b.legend(
    handles=[end_to_end, b1, b2, b3],
    labels=[
        "End-to-end wall time",
        "Matrix-free low-mode solve",
        "Dense Hessian construction",
        "Dense eigensolve",
    ],
    frameon=False,
    loc="upper left"
)

# ============================================================
# (c) PEAK PROCESS-TREE RSS
# ============================================================

ax_c.plot(
    dof,
    rss_gib,
    marker="o",
    linewidth=1.8
)

# Collision-aware labels: alternate vertical/horizontal offsets so nearby
# protein points do not overlap.  The labels remain tied to the measured data.
offsets_c = [
    (8, 8),
    (-8, 10),
    (8, -18),
    (-8, 10),
    (8, -18),
    (-8, 10),
]

for i, (d, value, p) in enumerate(zip(dof, rss_gib, protein)):
    dx, dy = offsets_c[i % len(offsets_c)]
    ax_c.annotate(
        f"{p}\n{value:.2f} GiB",
        (d, value),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        fontsize=7.0,
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            edgecolor="none",
            alpha=0.80
        )
    )

ax_c.set_xlabel(r"Cartesian degrees of freedom, $d=3N$")
ax_c.set_ylabel("Peak process-tree RSS (GiB)")
ax_c.set_title(
    "(c) Peak process-tree RSS",
    fontweight="bold",
    loc="left"
)
ax_c.set_xlim(dof.min() * 0.91, dof.max() * 1.06)
ax_c.set_ylim(0, rss_gib.max() * 1.25)
ax_c.grid(True, linestyle="--", alpha=0.25)
ax_c.set_axisbelow(True)

# ============================================================
# (d) COMPACT REPRESENTATION
# ============================================================

ax_d.plot(
    x,
    nq_full,
    marker="o",
    linewidth=1.8,
    label=r"Full Cartesian index: $n_q^{\mathrm{full}}$"
)

ax_d.plot(
    x,
    nq_red,
    marker="s",
    linestyle="--",
    linewidth=1.8,
    label=rf"Reduced modal space: $D_{{\mathrm{{red}}}}={D_RED}$, "
          rf"$n_q^{{\mathrm{{red}}}}={NQ_RED}$"
)

for xi, q in zip(x, nq_full):
    ax_d.annotate(
        f"{q} q",
        (xi, q),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold"
    )

ax_d.set_xticks(x)
ax_d.set_xticklabels(protein)
ax_d.set_xlabel("Protein")
ax_d.set_ylabel(r"Qubit register, $n_q$")
ax_d.set_title(
    "(d) Compact representation",
    fontweight="bold",
    loc="left"
)

qmax = int(nq_full.max())
ax_d.set_ylim(NQ_RED - 0.8, qmax + 1.2)
ax_d.set_yticks(np.arange(NQ_RED, qmax + 1))
ax_d.grid(axis="y", linestyle="--", alpha=0.25)
ax_d.set_axisbelow(True)

ax_d.legend(
    frameon=False,
    loc="upper left"
)

ax_d.text(
    0.98,
    0.08,
    r"$n_q^{\mathrm{full}}=\lceil\log_2 d\rceil$"
    "\n"
    rf"$D_{{\mathrm{{red}}}}={D_RED}\Rightarrow "
    rf"n_q^{{\mathrm{{red}}}}={NQ_RED}$",
    transform=ax_d.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.5,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="0.70",
        alpha=0.95
    )
)

# ============================================================
# CLEAN AXES
# ============================================================

for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUTPUT_PNG,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nGenerated:")
print(f"  {OUTPUT_PNG} ({DPI} dpi)")
print(f"  {OUTPUT_PDF} (vector)")
print(f"  Protein systems: {len(df)}")
print(f"  D_red = {D_RED}; n_q(red) = {NQ_RED}")
