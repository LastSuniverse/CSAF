"""Hyperparameter sensitivity on the FULL official test sets.

(a) XD-Violence: full-test-set frame AP heatmap over w in {1,3,5} x
    lambda in [0,1] (step 0.1).
(b) UCF-Crime: full-test-set AUC vs lambda (w=5, recorded sweep), with
    the universal default lambda=0.1 marked.

Output: figures/fig_hyper_grid.{pdf,svg,tiff,png}
Data: test_diag/fullset_grid.json (XD, published family, full800),
      test_diag/ucf_fusion.json (UCF FULL sweep, recorded).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DIAG = Path("C:/Users/28385/.zcode/workspace/draft-3992a8a4-c245-449e-b35e-c93b66247510/test_diag")
AUDIT = Path("C:/Users/28385/.zcode/skills/nature-figure/scripts")
import sys
sys.path.insert(0, str(AUDIT))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.65,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
})

CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "paperblue", ["#F7F7F7", "#DDF3DE", "#AADCA9", "#767676", "#3775BA", "#0F4D92"])
SEL = "#B64342"
WS = [1, 3, 5]


def xd_panel(ax, grid):
    lam = [f"{0.1 * i:.1f}" for i in range(0, 11)]
    Z = np.array([[grid[f"w{w}"][f"lam{l}"]["ap"] for l in lam] for w in WS])
    vmin, vmax = Z.min(), Z.max()
    ax.imshow(Z, cmap=CMAP, vmin=vmin - 0.4 * (vmax - vmin), vmax=vmax,
              aspect="auto", origin="lower")
    for i in range(len(WS)):
        for j in range(len(lam)):
            color = "white" if Z[i, j] > vmin + 0.62 * (vmax - vmin) else "0.15"
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=5.6,
                    color=color)
    si, sj = WS.index(5), 1  # lambda=0.1
    ax.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False,
                               edgecolor=SEL, linewidth=1.4))
    ax.set_xticks(range(len(lam)), lam)
    ax.set_yticks(range(len(WS)), [f"w = {w}" for w in WS])
    ax.set_xlabel(r"Audio coefficient $\lambda$", labelpad=1.5, fontsize=6.6)
    ax.set_title("(a) XD-Violence: full-test-set frame AP (%)", loc="left", fontsize=7.2)
    ax.text(-0.16, 1.05, "a", transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")


def main():
    d = json.loads((DIAG / "fullset_grid.json").read_text(encoding="utf-8"))
    ucf_f = json.loads((DIAG / "ucf_fusion.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    xd_panel(axes[0], d["xd_full800_wxlambda"])

    ax = axes[1]
    u_lam = sorted(ucf_f["lambdas"], key=float)
    u_full = [round(ucf_f["lambdas"][l]["FULL"]["micro_auroc"] * 100, 2) for l in u_lam]
    ul = [float(x) for x in u_lam]
    ax.plot(ul, u_full, "o-", ms=3.2, lw=1.2, color="#3775BA")
    ax.axvline(0.1, color=SEL, lw=1.0)
    ax.annotate("default " + chr(955) + " = 0.1", (0.12, max(u_full) - 0.05),
                fontsize=6.0, color="0.25")
    ax.set_xlabel(r"Audio coefficient $\lambda$", labelpad=1.5, fontsize=6.6)
    ax.set_ylabel("UCF-Crime full-set AUC (%)", labelpad=1.5, fontsize=6.6)
    ax.set_title("(b) UCF-Crime: full-test-set frame AUC (%)", loc="left", fontsize=7.2)
    ax.text(-0.16, 1.05, "b", transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")
    ax.set_xlim(-0.03, 0.63)

    fig.subplots_adjust(bottom=0.24, top=0.86, left=0.075, right=0.99, wspace=0.30)
    fig.canvas.draw()
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig, axes=list(axes), panel_ids=["a", "b"], row_groups=[["a", "b"]],
        json_out=FIG / "fig_hyper_grid.alignment-layout.json",
        overlay_svg=FIG / "fig_hyper_grid.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=False, strict=True,
    )
    fig.savefig(FIG / "fig_hyper_grid.svg", bbox_inches="tight")
    fig.savefig(FIG / "fig_hyper_grid.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_hyper_grid.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig_hyper_grid.png", dpi=300, bbox_inches="tight")
    (FIG / "fig_hyper_grid_source_data.json").write_text(json.dumps({
        "data": {"xd": "fullset_grid.json xd_full800_wxlambda (full800, published family)",
                 "ucf": "ucf_fusion.json FULL sweep (recorded)"},
        "selected_default": "w=5, lambda=0.1 (universal, both datasets)",
    }, indent=2), encoding="utf-8")
    plt.close(fig)
    print("saved fig_hyper_grid assets (full-set)")


if __name__ == "__main__":
    main()
