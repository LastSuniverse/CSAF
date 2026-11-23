"""VadCLIP-style per-category visualization: score curves + GT shading +
normal/abnormal frame arrows. Two figures: UCF-Crime (6 cases) and
XD-Violence (6 cases). Data: test_diag/viz_categories.json (published 2B
curve family).

Style follows the reference: blue solid line = frame-wise anomaly score,
light orange shaded regions = ground-truth abnormal temporal locations,
green arrow = normal frame location, red arrow = abnormal frame location.
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

BLUE = "#3775BA"
SHADE = "#F6CFCB"
GREEN = "#1B7837"
RED = "#B2182B"

FIGS = {
    "ucf": ("fig_ucf_categories", "UCF-Crime",
            [("Arrest", "(a) Arrest"), ("Vandalism", "(b) Vandalism"),
             ("Explosion", "(c) Explosion"), ("Assault", "(d) Assault"),
             ("RoadAccidents", "(e) Road Accident"), ("Normal", "(f) Normal Video")]),
    "xd": ("fig_xd_categories", "XD-Violence",
           [("Explosion", "(a) Explosion"), ("Riot", "(b) Riot"),
            ("CarAccident", "(c) Car Accident"), ("Abuse", "(d) Abuse"),
            ("Shooting", "(e) Shooting"), ("Normal", "(f) Normal Video")]),
}


def draw_one(ax, case, letter, title):
    native = case["native"]
    x = np.arange(native)
    y = np.asarray(case["score"], dtype=np.float64)
    ax.plot(x, y, color=BLUE, lw=0.8, zorder=3)
    for a, b in case["gt"]:
        ax.axvspan(a, b, color=SHADE, alpha=0.55, lw=0, zorder=1)
    ymin, ymax = y.min(), y.max()
    rng = (ymax - ymin) or 1.0
    head_y = ymax + 0.10 * rng
    if case["green_arrow"] is not None:
        gx = case["green_arrow"]
        ax.annotate("", xy=(gx, y[gx]), xytext=(gx, head_y),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0), zorder=4)
    if case["red_arrow"] is not None:
        rx = case["red_arrow"]
        ax.annotate("", xy=(rx, y[rx]), xytext=(rx, head_y),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0), zorder=4)
    ax.set_ylim(ymin - 0.08 * rng, head_y + 0.03 * rng)
    ax.set_xlim(0, native - 1)
    ax.set_title(title, loc="left", fontsize=7.2)
    ax.tick_params(labelsize=6.0)


def main():
    d = json.loads((DIAG / "viz_categories.json").read_text(encoding="utf-8"))
    sys_txt = ("The blue solid line depicts the frame-wise anomaly detection score "
               "and the light orange shaded regions indicate the ground-truth "
               "abnormal temporal locations. Green arrow represents the location "
               "of a normal frame and red arrow the location of an abnormal frame.")
    for key, (stem, ds_name, panels) in FIGS.items():
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9))
        for i, (case_key, title) in enumerate(panels):
            ax = axes[i // 3][i % 3]
            case = d[key][case_key]
            scale = 1000 if key == "ucf" and case["native"] > 20000 else 1
            if scale != 1:
                sc = case["score"][::scale]
                case = dict(case, native=len(sc), score=sc,
                            green_arrow=min(case["green_arrow"] // scale, len(sc) - 1),
                            red_arrow=(min(case["red_arrow"] // scale, len(sc) - 1)
                                       if case["red_arrow"] is not None else None),
                            gt=[(a // scale, b // scale) for a, b in case["gt"]])
            draw_one(ax, case, "", title)
        handles = [
            plt.Line2D([], [], color=BLUE, lw=1.2, label="Anomaly score"),
            plt.Rectangle((0, 0), 1, 1, fc=SHADE, alpha=0.55, label="Ground-truth anomaly"),
            plt.Line2D([], [], lw=0, marker=r"$\downarrow$", ms=8,
                       markerfacecolor=GREEN, markeredgecolor=GREEN, label="Normal frame"),
            plt.Line2D([], [], lw=0, marker=r"$\downarrow$", ms=8,
                       markerfacecolor=RED, markeredgecolor=RED, label="Abnormal frame"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   fontsize=6.4, bbox_to_anchor=(0.5, 0.004))
        fig.suptitle(f"Visualization results of ground-truth and anomaly score on {ds_name}",
                     fontsize=7.6, y=0.995)
        fig.subplots_adjust(bottom=0.14, top=0.90, left=0.055, right=0.99,
                            wspace=0.24, hspace=0.42)
        fig.canvas.draw()
        fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight")
        fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        (FIG / f"{stem}_source_data.json").write_text(json.dumps({
            "dataset": ds_name, "cases": {k: {kk: vv for kk, vv in v.items()}
                                           for k, v in d[key].items()},
            "style": sys_txt,
        }, indent=1), encoding="utf-8")
        print(f"saved {stem}")
    print("saved category viz figures")


if __name__ == "__main__":
    main()
