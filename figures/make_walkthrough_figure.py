"""Build the two-case walkthrough figure.

Two XD-Violence videos through the full pipeline (rebuilt published curve
family; lambda=0.1 (universal default)):
  left  column: Kingsman.The.Secret.Service.2014 B2-0-0 (shooting, 60 s)
  right column: The.Fast.and.the.Furious.2001 B6-0-0 (car accident, 50 s)
Rows share the time axis (seconds) within each column:
  row 1: filmstrip + GT shading
  row 2: anchor cosines on the 1-fps grid (raw)
  row 3: contrast before/after w=5 smoothing
  row 4: native fusion: visual, audio contribution 0.1*s_a, fused

Output: figures/fig_walkthrough.{pdf,svg,tiff,png}
Data: test_diag/walk_kingsman.json / walk_fastfurious.json (westc
ucf_test_diag, honest_rebuild.py export).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DIAG = Path("C:/Users/28385/.zcode/workspace/draft-3992a8a4-c245-449e-b35e-c93b66247510/test_diag")
CASES = [
    ("walk_full_kingsman.json",
     "Kingsman (B2-0-0, shooting)",
     FIG / "event_frames/xd/Kingsman.The.Secret.Service.2014__#00-22-10_00-23-10_label_B2-0-0"),
    ("walk_full_fastfurious.json",
     "Fast and Furious (B6-0-0, car accident)",
     FIG / "event_frames/xd/The.Fast.and.the.Furious.2001__#01-37-40_01-38-30_label_B6-0-0"),
]

AUDIT_SCRIPTS = Path("C:/Users/28385/.zcode/skills/nature-figure/scripts")
sys.path.insert(0, str(AUDIT_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.65,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
})

NORMAL_GRAY = "#CFCECE"
ANOM_COLOR = "#B64342"
NORM_LINE = "#0F4D92"
AUDIO_BLUE = "#42949E"
GT_BAND = "#B64342"
LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h"]


def main():
    loaded = []
    for json_name, title, frames in CASES:
        w = json.loads((DIAG / json_name).read_text(encoding="utf-8"))
        native = int(w["native_count"])
        fps = native / float(w["grid_T"])
        t_nat = np.arange(native) / fps
        t_grid = np.asarray(w["grid"]["t_sec"], dtype=float)
        gt = [(a / fps, b / fps) for a, b in (w["gt_intervals"] or [])]
        loaded.append((w, native, fps, t_nat, t_grid, gt, title, frames))

    fig = plt.figure(figsize=(7.16, 5.85))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.52, 0.72, 0.72, 0.87],
                          left=0.075, right=0.985, top=0.955, bottom=0.075,
                          hspace=0.16, wspace=0.17)
    axes = {}
    for col in range(2):
        for row in range(4):
            axes[(row, col)] = fig.add_subplot(gs[row, col])

    for col, (w, native, fps, t_nat, t_grid, gt, title, frames) in enumerate(loaded):
        tend = float(w["grid_T"])

        # ---- row 0: filmstrip (tiles drawn after layout, see below) ------
        ax = axes[(0, col)]
        ax.set_ylim(0, 1)
        ax.set_xlim(0, tend)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        for a, b in gt:
            ax.axvspan(a, b, ymin=0.15, ymax=0.79, color=GT_BAND, alpha=0.10, lw=0, zorder=1)
        ax.set_title(title, loc="left", fontsize=7.0, pad=2.0)

        # ---- row 1: raw anchor cosines -----------------------------------
        ax = axes[(1, col)]
        ax.plot(t_grid, w["grid"]["sim_anom"], color=ANOM_COLOR, lw=0.9,
                label="Mean cos, 6 anomalous anchors", zorder=3)
        ax.plot(t_grid, w["grid"]["sim_norm"], color=NORM_LINE, lw=0.9,
                label="Mean cos, 6 normal anchors", zorder=3)
        ax.set_ylabel("Mean cosine", labelpad=1.5, fontsize=6.2)
        ax.tick_params(labelbottom=False)
        ax.legend(loc="upper right", fontsize=5.0, frameon=False,
                  handlelength=1.2, borderaxespad=0.15, labelspacing=0.2)
        lo = min(np.min(w["grid"]["sim_anom"]), np.min(w["grid"]["sim_norm"]))
        hi = max(np.max(w["grid"]["sim_anom"]), np.max(w["grid"]["sim_norm"]))
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad + 0.28 * (hi - lo))

        # ---- row 2: raw vs smoothed contrast -----------------------------
        ax = axes[(2, col)]
        ax.plot(t_grid, w["grid"]["raw"], color="0.55", lw=0.7,
                label="Raw contrast (1 fps)", zorder=2)
        ax.plot(t_grid, w["grid"]["w5"], color=ANOM_COLOR, lw=1.1,
                label="Smoothed (w = 5)", zorder=3)
        ax.set_ylabel("Contrast score", labelpad=1.5, fontsize=6.2)
        ax.tick_params(labelbottom=False)
        ax.legend(loc="upper right", fontsize=5.0, frameon=False,
                  handlelength=1.2, borderaxespad=0.15, labelspacing=0.2)
        lo = min(np.min(w["grid"]["raw"]), np.min(w["grid"]["w5"]))
        hi = max(np.max(w["grid"]["raw"]), np.max(w["grid"]["w5"]))
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

        # ---- row 3: native fusion ----------------------------------------
        ax = axes[(3, col)]
        ax.plot(t_nat, w["native"]["visual_w5"], color="0.45", lw=0.8,
                label="Visual (w = 5, native)", zorder=2)
        ax.plot(t_nat, w["native"]["audio_contribution"], color=AUDIO_BLUE, lw=0.8,
                label="Audio contribution (0.1" + chr(183) + "s" + chr(8336) + ")", zorder=2)
        ax.plot(t_nat, w["native"]["fused"], color=ANOM_COLOR, lw=1.1,
                label="Fused = visual + 0.1" + chr(183) + "audio", zorder=3)
        ax.set_ylabel("Native-grid score", labelpad=1.5, fontsize=6.2)
        ax.set_xlabel("Time (s)", labelpad=1.5, fontsize=6.6)
        ax.legend(loc="upper right", fontsize=5.0, frameon=False,
                  handlelength=1.2, borderaxespad=0.15, labelspacing=0.2)
        ax.set_ylim(-0.1, 0.2)  # shared score scale (Fig. 4d / Fig. 8)
        ax.set_yticks([round(-0.1 + 0.05 * i, 2) for i in range(7)])
        ax.set_xticks([0, int(tend // 2), int(tend)])

        for row in range(1, 4):
            ax = axes[(row, col)]
            ax.set_xlim(0, tend)
            for a, b in gt:
                ax.axvspan(a, b, color=GT_BAND, alpha=0.10, lw=0, zorder=1)

    # ---- draw filmstrips with real layout geometry (fills the row) ----
    fig.canvas.draw()
    fig_w_in, fig_h_in = fig.get_size_inches()
    for col, (w, native, fps, t_nat, t_grid, gt, title, frames) in enumerate(loaded):
        ax = axes[(0, col)]
        tend = float(w["grid_T"])
        bb = ax.get_position()
        ax_w_in = bb.width * fig_w_in
        ax_h_in = bb.height * fig_h_in
        n_tiles = 6
        spacing = tend / n_tiles
        tile_w = spacing * 0.97
        y0 = 0.17
        h_frac = 0.60
        for i in range(n_tiles):
            sc = spacing * (i + 0.5)
            fidx = int(min(round(sc), w["grid_T"] - 1)) + 1
            img = Image.open(frames / f"{fidx:05d}.png")
            ax.imshow(img, extent=(sc - tile_w / 2, sc + tile_w / 2, y0, y0 + h_frac),
                      aspect="auto", zorder=3, interpolation="bilinear")
            ax.text(sc, y0 - 0.045, f"{int(round(sc)):d}s", ha="center", va="top",
                    fontsize=5.2, color="0.25", zorder=4)

    flat = [axes[(r, c)] for r in range(4) for c in range(2)]
    for ax_, letter in zip(flat, LETTERS):
        ax_.text(-0.115, 1.05, letter, transform=ax_.transAxes, fontsize=8,
                 fontweight="bold", ha="left", va="bottom")

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=flat, panel_ids=LETTERS,
        row_groups=[["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]],
        json_out=FIG / "fig_walkthrough.alignment-layout.json",
        overlay_svg=FIG / "fig_walkthrough.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True,
    )
    fig.savefig(FIG / "fig_walkthrough.svg", bbox_inches="tight")
    fig.savefig(FIG / "fig_walkthrough.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_walkthrough.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig_walkthrough.png", dpi=300, bbox_inches="tight")
    (FIG / "fig_walkthrough_source_data.json").write_text(json.dumps({
        "videos": [c[1] for c in CASES],
        "lambda": 0.5,
        "lambda_source": "50-video seeded holdout, AP argmax (honest protocol)",
        "curve_family": "rebuilt published family (xd_free_anchors + num_frames grid); "
                        "supersedes the h0-npz export used in the first walkthrough version",
        "panels": "rows: filmstrip / raw anchor cosines / raw vs w5 contrast / "
                  "native fusion (visual, 0.5*audio, fused); columns: the two cases",
        "source": "results/vad_mamba/ucf_test_diag walk_kingsman.json walk_fastfurious.json",
    }, indent=2), encoding="utf-8")
    plt.close(fig)
    print("saved fig_walkthrough assets (two cases)")


if __name__ == "__main__":
    main()
