"""Build the score-distribution figure (Fig. 4): full-test-set score
behavior on BOTH datasets under universal defaults (w=5, lambda=0.1).

2x2 panels, single column:
(a) UCF-Crime video-level max-score distributions, full 290-video test set
    (visual-only scoring).
(b) XD-Violence video-level max-score distributions, full 800-video test
    set (fused, lambda=0.1).
(c) Per-video frame-AUC histograms for videos containing both classes
    (UCF visual-only; XD fused).
(d) Normal-video control curves on the native timeline (protocol control
    iii); XD curves include audio at lambda=0.1.

Inputs: test_diag/ucf_test_diag.json + control_curves.json (UCF, server),
        test_diag/xd_test_diag.json (XD, server).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DIAG = Path("C:/Users/28385/.zcode/workspace/draft-3992a8a4-c245-449e-b35e-c93b66247510/test_diag")

AUDIT_SCRIPTS = Path("C:/Users/28385/.zcode/skills/nature-figure/scripts")
sys.path.insert(0, str(AUDIT_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.65,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

BLUE = "#0F4D92"
BLUE2 = "#3775BA"
RUST = "#B64342"
TEAL = "#42949E"
NORMAL_GRAY = "#8A94A0"


def smax_hist_panel(ax, records, score_key, auc_note, bin_lo, bin_hi):
    s = np.array([r[score_key] for r in records])
    labs = np.array([r["label"] for r in records])
    bins = np.linspace(bin_lo, bin_hi, 24)
    ax.hist(s[labs == 0], bins=bins, color=NORMAL_GRAY, alpha=0.85,
            label=f"Normal ({int((labs == 0).sum())})", edgecolor="none")
    ax.hist(s[labs == 1], bins=bins, color=RUST, alpha=0.62,
            label=f"Anomalous ({int((labs == 1).sum())})", edgecolor="none")
    ax.set_xlabel("Video-level max anomaly score", labelpad=1.5)
    ax.set_ylabel("Videos", labelpad=1.5)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.18)  # headroom for the legend block
    leg = ax.legend(frameon=False, loc="upper right", handlelength=1.2,
                    borderaxespad=0.2, title=auc_note)
    leg.get_title().set_fontsize(6.0)
    leg.get_title().set_color("0.25")


def main():
    ucf = json.loads((DIAG / "ucf_test_diag.json").read_text(encoding="utf-8"))
    xd = json.loads((DIAG / "xd_test_diag.json").read_text(encoding="utf-8"))
    ucf_curves = json.loads((DIAG / "control_curves.json").read_text(encoding="utf-8"))
    xd_curves = json.loads((DIAG / "xd_normal_curves.json").read_text(encoding="utf-8"))["curves"]

    fig, axes = plt.subplots(4, 1, figsize=(3.45, 7.7))
    fig.subplots_adjust(hspace=0.62, top=0.975, bottom=0.04,
                        left=0.16, right=0.955)

    # (a) UCF video-level max scores, full 290 (visual-only).
    ax = axes[0]
    smax_hist_panel(
        ax, ucf["records"], "smax",
        f"video-level AUC {ucf['video_level_auc_full']:.2f}",
        -0.09, 0.20)
    ax.set_title("(a) UCF-Crime (full 290)", loc="left", fontsize=7.0)

    # (b) XD video-level max scores, full 800 (fused).
    ax = axes[1]
    xr = xd["records"]
    lo = min(min(r["smax_fused"] for r in xr), -0.02)
    hi = max(r["smax_fused"] for r in xr)
    smax_hist_panel(
        ax, xr, "smax_fused",
        f"video-level AUC {xd['video_level_auc_fused']:.2f}",
        lo, hi)
    ax.set_title("(b) XD-Violence (full 800)", loc="left", fontsize=7.0)

    # (c) per-video frame AUC histograms.
    ax = axes[2]
    u_pv = np.array(list(ucf["per_video_auc"].values())) * 100.0
    x_pv = np.array([v["fused"] for v in xd["per_video_auc"].values()])
    bins = np.linspace(35, 100, 27)
    ax.hist(u_pv, bins=bins, color=BLUE, alpha=0.55,
            label=f"UCF, n = {len(u_pv)}", edgecolor="none")
    ax.hist(x_pv, bins=bins, color=TEAL, alpha=0.55,
            label=f"XD, n = {len(x_pv)}", edgecolor="none")
    ax.set_xlabel("Per-video frame AUC (%)", labelpad=1.5)
    ax.set_ylabel("Videos", labelpad=1.5)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    ax.set_title("(c) Videos with both classes", loc="left", fontsize=7.0)

    # (d) normal-video control curves.
    ax = axes[3]
    styles = {"Normal_Videos_003_x264": dict(color=BLUE, ls="-",
                                             label="UCF 003"),
              "Normal_Videos_006_x264": dict(color="0.45", ls="--",
                                             label="UCF 006")}
    for vid, c in ucf_curves.items():
        st = styles.get(vid, dict(color=BLUE, ls="-", label=vid))
        s = np.array(c["s"], dtype=float)
        ax.plot(np.arange(len(s)), s, color=st["color"], ls=st["ls"],
                lw=0.9, label=st["label"])
    xd_styles = [dict(color=TEAL, ls="-"), dict(color=RUST, ls="--")]
    for i, c in enumerate(xd_curves):
        s = np.asarray(c["fused"], dtype=float)
        toks = c["vid"].split("__")[0].replace(".", " ").split()
        name = " ".join(t for t in toks if not t.isdigit())
        st = xd_styles[i % 2]
        ax.plot(np.arange(len(s)), s, color=st["color"], ls=st["ls"], lw=0.9,
                label="XD " + name)
    ax.axhline(0.0, color="0.6", lw=0.5, ls=":")
    ax.set_ylim(-0.1, 0.2)  # shared scale with the qualitative figure (Fig. 8)
    ax.set_yticks(np.arange(-0.1, 0.21, 0.05))
    ax.set_xlabel("Native frame index", labelpad=1.5)
    ax.set_ylabel("Anomaly score", labelpad=1.5)
    ax.ticklabel_format(axis="y", style="plain")
    legd = ax.legend(frameon=True, loc="upper right", fontsize=5.6,
                     handlelength=1.6, framealpha=0.85, facecolor="white",
                     edgecolor="none")
    legd.set_zorder(6)
    ax.set_title("(d) Normal-video controls", loc="left", fontsize=7.0)

    letters = iter("abcd")
    for ax_ in list(axes):
        ax_.text(-0.24, 1.10, next(letters), transform=ax_.transAxes,
                 fontsize=8, fontweight="bold", ha="left", va="bottom")

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=list(axes),
        panel_ids=["a", "b", "c", "d"],
        json_out=FIG / "fig_score_distribution.alignment-layout.json",
        overlay_svg=FIG / "fig_score_distribution.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=False, strict=True,
    )
    fig.savefig(FIG / "fig_score_distribution.svg", bbox_inches="tight")
    fig.savefig(FIG / "fig_score_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_score_distribution.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig_score_distribution.png", dpi=300, bbox_inches="tight")
    (FIG / "fig_score_distribution_source_data.json").write_text(
        json.dumps({"ucf": {"video_level_auc_full": ucf["video_level_auc_full"],
                            "n": len(ucf["records"]),
                            "n_per_video_auc": ucf["n_per_video_auc"],
                            "pooled_frame_auc_vis": 86.58},
                    "xd": {"video_level_auc_vis": xd["video_level_auc_vis"],
                           "video_level_auc_fused": xd["video_level_auc_fused"],
                           "n": xd["n_videos"], "n_normal": xd["n_normal"],
                           "n_per_video_auc": xd["n_per_video_auc"],
                           "per_video_auc_mean_fused": xd["per_video_auc_mean_fused"],
                           "pooled_frame_auc_fused": 93.21},
                    "protocol": "universal defaults w=5, lambda=0.1; full test sets",
                    "source": "results/vad_mamba (server caches)"},
                   indent=2), encoding="utf-8")
    plt.close(fig)
    print("saved fig_score_distribution assets (2x2 full-set, UCF+XD)")


if __name__ == "__main__":
    main()
