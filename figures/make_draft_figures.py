"""Generate vector draft figures from the user's recorded experiment outputs.

Only values present in PAPER_TABLES.md and unified_viz_curves.json are plotted.
Conceptual and pipeline figures remain explicit placeholders in the manuscript.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, "D:/codex-skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


DIAG = Path("C:/Users/28385/.zcode/workspace/draft-3992a8a4-c245-449e-b35e-c93b66247510/test_diag")


SEL = "#B64342"

def modality_gating():
    grid = json.loads((DIAG / "fullset_grid.json").read_text(encoding="utf-8"))
    ucf_f = json.loads((DIAG / "ucf_fusion.json").read_text(encoding="utf-8"))

    lam = [round(0.1 * i, 1) for i in range(0, 11)]
    w5 = grid["xd_full800_wxlambda"]["w5"]
    xd_ap = [w5[f"lam{l:.1f}"]["ap"] for l in lam]
    xd_auc = [w5[f"lam{l:.1f}"]["auc"] for l in lam]
    u_lam = sorted(ucf_f["lambdas"], key=float)
    u_full = [round(ucf_f["lambdas"][l]["FULL"]["micro_auroc"] * 100, 2) for l in u_lam]
    ul = [float(x) for x in u_lam]

    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.45))
    ax = axs[0]
    ax.plot(lam, xd_ap, "o-", ms=2.8, lw=1.2, color="#3775BA", label="Full-test-set AP")
    ax.plot(lam, xd_auc, "s--", ms=2.6, lw=1.1, color="0.45", label="Full-test-set AUC")
    ax.axvline(0.1, color=SEL, lw=1.0)
    ax.annotate("default " + chr(955) + " = 0.1", (0.13, 87.5), fontsize=6, color="0.25")
    ax.set(xlabel=r"Audio coefficient $\lambda$",
           ylabel="XD-Violence full test set (%)",
           title="(a) XD-Violence: universal default " + chr(955) + " = 0.1")
    ax.set_ylim(71, 96)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")

    ax = axs[1]
    ax.plot(ul, u_full, "o-", ms=2.8, lw=1.2, color="#3775BA")
    ax.axvline(0.1, color=SEL, lw=1.0)
    ax.annotate("default " + chr(955) + " = 0.1", (0.12, max(u_full) - 0.05),
                fontsize=6, color="0.25")
    ax.set(xlabel=r"Audio coefficient $\lambda$",
           ylabel="UCF-Crime full-set AUC (%)",
           title="(b) UCF-Crime: universal default within split noise")
    ax.set_ylim(84.3, 86.9)
    fig.tight_layout()
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axs),
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
        json_out=OUT / "fig3_modality_gating.alignment-layout.json",
        overlay_svg=OUT / "fig3_modality_gating.alignment.svg",
        strict=True,
    )
    fig.savefig(OUT / "fig3_modality_gating.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig3_modality_gating.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_modality_gating.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig3_modality_gating.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def category_ap():
    # XD-Violence annotation codes (official ReadMe): B1 fighting, B2 shooting,
    # B4 riot, B5 abuse, B6 car accident, G explosion; hyphenated suffixes are
    # co-occurring tags, 0 an empty slot.
    labels = [
        "B2-0-0 (shooting)",
        "B2-B1-0 (shoot.+fight.)",
        "B2-G-0 (shoot.+expl.)",
        "B2-B6-0 (shoot.+car acc.)",
        "B6-0-0 (car accident)",
        "B4-B5-0 (riot+abuse)",
        "B4-0-0 (riot)",
        "G-B2-B6 (expl.+shoot.+car)",
    ]
    honest = json.loads((DIAG / "honest_xd.json").read_text(encoding="utf-8"))
    pc = honest["per_category_ap"]
    keys = ["B2-0-0", "B2-B1-0", "B2-G-0", "B2-B6-0",
            "B6-0-0", "B4-B5-0", "B4-0-0", "G-B2-B6"]
    visual = np.array([pc["vis"][k] for k in keys])
    fusion = np.array([pc["add0.5"][k] for k in keys])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    h = 0.34
    ax.barh(y - h / 2, visual, height=h, color="#B8C4D6", label="Visual")
    ax.barh(y + h / 2, fusion, height=h, color="#3775BA", label="Visual + audio")
    for i, (v, f) in enumerate(zip(visual, fusion)):
        if f > v + 1:
            ax.text(f + 1.0, i + h / 2, f"+{f-v:.1f}", va="center", fontsize=6)
        elif f < v - 1:
            ax.text(v + 1.0, i - h / 2, f"{f-v:.1f}", va="center", fontsize=6,
                    color="#B64342")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="XD category AP (%)", ylim=(len(labels) - 0.5, -0.5))
    ax.legend(frameon=False, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), ncol=2)
    fig.tight_layout()
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        panel_ids=["category_ap"],
        json_out=OUT / "fig4_category_ap.alignment-layout.json",
        overlay_svg=OUT / "fig4_category_ap.alignment.svg",
        strict=True,
    )
    fig.savefig(OUT / "fig4_category_ap.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig4_category_ap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_category_ap.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig4_category_ap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def qualitative_curves():
    src = ROOT.parent / "results" / "unified_viz_curves.json"
    data = json.loads(src.read_text(encoding="utf-8"))

    fig, axs = plt.subplots(3, 2, figsize=(7.1, 5.0))
    # Top row: successful UCF cases.
    for ax, idx in zip(axs[0], [0, 1]):
        r = data["ucf"][idx]
        s = np.asarray(r["scores"])
        n = int(r["n"])
        t = np.linspace(0, n - 1, len(s))
        ax.plot(t, s, lw=1.0, color="#3775BA")
        for a, b in r["gt"]:
            ax.axvspan(a, b, color="#E9B7B9", alpha=0.35, lw=0)
        ax.set_title(r["vid"].replace("_x264", ""), fontsize=7)
        ax.set(xlabel="Native frame", ylabel="Score")
    # Middle-left: a failure and normal control.
    r = data["ucf"][2]
    s = np.asarray(r["scores"])
    t = np.linspace(0, int(r["n"]) - 1, len(s))
    axs[1, 0].plot(t, s, lw=1.0, color="#B04A42")
    for a, b in r["gt"]:
        axs[1, 0].axvspan(a, b, color="#E9B7B9", alpha=0.35, lw=0)
    axs[1, 0].set_title("Failure: Shoplifting031", fontsize=7)
    axs[1, 0].set(xlabel="Native frame", ylabel="Score")

    r = data["ucf"][3]
    s = np.asarray(r["scores"])
    t = np.linspace(0, int(r["n"]) - 1, len(s))
    axs[1, 1].plot(t, s, lw=1.0, color="#444444")
    axs[1, 1].set_title("Normal control", fontsize=7)
    axs[1, 1].set(xlabel="Native frame", ylabel="Score")

    # Bottom row: XD visual versus fused score.
    for ax, idx in zip(axs[2], [0, 2]):
        r = data["xd"][idx]
        n = int(r["n"])
        sv = np.asarray(r["scores_vis"])
        sf = np.asarray(r["scores_fuse"])
        tv = np.linspace(0, n - 1, len(sv))
        tf = np.linspace(0, n - 1, len(sf))
        ax.plot(tv, sv, "--", lw=0.9, color="#7D8FA6", label="Visual")
        ax.plot(tf, sf, "-", lw=1.1, color="#3775BA", label="Fused")
        for a, b in r["gt"]:
            ax.axvspan(a, b, color="#E9B7B9", alpha=0.30, lw=0)
        ax.set_title(f"XD {r['cat']}", fontsize=7)
        ax.set(xlabel="Native frame", ylabel="Score")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    modality_gating()
    category_ap()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
