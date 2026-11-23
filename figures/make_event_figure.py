"""Build Fig. 5: qualitative filmstrip + score panels in a 3x2 grid.

Layout per user spec: two examples per row, one video per cell (filmstrip strip
above, aligned score curve below, shared native-frame coordinate), full-video
context retained with six tiles per cell. The sixth cell is a reading guide.
Score curves are unchanged from the recorded experiment; images are the
recorded 1-fps benchmark samples mapped to native-frame positions. No crop,
tonal adjustment, pseudo-colour, or compositing is applied.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FRAME_ROOT = FIG / "event_frames"
SCORE_JSON = ROOT / "source_data" / "unified_viz_curves_remote.json"
if not SCORE_JSON.exists():
    SCORE_JSON = ROOT.parent / "results" / "unified_viz_curves_remote.json"

AUDIT_SCRIPTS = Path("C:/Users/28385/.zcode/skills/nature-figure/scripts")
sys.path.insert(0, str(AUDIT_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 6.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

SUCCESS_COLOR = "#0F4D92"
FAILURE_COLOR = "#B64342"
EVENT_BAND = "#E9D5D2"
EVENT_BAND_LIGHT = "#E9D5D2"
VISUAL_COLOR = "#8996A6"

# Shared vertical scale for all score panels (recorded global range
# [-0.060, +0.166] plus margin; absolute levels stay comparable across videos).
YLO, YHI = -0.1, 0.2


def load_frames(folder: Path):
    files = sorted(folder.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(folder)
    return files


def image_for_native_time(frames, native_time, native_max):
    index = int(round(native_time / max(native_max, 1) * (len(frames) - 1)))
    return frames[max(0, min(index, len(frames) - 1))]


def all_spans(item):
    return sorted((tuple(map(float, span)) for span in item["gt"]), key=lambda z: z[0])


def phase_for_time(t, spans):
    for a, b in spans:
        if a <= t <= b:
            return "in-event"
    if not spans or t < spans[0][0]:
        return "pre-event"
    if t > spans[-1][1]:
        return "post-event"
    return "between-events"


def choose_context_times(item, n_samples=6):
    """Full-video samples forced to include pre/in/post context."""
    n = float(item["n"] - 1)
    spans = all_spans(item)
    if not spans:
        return np.linspace(0, n, n_samples).tolist()
    first_a, first_b = spans[0]
    last_a, last_b = spans[-1]
    event_len = max(first_b - first_a, last_b - last_a, 1.0)
    candidates = [
        0.02 * n,
        max(0.0, first_a - 0.30 * event_len),
        first_a,
        first_a + 0.50 * (first_b - first_a),
        last_b,
        min(n, last_b + 0.30 * event_len),
        0.98 * n,
    ]
    result = []
    for t in candidates:
        t = float(np.clip(t, 0, n))
        if not result or abs(t - result[-1]) > max(1.0, 0.012 * n):
            result.append(t)
    extra = np.linspace(0.02 * n, 0.98 * n, 12).tolist()
    for t in extra:
        if len(result) >= n_samples:
            break
        if all(abs(t - old) > max(1.0, 0.01 * n) for old in result):
            result.append(float(t))
    return sorted(result)[:n_samples]


def format_frame_tick(x, _pos=None):
    return f"{int(round(x))}"


def add_filmstrip(ax, item, frames, spans, title):
    """Tile context frames across the full shared native-frame axis."""
    n = float(item["n"] - 1)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")
    times = choose_context_times(item, n_samples=6)
    boundaries = [0.0]
    boundaries.extend((a + b) / 2.0 for a, b in zip(times[:-1], times[1:]))
    boundaries.append(n)
    provenance = []
    for i, t in enumerate(times):
        left, right = boundaries[i], boundaries[i + 1]
        path = image_for_native_time(frames, t, n)
        image = Image.open(path).convert("RGB")
        ax.imshow(image, extent=(left, right, 0.0, 1.0),
                  interpolation="none", aspect="auto", zorder=1)
        provenance.append({
            "native_frame_position": int(round(t)),
            "sample_grid_fps": 1.0,
            "sample_grid_file": path.name,
            "sample_grid_index": int(path.stem),
            "phase": phase_for_time(t, spans),
            "tile_interval_native_frame": [int(round(left)), int(round(right))],
            "mapping_method": "Voronoi tiling around recorded 1-fps samples on the shared native-frame coordinate",
        })
    for a, b in spans:
        ax.axvspan(a, b, color=EVENT_BAND_LIGHT, alpha=0.24, lw=0, zorder=2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0, pad=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", pad=1.6, fontsize=6.5, color="#28323A")
    return times, provenance


def add_score_axis(ax, item, spans, *, show_xlabel=False):
    n = int(item["n"])
    for a, b in spans:
        ax.axvspan(a, b, color=EVENT_BAND, alpha=0.55, lw=0, zorder=0)
    if "scores" in item:
        y = np.asarray(item["scores"], dtype=float)
        x = np.linspace(0, n - 1, len(y))
        color = FAILURE_COLOR if item.get("_failure") else SUCCESS_COLOR
        ax.plot(x, y, color=color, lw=0.8, zorder=3)
        ax.fill_between(x, y, np.min(y), color=color, alpha=0.07, lw=0)
    else:
        yv = np.asarray(item["scores_vis"], dtype=float)
        yf = np.asarray(item["scores_fuse"], dtype=float)
        xv = np.linspace(0, n - 1, len(yv))
        xf = np.linspace(0, n - 1, len(yf))
        ax.plot(xv, yv, color=VISUAL_COLOR, lw=0.7, ls="--", zorder=3)
        ax.plot(xf, yf, color=FAILURE_COLOR, lw=0.9, zorder=4)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(YLO, YHI)
    ax.axhline(0.0, color="0.55", lw=0.5, ls=":", zorder=1)
    ax.set_yticks([YLO, 0.0, 0.1, YHI])
    ax.yaxis.set_major_formatter(lambda v, _pos=None: f"{v:.1f}")
    ax.tick_params(axis="y", pad=1.0, length=1.4, labelsize=5.6)
    ax.set_ylabel("Anomaly\nscore", labelpad=1.2, fontsize=6.3)
    ax.set_xticks(np.linspace(0, n - 1, 4).astype(int))
    ax.xaxis.set_major_formatter(format_frame_tick)
    ax.tick_params(axis="x", pad=1.2, length=1.6, labelsize=5.8)
    if show_xlabel:
        ax.set_xlabel("Native frame index", labelpad=1.2, fontsize=6.5)


def add_guide_cell(ax):
    """Reading-guide cell: line-style and band samples with short labels."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    rows = [
        ("Shaded band", "patch", EVENT_BAND),
        ("UCF success score", "line", SUCCESS_COLOR),
        ("UCF failure score", "line", FAILURE_COLOR),
        ("Visual (XD)", "dashed", VISUAL_COLOR),
        ("Visual + audio (XD)", "line", FAILURE_COLOR),
    ]
    y = 0.88
    for label, kind, color in rows:
        if kind == "patch":
            ax.add_patch(Rectangle((0.04, y - 0.045), 0.14, 0.09,
                                   facecolor=color, alpha=0.65, edgecolor="none"))
        elif kind == "dashed":
            ax.add_line(Line2D([0.04, 0.18], [y, y], color=color, lw=1.0, ls="--"))
        else:
            ax.add_line(Line2D([0.04, 0.18], [y, y], color=color, lw=1.2))
        ax.text(0.24, y, label, ha="left", va="center", fontsize=6.3, color="#28323A")
        y -= 0.16
    ax.text(0.04, 0.06, "Filmstrip and score share the native-frame coordinate;\nimages are 1-fps samples mapped to that coordinate.",
            ha="left", va="center", fontsize=5.6, color="#4A5560", linespacing=1.25)


def main():
    data = json.loads(SCORE_JSON.read_text(encoding="utf-8"))
    ucf = {r["vid"]: r for r in data["ucf"]}
    xd = {r["vid"]: r for r in data["xd"]}

    cases = [
        {"vid": "Arson016_x264", "item": ucf["Arson016_x264"],
         "folder": FRAME_ROOT / "ucf" / "Arson016_x264",
         "title": "UCF-Crime | Arson (success)"},
        {"vid": "Robbery050_x264", "item": ucf["Robbery050_x264"],
         "folder": FRAME_ROOT / "ucf" / "Robbery050_x264",
         "title": "UCF-Crime | Robbery (success)"},
        {"vid": "Shoplifting031_x264", "item": ucf["Shoplifting031_x264"],
         "folder": FRAME_ROOT / "ucf" / "Shoplifting031_x264",
         "title": "UCF-Crime | Shoplifting (failure)"},
        {"vid": "Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0",
         "item": xd["Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0"],
         "folder": FRAME_ROOT / "xd" / "Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0",
         "title": "XD-Violence | B2-0-0 shooting"},
        {"vid": "Casino.Royale.2006__#01-46-40_01-47-14_label_B6-0-0",
         "item": xd["Casino.Royale.2006__#01-46-40_01-47-14_label_B6-0-0"],
         "folder": FRAME_ROOT / "xd" / "Casino.Royale.2006__#01-46-40_01-47-14_label_B6-0-0",
         "title": "XD-Violence | B6-0-0 car accident"},
        {"vid": "A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A",
         "item": xd["A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A"],
         "folder": FRAME_ROOT / "xd" / "A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A",
         "title": "XD-Violence | Normal clip (label A)"},
    ]
    for case in cases:
        item = case["item"]
        case["frames"] = load_frames(case["folder"])
        case["spans"] = all_spans(item)
        if case["title"].endswith("(failure)"):
            item["_failure"] = True

    fig = plt.figure(figsize=(7.16, 6.75), facecolor="white")
    grid = fig.add_gridspec(3, 2, left=0.062, right=0.992, bottom=0.115,
                            top=0.985, hspace=0.34, wspace=0.24)
    score_axes, film_axes = [], []
    provenance = []
    letters = ["a", "b", "c", "d", "e", "f"]
    for i, case in enumerate(cases):
        row, col = divmod(i, 2)
        cell = grid[row, col].subgridspec(2, 1, height_ratios=[0.52, 1.0], hspace=0.10)
        film = fig.add_subplot(cell[0, 0])
        score = fig.add_subplot(cell[1, 0])
        film_axes.append(film)
        times, frame_prov = add_filmstrip(
            film, case["item"], case["frames"], case["spans"], case["title"])
        add_score_axis(score, case["item"], case["spans"],
                       show_xlabel=(row == 2))
        score_axes.append(score)
        score.text(-0.052, 1.04, letters[i], transform=score.transAxes,
                   fontsize=8, fontweight="bold", ha="left", va="bottom")
        provenance.append({
            "panel": letters[i],
            "video_id": case["vid"],
            "title": case["title"],
            "event_intervals": [[int(a), int(b)] for a, b in case["spans"]],
            "sample_times_native_frame": [int(round(t)) for t in times],
            "frame_samples": frame_prov,
            "sampling_grid_fps": 1.0,
            "mapping_method": "Voronoi tiling around recorded 1-fps samples on the shared native-frame coordinate",
            "score_definition": "annotated interval(s) + recorded per-frame anomaly score",
            "image_processing": "uniform resize by layout; no crop or tonal adjustment",
            "selection": "full-video context samples covering pre-event, in-event, and post-event context; illustrative cases, not random prevalence sample",
        })

    legend_handles = [
        Patch(facecolor=EVENT_BAND, alpha=0.65, label="Annotated anomaly interval"),
        Line2D([], [], color=SUCCESS_COLOR, lw=1.2, label="UCF success score"),
        Line2D([], [], color=FAILURE_COLOR, lw=1.2, label="UCF failure score"),
        Line2D([], [], color=VISUAL_COLOR, lw=1.0, ls="--", label="Visual (XD)"),
        Line2D([], [], color=FAILURE_COLOR, lw=1.2, label="Visual + audio (XD)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               frameon=False, fontsize=6.0, bbox_to_anchor=(0.5, -0.006),
               handlelength=1.6, columnspacing=1.1, handletextpad=0.5)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=score_axes,
        panel_ids=["a", "b", "c", "d", "e", "f"],
        row_groups=[["a", "b"], ["c", "d"], ["e", "f"]],
        column_groups=[["a", "c", "e"], ["b", "d", "f"]],
        json_out=FIG / "fig5_qualitative.alignment-layout.json",
        overlay_svg=FIG / "fig5_qualitative.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True,
    )
    fig.savefig(FIG / "fig5_qualitative.svg", bbox_inches="tight")
    fig.savefig(FIG / "fig5_qualitative.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig5_qualitative.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig5_qualitative.png", dpi=300, bbox_inches="tight")
    (FIG / "fig5_source_data.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    plt.close(fig)
    print("saved 3x2 grid Fig. 5 assets")


if __name__ == "__main__":
    main()
