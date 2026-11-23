"""CLIP-ViT-L/336 encoder control under the paper's protocol.

Computes, for UCF and XD full test sets, the frozen-CLIP visual score
    s = mean cos(frame, anomalous anchors) - mean cos(frame, normal anchors)
smoothed w=5 and interpolated to the native frame rate, over a
2x2 factorial: {Qwen (recorded), CLIP} x {domain anchors, UNI6 anchors}.

Gates (must reproduce):
  CLIP + UNI6 on UCF  -> 67.70 AUC   (recorded 2026-11-02, stride protocol)
  CLIP + UNI6 on XD   -> new
Output: results/vad_mamba/clip_control/clip_control.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)

ROOT = Path("/root/autodl-tmp/qwen_trac")
sys.path.insert(0, str(ROOT))

VD = Path("/root/autodl-tmp/vad_data")
OUT = VD / "results/vad_mamba/clip_control"
CLIP_DIR = "/root/autodl-tmp/models/clip-vit-large-patch14-336"


# ---- vendored (avoid package import chain) ----
def auroc(scores, labels):
    pos = labels > 0.5
    n_pos = int(pos.sum()); n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ss = scores[order]; i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores, labels):
    pos = labels > 0.5
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = pos[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def to_native(scores, n_native):
    if len(scores) == n_native:
        return scores
    return np.interp(np.linspace(0.0, 1.0, n_native),
                     np.linspace(0.0, 1.0, len(scores)), scores)


def l2(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-5)


def smooth(scores, w):
    if w <= 1:
        return scores
    return np.convolve(np.asarray(scores, dtype=np.float64), np.ones(w) / w, mode="same")


def frame_labels(native, gt):
    y = np.zeros(native, dtype=np.int8)
    for a, b in (gt or []):
        a = int(a); b = min(int(b), native - 1)
        if b >= a:
            y[a:b + 1] = 1
    return y


# ---- anchors (published, verbatim) ----
UCF_ANOM6 = [
    "a dangerous or illegal incident captured by a security camera",
    "suspicious criminal activity happening in a surveillance video",
    "an unusual event that poses a threat to safety or property",
    "a crime or violent act in progress in the scene",
    "abnormal behavior indicating an emergency or an offense",
    "a security threat or law-breaking event in progress",
]
UCF_NORM6 = [
    "people walking normally in a surveillance scene",
    "an ordinary surveillance scene with no unusual events",
    "normal everyday activity captured by a security camera",
    "a peaceful public scene with regular pedestrian activity",
    "routine daily activity with nothing unusual happening",
    "a calm street or indoor view with people going about their business",
]
XD_ANOM = [
    "a violent or dangerous event in a video",
    "physically aggressive behavior causing harm",
    "a fight, assault or act of violence in progress",
    "an explosion, shooting or weapon attack scene",
    "a violent scene with blood, injury or destruction",
    "people being attacked, beaten or threatened",
]
XD_NORM6 = [
    "a peaceful scene with calm everyday activity",
    "people talking, walking or working without aggression",
    "an ordinary scene from daily life or a movie with no violence",
    "a calm conversation or quiet moment in a video",
    "normal social interaction with no physical conflict",
    "a tranquil indoor or outdoor scene with nothing unusual",
]
UNI_ANOM6 = [
    "an anomalous event occurring in the video",
    "an unusual incident that deviates from normal activity",
    "abnormal or dangerous behavior happening in the scene",
    "something irregular that breaks the routine of the scene",
    "an unexpected event posing a risk to people or property",
    "an abnormal situation unfolding in front of the camera",
]
UNI_NORM6 = [
    "normal everyday activity in the scene",
    "a routine moment with nothing unusual happening",
    "ordinary calm behavior continuing as expected",
    "a typical scene following its usual course",
    "regular activity with no sign of trouble",
    "a peaceful ordinary moment in the video",
]


# prefer the server's own anchor definitions (verbatim, no transcription risk)
try:
    sys.path.insert(0, str(ROOT / "code"))
    from vad_mamba.ucf_free_anchors import ANOM_PARAPHRASES as _AP, NORM_PARAPHRASES as _NP
    from vad_mamba.xd_free_anchors import XD_ANOM as _XA, XD_NORM6 as _XN
    from vad_mamba.unified_free import UNI_ANOM6 as _UA, UNI_NORM6 as _UN
    UCF_ANOM6, UCF_NORM6 = list(_AP[1:]), list(_NP)
    XD_ANOM, XD_NORM6 = list(_XA), list(_XN)
    UNI_ANOM6, UNI_NORM6 = list(_UA), list(_UN)
    print("anchors imported from vad_mamba modules", flush=True)
except Exception as e:  # pragma: no cover
    print("falling back to hardcoded anchors:", e, flush=True)


def main():
    t0 = time.time()
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    cproc = CLIPProcessor.from_pretrained(CLIP_DIR)
    cmodel = CLIPModel.from_pretrained(CLIP_DIR, torch_dtype=torch.float32).eval()
    dummy = Image.new("RGB", (336, 336))

    def clip_text(texts):
        ct = cproc(text=texts, images=[dummy], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = cmodel(**ct)
        return l2(out.text_embeds.numpy())

    T = {}
    for tag, (an, nn) in {
        "ucf_domain": (UCF_ANOM6, UCF_NORM6),
        "xd_domain": (XD_ANOM, XD_NORM6),
        "uni6": (UNI_ANOM6, UNI_NORM6),
    }.items():
        T[tag] = (clip_text(an), clip_text(nn))
        # degenerate guard: distinct texts must not collapse
        c = T[tag][0] @ T[tag][1].T
        assert np.abs(c - c[0, 0]).max() > 1e-3, f"degenerate CLIP text for {tag}"
        print(f"  {tag}: cross-cos {c.mean():.3f} spread {np.abs(c - c[0,0]).max():.4f}", flush=True)
    del cmodel
    print(f"clip text encoded ({time.time()-t0:.0f}s)", flush=True)

    out = {"model": "CLIP-ViT-L/336 (clip-vit-large-patch14-336)",
           "protocol": {"w": 5, "grid": "1 fps, frame-aligned with the Qwen cache",
                        "sets": "full official test sets"},
           "anchors": {"ucf_domain": "UCF surveillance 6+6", "xd_domain": "XD violence 6+6",
                       "uni6": "dataset-neutral 6+6"},
           "results": {}}

    specs = [("ucf", "ucf_test_official.jsonl", "clip_feats_ucf_v2/test"),
             ("xd", "xd_test.jsonl", "clip_feats_xd/test")]
    anchor_for = {"ucf": ["ucf_domain", "uni6"], "xd": ["xd_domain", "uni6"]}

    for ds, mani, clipd in specs:
        rows = [json.loads(l) for l in (VD / "manifests" / mani).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        embs = {}
        for r in rows:
            p = VD / clipd / f"{r['vid']}.npy"
            if p.exists():
                embs[r["vid"]] = l2(np.load(p).astype(np.float32))
        print(f"{ds}: {len(embs)}/{len(rows)} clip features", flush=True)

        for atag in anchor_for[ds]:
            Ta, Tn = T[atag]
            ss, yy = [], []
            for r in rows:
                v = r["vid"]
                if v not in embs:
                    continue
                e = embs[v]
                s = (e @ Ta.T).mean(axis=1) - (e @ Tn.T).mean(axis=1)
                s = np.asarray(to_native(smooth(s, 5), int(r["num_frames"])), dtype=np.float64)
                ss.append(s)
                yy.append(frame_labels(int(r["num_frames"]), r.get("gt") or []))
            ss_all, yy_all = np.concatenate(ss), np.concatenate(yy)
            res = {"auc": round(auroc(ss_all, yy_all) * 100, 2),
                   "ap": round(average_precision(ss_all, yy_all) * 100, 2),
                   "n_videos": len(ss)}
            out["results"][f"{ds}_clip_{atag}"] = res
            print(json.dumps({f"{ds}_clip_{atag}": res}), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clip_control.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"done in {time.time()-t0:.0f}s -> {OUT/'clip_control.json'}", flush=True)


if __name__ == "__main__":
    main()
