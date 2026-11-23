"""Wording x encoder control matrix (both datasets, full test sets).

Rows: matched domain wording / dataset-neutral UNI6 / cross-domain swap
(the other dataset's anchors).  Columns: Qwen3-VL-2B and CLIP-ViT-L/336.
Protocol: visual-only, w=5, to_native, pooled AUC/AP.
Output: results/vad_mamba/clip_control/wording_matrix.json
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
sys.path.insert(0, str(ROOT / "code"))

from vad_mamba.ucf_free_anchors import ANOM_PARAPHRASES, NORM_PARAPHRASES
from vad_mamba.xd_free_anchors import XD_ANOM, XD_NORM6
from vad_mamba.unified_free import UNI_ANOM6, UNI_NORM6

VD = Path("/root/autodl-tmp/vad_data")
OUT = VD / "results/vad_mamba/clip_control"
QWEN_DIR = "/root/autodl-tmp/models/Qwen3-VL-Embedding-2B"
CLIP_DIR = "/root/autodl-tmp/models/clip-vit-large-patch14-336"
INSTRUCTION = "Describe the surveillance scene and the behavior of people in it."


def auroc(sc, lb):
    pos = lb > 0.5
    n_pos = int(pos.sum()); n_neg = int(len(lb) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(sc, kind="mergesort"); r = np.empty(len(sc), dtype=np.float64)
    s2 = sc[order]; i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and s2[j + 1] == s2[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ap(sc, lb):
    pos = lb > 0.5
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    o = np.argsort(-sc, kind="mergesort"); y = pos[o].astype(np.float64)
    tp = np.cumsum(y); pr = tp / np.arange(1, len(y) + 1); rc = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], rc])) * pr))


def l2(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-5)


def smooth(s, w=5):
    return np.convolve(np.asarray(s, dtype=np.float64), np.ones(w) / w, mode="same")


def to_native(s, n):
    return s if len(s) == n else np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s)), s)


def fl(n, gt):
    y = np.zeros(n, dtype=np.int8)
    for a, b in (gt or []):
        a = int(a); b = min(int(b), n - 1)
        if b >= a:
            y[a:b + 1] = 1
    return y


def eval_feats(feats, rows, Ta, Tn):
    ss, yy = [], []
    for r in rows:
        v = r["vid"]
        if v not in feats:
            continue
        n = int(r["num_frames"])
        e = feats[v]
        s = to_native(smooth((e @ Ta.T).mean(1) - (e @ Tn.T).mean(1)), n)
        ss.append(np.asarray(s, dtype=np.float64)); yy.append(fl(n, r.get("gt") or []))
    S, Y = np.concatenate(ss), np.concatenate(yy)
    return {"auc": round(auroc(S, Y) * 100, 2), "ap": round(ap(S, Y) * 100, 2), "n": len(ss)}


def main():
    t0 = time.time()
    UCF_A, UCF_N = list(ANOM_PARAPHRASES[1:]), list(NORM_PARAPHRASES)
    XD_A, XD_N = list(XD_ANOM), list(XD_NORM6)
    UN_A, UN_N = list(UNI_ANOM6), list(UNI_NORM6)

    def rows_of(p):
        return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

    ucf_rows = rows_of(VD / "manifests/ucf_test_official.jsonl")
    xd_rows = rows_of(VD / "manifests/xd_test.jsonl")

    # ---- load cached visual features ----
    def load_npz_dir(d, rows):
        out = {}
        for r in rows:
            p = VD / d / f"{r['vid']}.npz"
            if p.exists():
                out[r["vid"]] = l2(np.asarray(np.load(p)["features"], dtype=np.float32))
        return out

    def load_npy_dir(d, rows):
        out = {}
        for r in rows:
            p = VD / d / f"{r['vid']}.npy"
            if p.exists():
                out[r["vid"]] = l2(np.load(p).astype(np.float32))
        return out

    qwen_ucf = load_npz_dir("cache/ucf/test/video", ucf_rows)
    qwen_xd = load_npz_dir("cache/xd/test/video", xd_rows)
    clip_ucf = load_npy_dir("clip_feats_ucf_v2/test", ucf_rows)
    clip_xd = load_npy_dir("clip_feats_xd/test", xd_rows)
    print(f"features qwen {len(qwen_ucf)}/{len(qwen_xd)} clip {len(clip_ucf)}/{len(clip_xd)}", flush=True)

    # ---- Qwen text anchors ----
    from uvcom_adapter.joint_encode import JointEncoder
    enc = JointEncoder(QWEN_DIR, device="cuda", instruction=INSTRUCTION)

    def qtext(texts):
        t = enc.encode_text(texts)
        if torch.is_tensor(t):
            t = t.float().cpu().numpy()
        return l2(np.asarray(t, dtype=np.float32))

    QT = {}
    for tag, (a, n) in {
        "ucf": (UCF_A, UCF_N), "xd": (XD_A, XD_N), "uni": (UN_A, UN_N),
    }.items():
        QT[tag] = (qtext(a), qtext(n))
        print(f"  qwen text {tag} ok", flush=True)
    del enc
    torch.cuda.empty_cache()

    # ---- CLIP text anchors ----
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    cp = CLIPProcessor.from_pretrained(CLIP_DIR)
    cm = CLIPModel.from_pretrained(CLIP_DIR, torch_dtype=torch.float32).eval()
    dummy = Image.new("RGB", (336, 336))

    def ctext(texts):
        x = cp(text=texts, images=[dummy], return_tensors="pt", padding=True)
        with torch.no_grad():
            o = cm(**x)
        return l2(o.text_embeds.numpy())

    CT = {}
    for tag, (a, n) in {
        "ucf": (UCF_A, UCF_N), "xd": (XD_A, XD_N), "uni": (UN_A, UN_N),
    }.items():
        CT[tag] = (ctext(a), ctext(n))
        c = CT[tag][0] @ CT[tag][1].T
        assert np.abs(c - c[0, 0]).max() > 1e-3, f"degenerate clip text {tag}"
        print(f"  clip text {tag} ok", flush=True)
    del cm
    print(f"text anchors done ({time.time()-t0:.0f}s)", flush=True)

    res = {"protocol": {"w": 5, "grid": "1 fps", "sets": "full official test sets",
                        "note": "visual-only; every cell uses the same features, only wording changes"},
           "cells": {}}

    def add(ds, enc_name, wording, feats, rows, T):
        key = f"{ds}_{enc_name}_{wording}"
        res["cells"][key] = eval_feats(feats, rows, T[0], T[1])
        print(json.dumps({key: res["cells"][key]}), flush=True)

    for ds, feats, rows, own, other in (
        ("ucf", qwen_ucf, ucf_rows, "ucf", "xd"),
        ("xd", qwen_xd, xd_rows, "xd", "ucf"),
    ):
        for wording, tag in (("domain", own), ("uni6", "uni"), ("swap", other)):
            add(ds, "qwen", wording, feats, rows, QT[tag])

    for ds, feats, rows, own, other in (
        ("ucf", clip_ucf, ucf_rows, "ucf", "xd"),
        ("xd", clip_xd, xd_rows, "xd", "ucf"),
    ):
        for wording, tag in (("domain", own), ("uni6", "uni"), ("swap", other)):
            add(ds, "clip", wording, feats, rows, CT[tag])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wording_matrix.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"done in {time.time()-t0:.0f}s -> {OUT/'wording_matrix.json'}", flush=True)


if __name__ == "__main__":
    main()
