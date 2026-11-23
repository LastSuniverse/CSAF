"""BLIP-ITM encoder control (rewritten for table filling).

Order optimised for Table V/VI:
  1. ucf / ucf_domain   (Table V row, matched anchors)
  2. xd  / xd_domain    (Table V row, matched anchors)
  3. xd  / uni6         (Table VI)
  4. ucf / uni6         (Table VI)
  5. ucf / xd_domain    (Table VI, cross-domain swap)
(XD cross-domain is already on disk from the earlier run.)

Per config: pooled AUC/AP + per-video visual curves (.npz) so the fused
XD row can be completed with the CLAP branch.
Incremental: writes blip_<ds>_<tag>.json and curves_blip_<ds>_<tag>.npz
right after each config, and skips configs already on disk.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/qwen_trac")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from vad_mamba.ucf_free_anchors import ANOM_PARAPHRASES, NORM_PARAPHRASES
from vad_mamba.xd_free_anchors import XD_ANOM, XD_NORM6
from vad_mamba.unified_free import UNI_ANOM6, UNI_NORM6

VD = Path("/root/autodl-tmp/vad_data")
OUT = VD / "results/vad_mamba/clip_control"
BLIP_DIR = "/root/autodl-tmp/models/blip-itm-large-coco"
LAM = 0.1


def auroc(sc, lb):
    pos = lb > 0.5
    n_pos = int(pos.sum()); n_neg = int(len(lb) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(sc, kind="mergesort")
    r = np.empty(len(sc), dtype=np.float64)
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


def rows_of(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    t0 = time.time()
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    from PIL import Image

    proc = BlipProcessor.from_pretrained(BLIP_DIR)
    model = BlipForImageTextRetrieval.from_pretrained(
        BLIP_DIR, dtype=torch.float16).to("cuda").eval()
    print("blip loaded", flush=True)

    UCF_A, UCF_N = list(ANOM_PARAPHRASES[1:]), list(NORM_PARAPHRASES)
    SETS = {"ucf_domain": (UCF_A, UCF_N),
            "xd_domain": (list(XD_ANOM), list(XD_NORM6)),
            "uni6": (list(UNI_ANOM6), list(UNI_NORM6))}
    ORDER = [("ucf", "ucf_domain"), ("xd", "xd_domain"),
             ("xd", "uni6"), ("ucf", "uni6"), ("ucf", "xd_domain")]

    FR = {"ucf": VD / "frames_ucf_1fps", "xd": VD / "frames_xd_test_1fps"}
    MANI = {"ucf": "ucf_test_official.jsonl", "xd": "xd_test.jsonl"}
    aud = {k: np.asarray(v, dtype=np.float64)
           for k, v in np.load(VD / "results/vad_mamba/xd/xd_audio_scores.npz").items()}

    for ds, atag in ORDER:
        flag = OUT / f"blip_{ds}_{atag}.json"
        if flag.exists():
            print(f"SKIP {ds}/{atag} (done)", flush=True)
            continue
        A, N = SETS[atag]
        texts = list(A) + list(N)
        n_anom = len(A)
        rows = rows_of(VD / "manifests" / MANI[ds])
        curves, yy, nv = {}, [], 0
        tcfg = time.time()
        for r in rows:
            fdir = FR[ds] / r["vid"]
            if not fdir.is_dir():
                continue
            names = sorted(p for p in fdir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
            native = int(r["num_frames"])
            pmatch = np.zeros((len(names), len(texts)), dtype=np.float64)
            pairs = [(i, j) for i in range(len(names)) for j in range(len(texts))]
            B = 128
            for i0 in range(0, len(pairs), B):
                chunk = pairs[i0:i0 + B]
                imgs = [Image.open(names[i]).convert("RGB") for i, _ in chunk]
                txts = [texts[j] for _, j in chunk]
                inp = proc(images=imgs, text=txts, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    o = model(**inp)
                sc = o.itm_score if hasattr(o, "itm_score") else o.logits
                probs = torch.softmax(sc.float(), dim=-1)[:, 1].cpu().numpy()
                for (i, j), pv in zip(chunk, probs):
                    pmatch[i, j] = pv
            s = to_native(smooth(pmatch[:, :n_anom].mean(1) - pmatch[:, n_anom:].mean(1)), native)
            curves[r["vid"]] = np.asarray(s, dtype=np.float64)
            yy.append(fl(native, r.get("gt") or []))
            nv += 1
            if nv % 50 == 0:
                print(f"  {ds}/{atag} {nv}/{len(rows)} ({time.time()-tcfg:.0f}s)", flush=True)

        S = np.concatenate([curves[r["vid"]] for r in rows if r["vid"] in curves])
        Y = np.concatenate(yy)
        res = {"auc": round(auroc(S, Y) * 100, 2), "ap": round(ap(S, Y) * 100, 2),
               "n_videos": nv}
        if ds == "xd":
            fs, fy = [], []
            for r in rows:
                v = r["vid"]
                if v not in curves:
                    continue
                c = curves[v].copy()
                if v in aud:
                    c = c + LAM * aud[v][:len(c)]
                fs.append(c); fy.append(fl(len(c), r.get("gt") or []))
            FS, FY = np.concatenate(fs), np.concatenate(fy)
            res["fused"] = {"auc": round(auroc(FS, FY) * 100, 2),
                            "ap": round(ap(FS, FY) * 100, 2)}

        np.savez_compressed(OUT / f"curves_blip_{ds}_{atag}.npz", **curves)
        (OUT / f"blip_{ds}_{atag}.json").write_text(json.dumps(
            {"model": "BLIP-ITM-large-coco",
             "score": "mean_a ITM(frame, anom) - mean_n ITM(frame, norm)",
             "protocol": {"w": 5, "grid": "1 fps", "sets": "full official test sets"},
             "anchors": atag, f"{ds}_blipitm_{atag}": res}, indent=1), encoding="utf-8")
        print(json.dumps({f"{ds}_{atag}": res}), flush=True)
        print(f"  saved ({time.time()-tcfg:.0f}s for this config)", flush=True)

    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
