"""LOCAL rerun of the 2/4/6 normal-acoustic-anchor ablation (server is in
card-less mode with a 2GiB cgroup cap, so the Qwen visual pass runs here).

Replicates the recorded XD recipe exactly: 6 anomalous sound anchors,
normal set varies (2/4/6), RAW CLAP pooler text anchors, l2 audio
embeddings, w=5 smoothing, native interpolation, fusion s = s_v + 0.1 s_a
with visual curves from the published 2B bf16 pipeline.

Run:  D:/anaconda3/python.exe k_audio_norm_local.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)

BASE = Path(r"D:/Desktop/VAD/local_kab")
sys.path.insert(0, str(BASE / "qwen_vl_utils_stub"))  # text-only stub
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(BASE / "code"))

# ---- vendored verbatim from vad_mamba (eval.py, ucf_free.py,
#      xd_free_anchors.py) to avoid the boundary_mamba import chain ----
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
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


def to_native(scores: np.ndarray, n_native: int) -> np.ndarray:
    if len(scores) == n_native:
        return scores
    return np.interp(np.linspace(0.0, 1.0, n_native),
                     np.linspace(0.0, 1.0, len(scores)), scores)


def l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-5)


def smooth(scores: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return scores
    kernel = np.ones(w) / w
    return np.convolve(scores, kernel, mode="same")


def encode_texts(enc, texts, instruction, batch=32):
    inp = enc.emb._preprocess_inputs([
        enc.emb.format_model_input(text=t, instruction=instruction) for t in texts])
    inp = {k: (v.to(enc.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.no_grad():
        o = enc.emb.model(**inp)
        h = enc.emb._pooling_last(o.last_hidden_state, inp["attention_mask"])
    return torch.nn.functional.normalize(h, p=2, dim=-1).float().cpu().numpy()


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

INSTRUCTION = "Describe the surveillance scene and the behavior of people in it."
LAM = 0.1
QWEN_DIR = str(BASE / "models" / "Qwen3-VL-Embedding-2B")
CLAP_NPZ = r"D:/Desktop/VAD/clap_text_anchors.npz"
OUT = Path(r"D:/Desktop/VAD/k_audio_norm_local.json")


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def pooled(rows, curves):
    ss, yy = [], []
    for r in rows:
        v = r["vid"]
        if v in curves:
            ss.append(curves[v])
            yy.append(r["y"])
    return {"auc": round(auroc(np.concatenate(ss), np.concatenate(yy)) * 100, 2),
            "ap": round(average_precision(np.concatenate(ss), np.concatenate(yy)) * 100, 2)}


def frame_labels(native, gt):
    y = np.zeros(native, dtype=np.int8)
    for a, b in (gt or []):
        a = int(a); b = min(int(b), native - 1)
        if b >= a:
            y[a:b + 1] = 1
    return y


def main():
    t0 = time.time()
    rows = []
    for line in (BASE / "xd_test.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"vid": r["vid"], "native": int(r["num_frames"]),
                         "y": frame_labels(int(r["num_frames"]), r.get("gt") or [])})
    log(f"rows={len(rows)}")

    ta_npz = np.load(CLAP_NPZ)
    Ta = ta_npz["anom"]
    T6 = ta_npz["norm6"]
    T4 = T6[:4]
    log("clap text anchors loaded", Ta.shape, T4.shape, T6.shape)

    # ---- audio embeddings cache pass ----
    audE = {}
    for i, r in enumerate(rows):
        pa = BASE / "cache" / "xd" / "test" / "audio" / f"{r['vid']}.npz"
        if not pa.exists():
            continue
        z = np.load(pa)
        e = l2(np.asarray(z["features"], dtype=np.float32))
        audE[r["vid"]] = e
        if (i + 1) % 200 == 0:
            log(f"  audio cache {i+1}")
    log(f"audio embedding videos={len(audE)} ({time.time()-t0:.0f}s)")

    # ---- visual curves (published 2B bf16 recipe) ----
    from uvcom_adapter.joint_encode import JointEncoder
    enc = JointEncoder(QWEN_DIR, device="cpu", instruction=INSTRUCTION)
    # transformers 5.12 collapses a multi-conversation batch into one sample
    # locally (server version does not) -> encode ONE text per call
    def encode_texts_1by1(enc_, texts_):
        return np.stack([encode_texts(enc_, [t], INSTRUCTION)[0] for t in texts_])

    Tva = l2(encode_texts_1by1(enc, XD_ANOM))
    Tvn = l2(encode_texts_1by1(enc, XD_NORM6))
    cross = Tva @ Tvn.T
    assert np.abs(cross - cross[0, 0]).max() > 1e-3, "degenerate: all anchors encode identically"
    log(f"anchor cross-cos mean {cross.mean():.3f}")
    vis = {}
    for i, r in enumerate(rows):
        p = BASE / "cache" / "xd" / "test" / "video" / f"{r['vid']}.npz"
        emb = l2(np.asarray(np.load(p)["features"], dtype=np.float32))
        s = (emb @ Tva.T).mean(axis=1) - (emb @ Tvn.T).mean(axis=1)
        vis[r["vid"]] = np.asarray(to_native(smooth(s, 5), r["native"]), dtype=np.float64)
        if (i + 1) % 100 == 0:
            log(f"  visual {i+1} ({time.time()-t0:.0f}s)")
    log(f"visual curves done ({time.time()-t0:.0f}s)")

    def audio_curves(TnK):
        out = {}
        for r in rows:
            if r["vid"] not in audE:
                continue
            e = audE[r["vid"]]
            sa = (e @ Ta.T).mean(axis=1)
            sn = (e @ TnK.T).mean(axis=1)
            out[r["vid"]] = np.asarray(to_native(smooth(sa - sn, 5), r["native"]),
                                       dtype=np.float64)
        return out

    res = {"protocol": {"w": 5, "lambda": LAM, "set": "full XD test (800)",
                        "anom": 6,
                        "norm_sets": {"K2": "random 2-subsets of NORM4 (seeds 11/22/33)",
                                      "K4": "NORM4 (recorded)",
                                      "K6": "NORM4 + 2 declared extensions"}},
           "extension_anchors": ["wind, rain and weather ambience",
                                 "distant traffic and city hum"],
           "note": "visual curves recomputed locally (2B bf16, CPU); expect "
                   "small numeric drift vs the server fp16-cache reference",
           "configs": {}}

    def run(tag, TnK):
        aud = audio_curves(TnK)
        fused = {v: vis[v] + LAM * aud[v] for v in aud}
        cfg = {"fused": pooled(rows, fused), "audio_only": pooled(rows, aud)}
        res["configs"][tag] = cfg
        log(tag, cfg)

    for seed in (11, 22, 33):
        rng = np.random.default_rng(seed)
        idx = rng.choice(4, size=2, replace=False)
        run(f"K2_seed{seed}", T4[idx])

    run("K4_recorded", T4)
    run("K6_extended", T6)

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log("GATE K4_recorded fused =", res["configs"]["K4_recorded"]["fused"],
        "(server reference: 93.21 / 80.24)")
    log(f"all done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
