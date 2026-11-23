"""LOCAL audio-anchor ablation suite (runs on this machine; no server needed).

XD (full 800, universal defaults w=5 / lambda=0.1, recorded 6+4 sound anchors):
  2b  anomalous sound-anchor count 2/4/6 (random subsets, seeds 11/22/33)
  2c  anomalous sound-anchor wording resampling (two declared rewrites)
  2d  lambda fine grid 0.05:0.05:0.30 with the recorded anchors
UCF (full 290):
  2a  recorded recipe gate: audio curve = smooth(sa) only  -> must
      reproduce the fused FULL 86.41 at lambda=0.1
      vs Eq.(4) form: audio curve = smooth(sa - sn)
CLAP text conventions replicated per dataset: XD raw pooler output,
UCF l2(pooler output). Audio embeddings l2. All curves smooth(w=5)
+ native interpolation. Fusion s = s_v + lambda * s_a.
"""
import gc
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
CLAP_DIR = r"D:/Desktop/VAD/models/clap_htsat"
CLAP_NPZ = r"D:/Desktop/VAD/clap_text_anchors.npz"
OUT = Path(r"D:/Desktop/VAD/k_audio_suite_local.json")

AUDIO_ANOM = [
    "gunshots and gunfire",
    "a car crash with screeching tires and impact",
    "screaming and shouting in panic",
    "an explosion blast",
    "breaking glass and violent smashing",
    "loud alarms, sirens and chaotic noise",
]
NORM4 = [
    "calm ambient sounds",
    "quiet conversation and gentle background noise",
    "peaceful everyday sounds",
    "soft background music",
]
# 2c: two declared rewrites of the anomalous sound anchors (normal side
# already shown insensitive by the 2/4/6 test)
ANOM_ALT1 = [
    "gunfire and shooting sounds",
    "vehicle collision with tire screech",
    "human screaming in fear",
    "explosion and blast waves",
    "glass shattering and objects smashing",
    "emergency sirens and alarm chaos",
]
ANOM_ALT2 = [
    "shots being fired",
    "cars crashing into each other",
    "people screaming loudly",
    "a bomb detonating",
    "objects being smashed loudly",
    "alarm bells ringing chaotically",
]


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


def load_manifest(p, audio=False):
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"vid": r["vid"], "native": int(r["num_frames"]),
                         "y": frame_labels(int(r["num_frames"]), r.get("gt") or []),
                         "gt": r.get("gt") or []})
    return rows


def clap_text(texts, l2norm):
    """Local CLAP text encoding. XD recipe: raw pooler; UCF recipe: l2."""
    from transformers import ClapModel, ClapProcessor
    if not hasattr(clap_text, "cache"):
        proc = ClapProcessor.from_pretrained(CLAP_DIR)
        clap = ClapModel.from_pretrained(CLAP_DIR, torch_dtype=torch.float32).eval()
        clap_text.cache = (proc, clap)
    proc, clap = clap_text.cache
    ct = proc(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        h = clap.get_text_features(**ct)
    if hasattr(h, "pooler_output"):
        h = h.pooler_output
    h = h.numpy().astype(np.float32)
    return l2(h) if l2norm else h


def main():
    t0 = time.time()
    res = {"protocol": {"w": 5, "lambda": LAM, "sets": "full XD 800 / full UCF 290"}}

    # ================= XD stage =================
    xd_rows = load_manifest(BASE / "xd_test.jsonl")
    log(f"xd rows={len(xd_rows)}")

    from uvcom_adapter.joint_encode import JointEncoder
    enc = JointEncoder(QWEN_DIR, device="cpu", instruction=INSTRUCTION)

    def encode_1by1(texts):
        return np.stack([encode_texts(enc, [t], INSTRUCTION)[0] for t in texts])

    Tva = l2(encode_1by1(XD_ANOM))
    Tvn = l2(encode_1by1(XD_NORM6))
    cross = Tva @ Tvn.T
    assert np.abs(cross - cross[0, 0]).max() > 1e-3, "degenerate visual anchors"
    log("xd visual anchors ok")

    xd_vis = {}
    for i, r in enumerate(xd_rows):
        p = BASE / "cache" / "xd" / "test" / "video" / f"{r['vid']}.npz"
        emb = l2(np.asarray(np.load(p)["features"], dtype=np.float32))
        s = (emb @ Tva.T).mean(axis=1) - (emb @ Tvn.T).mean(axis=1)
        xd_vis[r["vid"]] = np.asarray(to_native(smooth(s, 5), r["native"]), dtype=np.float64)
        if (i + 1) % 200 == 0:
            log(f"  xd visual {i+1} ({time.time()-t0:.0f}s)")
    log(f"xd visual curves done ({time.time()-t0:.0f}s)")

    # per-video audio anchor projections P_a (T,6), P_n (T,4)
    xdE = {}
    for i, r in enumerate(xd_rows):
        pa = BASE / "cache" / "xd" / "test" / "audio" / f"{r['vid']}.npz"
        if not pa.exists():
            continue
        e = l2(np.asarray(np.load(pa)["features"], dtype=np.float32))
        xdE[r["vid"]] = e
        if (i + 1) % 200 == 0:
            log(f"  xd audio cache {i+1}")
    log(f"xd audio videos={len(xdE)} ({time.time()-t0:.0f}s)")

    Ta_raw = np.load(CLAP_NPZ)["anom"]            # 6 anomalous, raw pooler (XD convention)
    Tn_raw = clap_text(NORM4, l2norm=False)       # 4 normal, raw pooler
    Pa = {v: e @ Ta_raw.T for v, e in xdE.items()}
    Pn = {v: e @ Tn_raw.T for v, e in xdE.items()}

    def xd_audio_curve(vid, anom_cols, norm_cols):
        e_len = Pa[vid].shape[0]
        sa = Pa[vid][:, anom_cols].mean(axis=1)
        sn = Pn[vid][:, norm_cols].mean(axis=1)
        return np.asarray(to_native(smooth(sa - sn, 5), xd_len[vid]), dtype=np.float64)

    xd_len = {r["vid"]: r["native"] for r in xd_rows}
    xd_by_vid = {r["vid"]: r for r in xd_rows}

    def xd_eval(curves):
        return pooled(xd_rows, curves)

    # ---- 2b anomalous anchor count ----
    res["xb_anom_count"] = {}
    for K in (2, 4, 6):
        seeds = (11, 22, 33) if K < 6 else (None,)
        runs = []
        for seed in seeds:
            if seed is None:
                cols = list(range(6))
                tag = f"K6_recorded_anom"
            else:
                rng = np.random.default_rng(seed)
                cols = sorted(rng.choice(6, size=K, replace=False).tolist())
                tag = f"K{K}_seed{seed}"
            aud = {v: xd_audio_curve(v, cols, list(range(4))) for v in xdE}
            fused = {v: xd_vis[v] + LAM * aud[v] for v in aud}
            runs.append({"tag": tag, "cols": cols, "fused": xd_eval(fused),
                         "audio_only": xd_eval(aud)})
            log(f"xd anom {tag}: {runs[-1]['fused']}")
        res["xb_anom_count"][f"K{K}"] = runs

    # ---- 2c anomalous wording ----
    res["xc_anom_wording"] = {}
    for tag, texts in (("alt1", ANOM_ALT1), ("alt2", ANOM_ALT2)):
        Tw = clap_text(texts, l2norm=False)
        aud = {}
        for v, e in xdE.items():
            sa = (e @ Tw.T).mean(axis=1)
            sn = (e @ Tn_raw.T).mean(axis=1)
            aud[v] = np.asarray(to_native(smooth(sa - sn, 5), xd_len[v]), dtype=np.float64)
        fused = {v: xd_vis[v] + LAM * aud[v] for v in aud}
        res["xc_anom_wording"][tag] = {"fused": xd_eval(fused), "audio_only": xd_eval(aud)}
        log(f"xd wording {tag}: {res['xc_anom_wording'][tag]['fused']}")

    # ---- 2d lambda fine grid (recorded 6+4) ----
    aud_rec = {v: xd_audio_curve(v, list(range(6)), list(range(4))) for v in xdE}
    res["xd_lambda_grid"] = {}
    for lam in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        fused = {v: xd_vis[v] + lam * aud_rec[v] for v in aud_rec}
        res["xd_lambda_grid"][str(lam)] = xd_eval(fused)
        log(f"xd lam {lam}: {res['xd_lambda_grid'][str(lam)]}")

    del Pa, Pn, xdE
    gc.collect()

    # ================= UCF stage (2a) =================
    ucf_rows = load_manifest(BASE / "ucf_test_official.jsonl")
    log(f"ucf rows={len(ucf_rows)}")

    # vendored from ucf_free_anchors.py (ANOM_PARAPHRASES[1:] / NORM_PARAPHRASES)
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
    Uv_a = l2(encode_1by1(UCF_ANOM6))
    Uv_n = l2(encode_1by1(UCF_NORM6))
    ucross = Uv_a @ Uv_n.T
    assert np.abs(ucross - ucross[0, 0]).max() > 1e-3, "degenerate ucf visual anchors"

    ucf_vis = {}
    for i, r in enumerate(ucf_rows):
        p = BASE / "cache" / "ucf" / "test" / "video" / f"{r['vid']}.npz"
        emb = l2(np.asarray(np.load(p)["features"], dtype=np.float32))
        s = (emb @ Uv_a.T).mean(axis=1) - (emb @ Uv_n.T).mean(axis=1)
        ucf_vis[r["vid"]] = np.asarray(to_native(smooth(s, 5), r["native"]), dtype=np.float64)
        if (i + 1) % 100 == 0:
            log(f"  ucf visual {i+1} ({time.time()-t0:.0f}s)")
    log(f"ucf visual curves done ({time.time()-t0:.0f}s)")

    # UCF recipe: l2-normalised CLAP text anchors
    Ta_u = clap_text(AUDIO_ANOM, l2norm=True)
    Tn_u = clap_text(NORM4, l2norm=True)

    ucf_aud = {"sa_only": {}, "sa_sn": {}}
    n_audio = 0
    for r in ucf_rows:
        pa = BASE / "cache" / "ucf" / "test" / "audio" / f"{r['vid']}.npz"
        if not pa.exists():
            continue
        e = l2(np.asarray(np.load(pa)["features"], dtype=np.float32))
        sa = (e @ Ta_u.T).mean(axis=1)
        sn = (e @ Tn_u.T).mean(axis=1)
        n_audio += 1
        ucf_aud["sa_only"][r["vid"]] = np.asarray(to_native(smooth(sa, 5), r["native"]),
                                                  dtype=np.float64)
        ucf_aud["sa_sn"][r["vid"]] = np.asarray(to_native(smooth(sa - sn, 5), r["native"]),
                                                dtype=np.float64)
    log(f"ucf audio videos={n_audio}")

    res["ucf_recipe_audit"] = {"n_audio": n_audio}
    for tag, aud in ucf_aud.items():
        fused = {}
        for r in ucf_rows:
            v = r["vid"]
            fused[v] = ucf_vis[v] + LAM * aud.get(v, np.zeros(r["native"], dtype=np.float64))
        res["ucf_recipe_audit"][tag] = pooled(ucf_rows, fused)
        log(f"ucf {tag}: {res['ucf_recipe_audit'][tag]}")
    res["ucf_recipe_audit"]["gate_note"] = (
        "sa_only must reproduce the recorded fused FULL 86.41 at lambda=0.1")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"all done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
