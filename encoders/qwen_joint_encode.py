"""Per-second JOINT (text+vision) encoding -- the drop-in replacement for the
CLIP two-tower features that UVCOM consumes.

    CLIP two-tower (what B runs today):
        frame_token[t] = encoder(image = frame_t)      # image only
        text_token     = encoder(text  = query)        # text only
        -> UVCOM's CIM has to fuse them

    per-second joint (this file):
        frame_token[t] = encoder(text = query, image(s) = second t)
        -> already fused inside the frozen encoder, still [L, D]

Verified facts this file encodes (probed 2026-10-12, see probe_joint.py):
  * Qwen3-VL applies a CAUSAL mask and format_model_input orders content as
    [video][text]. Vision tokens therefore NEVER attend to the query text:
    vision hidden states for two totally different queries on the same video
    have cosine 1.0000. (So you cannot harvest vision tokens to get a
    query-conditioned sequence -- that was wrong.)
  * The pooled output is the LAST text token, which DOES attend to the
    vision input. Hence encode(text=q, image=f) is genuinely both
    query-conditioned and frame-conditioned (gap 0.38 on matched vs
    mismatched query, cos 0.64 vs image-only).
  * Text-first ordering does make vision tokens query-dependent, but only
    weakly (cos 0.9336), so we use the pooled-output path instead.

Which frames make up "one second" is a first-class knob:
  --frames-per-second 1    : the single keyframe at that second
  --frames-per-second all  : every decoded frame inside that second
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_INSTRUCTION = "Find the video segment that matches the description."


def load_official(model_dir: str, script: str):
    path = Path(model_dir) / "scripts" / script
    spec = importlib.util.spec_from_file_location("je_" + script[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class JointEncoder:
    """encode(text, group of frames) -> one fused vector per group."""

    def __init__(self, model_dir: str, device: str = "cuda",
                 instruction: str = DEFAULT_INSTRUCTION):
        E = load_official(model_dir, "qwen3_vl_embedding.py")
        self.emb = E.Qwen3VLEmbedder(model_dir, torch_dtype=torch.bfloat16)
        self.emb.model.to(device).eval()
        self.device = device
        self.instruction = instruction
        self.dim = int(self.emb.model.config.text_config.hidden_size)

    def _conversations(self, texts, frame_groups):
        """[system instruction] + [user: images..., text(query)] -- text LAST,
        matching the official order, so the pooled last-text-token sees the
        frames."""
        convs = []
        for t, fr in zip(texts, frame_groups):
            content = [{"type": "image", "image": f} for f in fr]
            content.append({"type": "text", "text": t})
            convs.append([
                {"role": "system",
                 "content": [{"type": "text", "text": self.instruction}]},
                {"role": "user", "content": content},
            ])
        return convs

    @torch.no_grad()
    def encode(self, texts, frame_groups, batch: int = 16):
        """texts: list[str]; frame_groups: list[list[PIL]] (same length).
        Returns [B, D] L2-normalized."""
        assert len(texts) == len(frame_groups)
        from concurrent.futures import ThreadPoolExecutor
        starts = list(range(0, len(texts), batch))

        def prep(i):
            convs = self._conversations(texts[i:i + batch],
                                        frame_groups[i:i + batch])
            return self.emb._preprocess_inputs(convs)

        out = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(prep, starts[0]) if starts else None
            for n, i in enumerate(starts):
                cpu_inp = fut.result()
                if n + 1 < len(starts):
                    fut = pool.submit(prep, starts[n + 1])
                inp = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                       for k, v in cpu_inp.items()}
                o = self.emb.model(**inp)
                h = self.emb._pooling_last(o.last_hidden_state, inp["attention_mask"])
                out.append(F.normalize(h, p=2, dim=-1))
        return torch.cat(out, 0)

    @torch.no_grad()
    def encode_text(self, texts, batch: int = 32):
        out = []
        for i in range(0, len(texts), batch):
            inp = self.emb._preprocess_inputs(
                [self.emb.format_model_input(text=t,
                                             instruction=self.instruction)
                 for t in texts[i:i + batch]])
            inp = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                   for k, v in inp.items()}
            o = self.emb.model(**inp)
            h = self.emb._pooling_last(o.last_hidden_state, inp["attention_mask"])
            out.append(F.normalize(h, p=2, dim=-1))
        return torch.cat(out, 0)

    @torch.no_grad()
    def encode_image(self, images, batch: int = 32):
        out = []
        for i in range(0, len(images), batch):
            inp = self.emb._preprocess_inputs(
                [self.emb.format_model_input(image=im)
                 for im in images[i:i + batch]])
            inp = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                   for k, v in inp.items()}
            o = self.emb.model(**inp)
            h = self.emb._pooling_last(o.last_hidden_state, inp["attention_mask"])
            out.append(F.normalize(h, p=2, dim=-1))
        return torch.cat(out, 0)


# ------------------------------------------------------------------ helpers
def group_by_second(frames, ts, fps: float, frames_per_second):
    """Bucket decoded frames into 1-second tokens.

    frames_per_second=1   -> only the first frame of each second (keyframe)
    frames_per_second=all -> every decoded frame in that second
    Returns (groups, seconds): groups[i] = list of PIL frames for second i.
    """
    n_sec = max(1, int(np.ceil((ts[-1] + 1.0 / fps) if ts else 1.0)))
    groups = [[] for _ in range(n_sec)]
    for f, t in zip(frames, ts):
        groups[min(int(t), n_sec - 1)].append(f)
    if frames_per_second == 1:
        groups = [g[:1] for g in groups]
    return [g if g else [frames[0]] for g in groups]


def decode_video(path: str, fps: float):
    import os
    import subprocess
    import tempfile
    from PIL import Image
    from cache.canonical_video import FFMPEG_BIN, ffprobe_duration

    tmp = tempfile.mkdtemp(prefix="je_")
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", path,
             "-vf", f"fps={fps},scale=448:-2", "-q:v", "3",
             os.path.join(tmp, "f%06d.jpg")],
            check=True, timeout=300,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        names = sorted(os.listdir(tmp))
        frames = [Image.open(os.path.join(tmp, n)).convert("RGB") for n in names]
        ts = [i / fps for i in range(len(frames))]
        return frames, ts, ffprobe_duration(path)
    finally:
        for n in os.listdir(tmp):
            os.remove(os.path.join(tmp, n))
        os.rmdir(tmp)


def curve_to_window(scores: np.ndarray, thresh_ratio: float = 0.5):
    """Training-free decoder: normalise the per-second curve, threshold at
    ratio*max, take the longest contiguous run above threshold."""
    if scores.max() <= 0:
        return 0, 0
    thr = thresh_ratio * scores.max()
    best = (0, 0)
    cur = None
    for i, v in enumerate(scores):
        if v >= thr:
            cur = i if cur is None else cur
        else:
            if cur is not None and (i - cur) > (best[1] - best[0]):
                best = (cur, i)
            cur = None
    if cur is not None and (len(scores) - cur) > (best[1] - best[0]):
        best = (cur, len(scores))
    return best  # [start, end) in seconds


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------- probe
def probe(args):
    enc = JointEncoder(args.model_dir, instruction=args.instruction)
    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    if args.n:
        rows = rows[:args.n]
    print(f"probing {len(rows)} queries | frames_per_second="
          f"{args.frames_per_second} | fpres={args.frames_per_second}",
          flush=True)

    stats = {k: [] for k in ("two_tower", "joint")}
    stats.update({f"{k}_win": [] for k in ("two_tower", "joint")})

    for i, r in enumerate(rows):
        vpath = f"{args.video_dir}/{r['vid']}.mp4"
        if not Path(vpath).exists():
            continue
        try:
            frames, ts, dur = decode_video(vpath, args.fps)
        except Exception:
            continue
        if len(frames) < 2:
            continue
        groups = group_by_second(frames, ts, args.fps, args.frames_per_second)
        L = len(groups)
        q = r["query"]
        gt = r["relevant_windows"][0]

        # --- (a) two-tower baseline: cos(text, image) per second
        img_emb = enc.encode_image([g[0] for g in groups])          # [L, D]
        txt_emb = enc.encode_text([q])                              # [1, D]
        a_curve = (img_emb @ txt_emb.T).squeeze(-1).float().cpu().numpy()

        # --- (b) per-second joint: cos(joint(q, second), text-only)
        joint = enc.encode([q] * L, groups)                         # [L, D]
        b_curve = (joint @ txt_emb.T).squeeze(-1).float().cpu().numpy()

        for name, curve in (("two_tower", a_curve), ("joint", b_curve)):
            if not np.isfinite(curve).all():
                continue
            w = curve_to_window(curve, args.thresh_ratio)
            stats[name].append(iou(w, gt))
            stats[f"{name}_win"].append(
                (curve[:min(L, int(gt[1]) + 1)].max() if gt[1] <= L
                 else curve.max()))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
            for name in ("two_tower", "joint"):
                v = np.array(stats[name])
                if len(v):
                    print(f"    {name:10s} IoU(mean)={v.mean():.3f} "
                          f"R@0.3={np.mean(v>=0.3)*100:.1f} "
                          f"R@0.5={np.mean(v>=0.5)*100:.1f} "
                          f"R@0.7={np.mean(v>=0.7)*100:.1f}", flush=True)

    print("\n===== FINAL =====")
    for name in ("two_tower", "joint"):
        v = np.array(stats[name])
        if not len(v):
            continue
        print(f"{name:10s} n={len(v)} IoU={v.mean():.3f} "
              f"R@0.3={np.mean(v>=0.3)*100:.1f} "
              f"R@0.5={np.mean(v>=0.5)*100:.1f} "
              f"R@0.7={np.mean(v>=0.7)*100:.1f}")


def linprobe(args):
    """Controlled feature comparison: identical frames, identical decoder,
    identical per-second classifier -- only the FEATURE changes.

      two_tower : encode(image=frame_t)                     (what B feeds UVCOM)
      joint     : encode(text=query, image(s)=second t)     (the proposal)

    A per-second logistic regression decides in-GT vs not; the resulting
    curve is decoded the same way for both. This isolates feature quality
    from any temporal model, which is exactly the claim being tested.
    """
    from sklearn.linear_model import LogisticRegression

    enc = JointEncoder(args.model_dir, instruction=args.instruction)
    train_rows = [json.loads(l) for l in
                  open(args.train_labels, encoding="utf-8")]
    test_rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]

    def collect(rows, maxn):
        Xj, Xt, Y = [], [], []
        used = 0
        for r in rows:
            if used >= maxn:
                break
            vpath = f"{args.video_dir}/{r['vid']}.mp4"
            if not Path(vpath).exists():
                continue
            try:
                frames, ts, dur = decode_video(vpath, args.fps)
            except Exception:
                continue
            if len(frames) < 2:
                continue
            groups = group_by_second(frames, ts, args.fps,
                                     args.frames_per_second)
            L = len(groups)
            q = r["query"]
            gt = r["relevant_windows"][0]
            qq = [q] * L
            Xj.append(enc.encode(qq, groups).float().cpu())
            Xt.append(enc.encode_image([g[0] for g in groups]).float().cpu())
            y = np.zeros(L, dtype=np.float32)
            y[max(0, int(np.floor(gt[0]))):min(L, int(np.ceil(gt[1])))] = 1.0
            Y.append(torch.from_numpy(y))
            used += 1
            if used % 25 == 0:
                print(f"    collected {used}/{maxn}", flush=True)
        return Xj, Xt, Y

    print(f"[collect] train (max {args.n_train})", flush=True)
    Xj_tr, Xt_tr, Y_tr = collect(train_rows, args.n_train)
    print(f"[collect] test  (max {args.n_test})", flush=True)
    Xj_te, Xt_te, Y_te = collect(test_rows, args.n_test)

    print("\n[fitting] two_tower", flush=True)
    clf_t = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_t.fit(torch.cat(Xt_tr).numpy(), torch.cat(Y_tr).numpy())
    print("[fitting] joint", flush=True)
    clf_j = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_j.fit(torch.cat(Xj_tr).numpy(), torch.cat(Y_tr).numpy())

    # rebuild GT list in the same order the test features were collected
    gts = []
    used = 0
    for r in test_rows:
        if used >= len(Xt_te):
            break
        vpath = f"{args.video_dir}/{r['vid']}.mp4"
        if not Path(vpath).exists():
            continue
        gts.append(r["relevant_windows"][0])
        used += 1

    print("\n===== LINEAR PROBE (per-second logistic regression) =====")
    for tag, clf, Xs in (("two_tower", clf_t, Xt_te), ("joint", clf_j, Xj_te)):
        ious = []
        for X, gt in zip(Xs, gts):
            p = clf.predict_proba(X.numpy())[:, 1]
            w = curve_to_window(p, args.thresh_ratio)
            ious.append(iou(w, gt))
        v = np.array(ious)
        print(f"{tag:10s} n={len(v)} IoU={v.mean():.3f} "
              f"R@0.3={np.mean(v>=0.3)*100:.1f} "
              f"R@0.5={np.mean(v>=0.5)*100:.1f} "
              f"R@0.7={np.mean(v>=0.7)*100:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/root/autodl-tmp/models/Qwen3-VL-Embedding-2B")
    ap.add_argument("--video-dir",
                    default="/root/autodl-tmp/qwen_trac/data/charades_sta/videos/Charades_v1_480")
    ap.add_argument("--labels",
                    default="/root/autodl-tmp/qwen_trac/data/uvcom_jsonl/charades_sta/charades_sta_test_tvr_format.jsonl")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--frames-per-second", default="1",
                    help="'1' = keyframe of each second; 'all' = every "
                         "decoded frame inside the second")
    ap.add_argument("--thresh-ratio", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--train-labels",
                    default="/root/autodl-tmp/qwen_trac/data/uvcom_jsonl/charades_sta/charades_sta_train_tvr_format.jsonl")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("mode", nargs="?", default="probe",
                    choices=["probe", "linprobe"])
    args = ap.parse_args()
    if args.frames_per_second != "all":
        args.frames_per_second = int(args.frames_per_second)
    if args.mode == "linprobe":
        linprobe(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()
