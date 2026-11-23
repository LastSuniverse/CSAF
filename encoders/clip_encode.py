"""Extract CLIP image features for VAD frame dirs, mirroring the Qwen cache layout.

Writes one npy per video: <out>/<split>/<vid>.npy  (T, D) float32.
Frames sampled identically to vad_encode.py (uniform stride to --max-frames),
so CLIP and Qwen caches are frame-aligned for the swap experiment.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=400)
    args = ap.parse_args()

    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    device = "cuda"
    proc = CLIPImageProcessor.from_pretrained(args.model)
    vis = CLIPVisionModelWithProjection.from_pretrained(
        args.model, dtype=torch.float16).to(device).eval()
    dim = vis.config.projection_dim
    print(json.dumps({"model": args.model, "dim": dim}), flush=True)

    rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_root = Path(args.out) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    pool = ThreadPoolExecutor(max_workers=8)

    def load(names):
        return list(pool.map(lambda q: Image.open(q).convert("RGB"), names))

    done = 0
    for r in rows:
        dst = out_root / f"{r['vid']}.npy"
        if dst.exists():
            done += 1
            continue
        d = Path(args.frames_root) / str(r["vid"])
        names = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
        stride = max(1, -(-len(names) // args.max_frames))
        imgs = load(names[::stride])
        feats = []
        for i in range(0, len(imgs), args.batch):
            px = proc(images=imgs[i:i + args.batch], return_tensors="pt").to(device)
            with torch.no_grad():
                f = vis(**px).image_embeds
            feats.append(f.float().cpu())
        np.save(dst, torch.cat(feats).numpy().astype(np.float32))
        done += 1
        if done % 25 == 0:
            print(f"progress {done}/{len(rows)}", flush=True)
    print("CLIP_ENCODE_DONE", flush=True)


if __name__ == "__main__":
    main()
