"""Core CSAF scoring: contrastive semantic anchors on a frozen grid.

Given per-video frame embeddings (one 1-fps grid, L2-normalised) and
L2-normalised text anchors, the visual score is

    s_vis(t) = mean_a cos(x_t, a_a) - mean_n cos(x_t, a_n)

smoothed with a centred w-point moving average and linearly interpolated
to the native frame count. Fusion with the audio branch is
    s = s_vis + LAMBDA * s_audio .

Both constants are the universal defaults of the paper: W = 5, LAMBDA = 0.1.
"""
import numpy as np

from .metrics import l2, smooth, to_native

W = 5          # smoothing width (universal default)
LAMBDA = 0.1   # audio coefficient (universal default)


def encode_texts(enc, texts, instruction, batch=32):
    """Encode anchor texts with the same frozen encoder as the frames.

    `enc` is a JointEncoder-like object exposing .emb._preprocess_inputs /
    .emb.format_model_input / .emb.model / ._pooling_last, i.e. the
    Qwen3-VL-Embedding sentence-transformers interface.
    """
    import torch
    inp = enc.emb._preprocess_inputs([
        enc.emb.format_model_input(text=t, instruction=instruction) for t in texts])
    inp = {k: (v.to(enc.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.no_grad():
        o = enc.emb.model(**inp)
        h = enc.emb._pooling_last(o.last_hidden_state, inp["attention_mask"])
    return torch.nn.functional.normalize(h, p=2, dim=-1).float().cpu().numpy()


def visual_curve(frame_emb, Ta, Tn, native, w=W):
    """Contrastive visual score for one video.

    frame_emb: (T, D) L2-normalised 1-fps frame embeddings.
    Ta, Tn:    (Ka, D), (Kn, D) L2-normalised anchor embeddings.
    Returns the w=5-smoothed curve interpolated to `native` frames.
    """
    frame_emb = l2(np.asarray(frame_emb, dtype=np.float32))
    s = (frame_emb @ Ta.T).mean(axis=1) - (frame_emb @ Tn.T).mean(axis=1)
    return np.asarray(to_native(smooth(s, w), native), dtype=np.float64)


def fuse(vis_curve, audio_curve, lam=LAMBDA):
    """Late fusion s = s_vis + lam * s_audio (audio already interpolated)."""
    return vis_curve + lam * np.asarray(audio_curve, dtype=np.float64)


def pooled_metrics(rows, curves):
    """Pooled frame-level AUC/AP over all videos in `rows` present in `curves`."""
    from .metrics import auroc, average_precision, frame_labels
    ss, yy = [], []
    for r in rows:
        v = r["vid"]
        if v not in curves:
            continue
        ss.append(curves[v])
        yy.append(frame_labels(int(r["native"]), r.get("gt") or []))
    S, Y = np.concatenate(ss), np.concatenate(yy)
    return {"auc": round(auroc(S, Y) * 100, 2),
            "ap": round(average_precision(S, Y) * 100, 2),
            "n_videos": len(ss)}
