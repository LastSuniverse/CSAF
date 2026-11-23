"""CSAF: Contrastive Semantic Anchor Fusion (training-free VAD).

Measurement, not training: each frame is contrasted with two small sets
of paraphrased text anchors in a frozen embedding space; a frozen audio
encoder contributes a second measurement fused by one fixed coefficient.
"""
from .anchors import (  # noqa: F401
    UCF_ANOMALOUS, UCF_NORMAL, XD_ANOMALOUS, XD_NORMAL,
    UNI_ANOMALOUS, UNI_NORMAL, AUDIO_ANOMALOUS, AUDIO_NORMAL, INSTRUCTION,
)
from .metrics import auroc, average_precision, to_native, l2, smooth, frame_labels  # noqa: F401
from .scoring import W, LAMBDA, encode_texts, visual_curve, fuse, pooled_metrics  # noqa: F401
