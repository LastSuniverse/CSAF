# CSAF — Contrastive Semantic Anchor Fusion

**Measure, don't train.** Training-free video anomaly detection that treats
scoring as a measurement in a frozen embedding space: each frame is
contrasted with two small sets of paraphrased text anchors, and the
difference between its mean cosine to the anomalous set and to the normal
set is the score. A frozen audio encoder (CLAP-HTSAT) contributes a second
measurement, fused by one fixed coefficient. Nothing is trained, captioned,
or normalized per video.

Companion code for the paper *"Measure, Don't Train: Contrastive Semantic
Anchors for Video Anomaly Detection"* (IEEE TCSVT submission).

## Method in one paragraph

Frames sampled at 1 fps are embedded by a frozen multimodal encoder
(Qwen3-VL-Embedding-2B by default; CLIP-ViT-L/336 and BLIP-ITM-large are
provided as encoder controls). Twelve hand-written text anchors — six
anomalous, six normal, wording matched to the deployment domain — are
embedded by the same model. The per-frame score is the contrast between the
two anchor sets, smoothed with a centred `w = 5` moving average and
interpolated back to the native frame rate. On XD-Violence a frozen CLAP
audio branch is fused as `s = s_vis + 0.1 * s_audio`. Both scalars are
**universal defaults shared by both datasets**; all numbers are reported on
the full official test sets.

## Repository layout

```
csaf/
  anchors.py     all anchor sets, verbatim (UCF / XD / UNI6, audio 6+4)
  metrics.py     AUC / AP / interpolation (identical to the paper's pipeline)
  scoring.py     the contrastive scoring rule + late fusion
encoders/
  qwen_joint_encode.py   Qwen3-VL-Embedding interface (sentence-transformers)
  clip_encode.py         CLIP-ViT-L/336 image feature extraction
scripts/
  clip_control.py        CLIP encoder control (2x2 encoder x anchors)
  wording_matrix.py      wording x encoder matrix (all three encoders)
  blip_itm_control.py    BLIP-ITM scoring (discriminative interface)
  k_audio_norm_local.py  audio normal-anchor count ablation (2/4/6)
  k_audio_suite_local.py audio anomalous count / wording / lambda suite
figures/                   figure-generation scripts (matplotlib, nature palette)
data/                      recorded diagnostic outputs behind every table/figure
```

## Quick start

1. Install dependencies (`requirements.txt`): PyTorch, transformers>=5,
   numpy, scipy, matplotlib.
2. Download the frozen encoders (one-time, offline thereafter):
   - `Qwen/Qwen3-VL-Embedding-2B` (default instrument)
   - `openai/clip-vit-large-patch14-336`, `Salesforce/blip-itm-large-coco`
     (encoder controls)
   - `laion/clap-htsat` (audio branch, XD only)
3. Encode your videos at 1 fps into per-video embedding caches (see
   `encoders/` for the extraction scripts used to build the paper caches).
4. Score with `csaf.scoring.visual_curve` + `pooled_metrics`, using the
   anchors from `csaf.anchors`.

The experiment scripts under `scripts/` reproduce the paper's tables. They
embed absolute dataset paths from the authors' training cluster; adapt the
`VD` / cache path constants to your layout. `data/` holds the recorded JSON
outputs behind every reported number, so the tables and figures can be
rebuilt without re-running the encoders.

## Benchmark results (full official test sets, universal defaults)

|                | UCF-Crime AUC | XD-Violence AUC / AP |
|----------------|---------------|----------------------|
| CSAF (visual)  | 86.58         | 91.92 / 77.22        |
| CSAF (+ audio) | 86.41         | 93.21 / 80.24        |

XD-Violence leads the training-free group on both frame metrics; UCF-Crime
records the strongest training-free full-set AUC. Encoder and wording
controls are in `data/wording_matrix.json` and `data/clip_control.json`.

## Citation

```
@article{csaf2026,
  title   = {Measure, Don't Train: Contrastive Semantic Anchors for Video Anomaly Detection},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT (see LICENSE). Datasets (UCF-Crime, XD-Violence) and model weights are
NOT redistributed; obtain them from their official distributors.
