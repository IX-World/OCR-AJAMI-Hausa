---
license: cc-by-sa-4.0
language:
- ha
tags:
- ocr
- ajami
- pytorch
- crnn
---

# Hausa OCR Ajami — V10, split 92/3/5

This repository contains the **trained checkpoint** from `train_hausa_ctc_v10_nouveau_92_3_5.py`, published on 13 September 2026. It replaces the previous V10 checkpoint as the current version; the previous files remain in the repository history.

The model reads pre-segmented Hausa Ajami line images and predicts Latin-script transcriptions using a CNN, a two-layer bidirectional LSTM (256 hidden units per direction), and CTC with greedy decoding.

## Checkpoint and verified results

`best_cer_v10.pt` is the checkpoint selected at **epoch 25** by minimum validation CER.

| Metric | Validation | Test |
|---|---:|---:|
| CER | 0.2473841555 | 0.2549638989 |
| WER | 0.6258741259 | 0.6963123644 |
| Exact-line rate | 0.2916666667 | 0.275 |
| Lines | 96 | 160 |

The checkpoint was reloaded and re-evaluated on its saved 160-line test split before publication. These results match the completed local training run's reported scores (CER 0.2550, WER 0.6963). No retraining was performed during publication. Full-precision test metrics and predictions are included.

## Training data and protocol

Dataset: [IntelligenceResearchLab/Hausa](https://huggingface.co/datasets/IntelligenceResearchLab/Hausa), using the local `Hausa_repo_nouveau` snapshot. The saved **92/3/5 split** contains 3,198 examples: **2,942 training / 96 validation / 160 test**, with seed 42. See `split.json` for exact assignments. This is a line-level split, not a manuscript-held-out benchmark. These scores are not directly comparable with the previous V10 release, which used a different 299-line test set.

Each training line has eight fixed views, for 23,536 examples per epoch. Dynamic augmentation is disabled. Training used 25 epochs, batch size 24, AdamW (initial LR 0.001, weight decay 0.0001), and ReduceLROnPlateau. Input preprocessing is grayscale, height 96 with aspect ratio preserved, horizontal mirroring for RTL input, and normalization to [-1, 1]. Labels use NFC normalization. The matching vocabulary has 81 entries, including CTC blank; it has no `<UNK>` entry.

The dataset's attribution and usage conditions remain applicable; consult its dataset card. The repository's pre-existing license declaration is retained.

## Run inference

Download the repository files and install PyTorch and Pillow. The checkpoint was validated with PyTorch 2.5.1 + CUDA 12.1 and Pillow 12.3.0.

```bash
python predict.py path/to/line_image.png
```

`predict.py` loads `best_cer_v10.pt` and the network in `model.py`, using the vocabulary stored in the checkpoint. This is a custom PyTorch model, not a Transformers `AutoModel` checkpoint or a hosted inference endpoint. It expects segmented line images, not complete pages. Batched padding can influence the bidirectional network; single-image predictions can differ from batched evaluation.

## Files

- `best_cer_v10.pt`: trained checkpoint, including model weights and optimizer state.
- `vocab.json`: matching character-to-index mapping.
- `training_config.json`: checkpoint metadata and training settings.
- `split.json`: saved split used by this training run.
- `training_source.py`: original training script; it retains the original local paths and should be configured before running elsewhere.
- `history_v10.csv`: per-epoch training and validation history.
- `test_metrics.json`, `test_predictions.csv`: verified results on the saved test split.
- `checkpoint_sha256.txt`: checksum of the current checkpoint.
- `model.py`, `predict.py`: local inference code.

The word error rate remains high; outputs require review. The experiment does not establish generalization to unseen manuscripts or writers.
