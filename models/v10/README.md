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

# Hausa OCR Ajami — trained V10

PyTorch CNN + two-layer bidirectional LSTM (256 hidden units per direction) with CTC, trained on images of Hausa Ajami text to predict Latin-script transcriptions. This repository contains the **trained checkpoint**, not just the training code.

## Checkpoint and measured results

`best_cer_v10.pt` was selected at **epoch 25** by minimum validation CER. It includes the model weights, vocabulary, preprocessing parameters and optimizer state from the local training run.

| Metric | Value |
|---|---:|
| Test CER | 0.3007341374 |
| Test WER | 0.7518518519 |
| Test exact-line rate | 0.2976588629 |
| Test lines | 299 |

These metrics were measured in the completed local run. Full precision is available in `test_metrics.json`; per-epoch validation results are in `history_v10.csv`.

## Training data and protocol

Source dataset: [IntelligenceResearchLab/Hausa](https://huggingface.co/datasets/IntelligenceResearchLab/Hausa). The local snapshot contained 3,199 lines, redistributed into **2,800 training / 100 validation / 299 test** lines with seed 42. This is a custom line-level split, not the original official test split or a manuscript-held-out evaluation. Scores are not directly comparable with earlier V7/V9 experiments using different splits.

Each training line had eight fixed views, producing 22,400 examples per epoch. Dynamic augmentation was disabled. Input height was 96 pixels, with aspect ratio preserved, grayscale conversion, horizontal mirroring for right-to-left input, and pixel normalization to [-1, 1]. Batch size was 24, using AdamW and a validation-driven learning-rate scheduler. The run completed 25 epochs. Labels use NFC normalization.

The dataset's attribution and usage conditions remain applicable; consult its dataset card. The repository's pre-existing license declaration is retained.

## Run inference

Download this repository's files and install PyTorch and Pillow in your environment. The checkpoint was trained with PyTorch 2.5.1 + CUDA 12.1 and Pillow 12.3.0.

```bash
python predict.py path/to/line_image.png
```

`model.py` contains the network definition. `predict.py` loads the trained weights and performs greedy CTC decoding on one text-line image. This is a custom PyTorch model, not a Transformers `AutoModel` checkpoint or a hosted inference endpoint. It expects pre-segmented lines, not complete pages.

The single-image loading/inference path was checked before publication; this does not repeat the full benchmark. Batched padding can influence the bidirectional network, so single-image output may differ from batched evaluation.

## Files

- `best_cer_v10.pt`: trained checkpoint.
- `vocab.json`: character-to-index mapping, including blank and unknown symbols.
- `training_config.json`: training configuration and checkpoint validation scores.
- `test_metrics.json`: final measured test metrics.
- `history_v10.csv`: per-epoch training and validation history.
- `checkpoint_sha256.txt`: checksum of the uploaded checkpoint.
- `model.py`, `predict.py`: local inference code.

The word error rate remains high; outputs require review. No retraining was performed as part of publication.
