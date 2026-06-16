# Titan — MTB Rider Detection & Attribute Classification (Prototype)

> Computer-vision pipeline for downhill MTB rider detection and multi-label attribute classification. Paused prototype.

## Problem

Downhill MTB events produce large volumes of race-day photography, but tagging riders by bike, helmet, apparel, sponsor text and competitor number is still a manual job for broadcasters, scoring crews and photo-sales workflows. The goal of this prototype was to automate that attribution so a photo set could be made searchable by rider attributes end-to-end.

## Approach

Two-stage pipeline:

1. **Detection** — Ultralytics **YOLOv11m** detects 9 object classes on the rider: `bicycle`, `bicycle_text`, `clothes_text`, `competitor_number`, `cyclist`, `cyclist_clothes`, `cyclist_with_bike`, `helmet`, `helmet_text` (boxes + polygon labels).
2. **Classification** — An **EfficientNet-B2** backbone with a multi-label head (`BCEWithLogitsLoss`, auto-computed positive-class weights, tunable inference threshold) classifies **116 attributes** per image: bike/helmet/apparel colors, bike brands (e.g. `SantaCruz`, `Trek`, `Mondraker`, `RockShox`), apparel brands (`FOX`, `LEATT`, `TroyLeeDesigns`, ...) and helmet brands.

Both stages support **ONNX export** out of the box (`src/detection/train.py --export onnx`, equivalent path on the classifier) and ship modular single-image and batch inference utilities. A Colab-oriented notebook drives the full training/eval flow end to end.

## Stack

| Component       | Tech                                       |
|-----------------|--------------------------------------------|
| Language        | Python 3.10+                               |
| Detection       | Ultralytics YOLOv11 (`yolo11m`)            |
| Classification  | EfficientNet-B2 (PyTorch / torchvision)    |
| Loss            | `BCEWithLogitsLoss` (auto `pos_weight`)    |
| Optimization    | AdamW + cosine schedule, warmup            |
| Export          | ONNX (also TorchScript / TFLite / CoreML via Ultralytics) |
| Notebooks       | Jupyter / Google Colab                     |
| Data processing | OpenCV, Pillow, pandas, NumPy              |
| Metrics         | scikit-learn (F1 macro, Hamming loss, mAP) |

## Repo layout

```
configs/
  detection_config.yaml         # YOLOv11 hyperparameters + class list
  classification_config.yaml    # EfficientNet-B2 + attribute groups
src/
  detection/                    # YOLOv11 train / validate / inference
  classification/               # multi-label dataset, model, train, inference
  utils/                        # metrics + visualization
notebooks/
  titan_training.ipynb          # Colab-oriented driver notebook
requirements.txt
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training and inference entry points (Python module form):

```bash
# Detection
python -m src.detection.train     --config configs/detection_config.yaml
python -m src.detection.validate  --config configs/detection_config.yaml
python -m src.detection.inference --model runs/detection/.../best.pt --source path/to/images

# Classification
python -m src.classification.train     --config configs/classification_config.yaml
python -m src.classification.inference --model runs/classification/.../best.pt --source path/to/images

# ONNX export (after training)
python -m src.detection.train --config configs/detection_config.yaml --export onnx
```

## Colab usage

1. Upload the two datasets to Google Drive at:
   - `MyDrive/titan_project/models_datasets/titan_detection.yolov11/`
   - `MyDrive/titan_project/models_datasets/titan_labels.multiclass/`
2. Open `notebooks/titan_training.ipynb` in Google Colab.
3. Run the cells in order — the notebook handles mounts, configs and both training loops.

## Datasets (training-time, internal)

| Stage          | Images (with augmentation) | Labels                                                                 |
|----------------|----------------------------|------------------------------------------------------------------------|
| Detection      | ~222                       | 9 classes, YOLO polygon format                                         |
| Classification | ~240                       | 116 trainable attributes, one-hot CSV (117 cols incl. filename)        |

The datasets themselves are not included in this repository.

## Status & limitations

- **Prototype. Project paused** — the downstream system this pipeline was meant to feed was cancelled.
- The labeled dataset is **small** (~222 detection / ~240 classification images, even with augmentation).
- The repo defines training targets (mAP@0.5, F1 macro, Hamming loss) but does **not** claim production-grade evaluated metrics.
- Kept public as a record of the architecture: two-stage detection + multi-label attribute classification with ONNX export.

## Why this is in the portfolio

A standalone computer-vision prototype outside my usual web/backend stack — broadens AI breadth beyond LLM work, in a non-trivial vertical (downhill MTB) that doesn't have off-the-shelf rider-attribution models. See the [portfolio entry](https://devjaes.dev/work/titan-training).

## License

MIT
