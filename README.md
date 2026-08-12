# Fundus Vessel Segmentation & Disease Classification Dashboard

Streamlit app that takes an uploaded eye fundus image and produces:
1. A predicted vessel segmentation mask (from the UKAN model).
2. A predicted disease class out of 4: **Normal, AMD, DR, Glaucoma** (from a
   vessel-mask-guided EfficientNet-B0 classifier).

## Setup

```bash
pip install -r requirements.txt
```

(Uses CPU by default; if you have a CUDA GPU with a matching torch build,
inference will automatically use it.)

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Folder contents

- `app.py` — the Streamlit app.
- `archs.py`, `kan.py` — UKAN segmentation architecture (copied from the
  original project, unmodified).
- `classifier_model.py` — the segmentation-guided EfficientNet-B0 classifier
  architecture (copied from the original project, unmodified).
- `utils.py` — small helpers used by the architecture code.
- `checkpoints/`
  - `model_seg.pth` + `config_seg.yml` — trained UKAN segmentation checkpoint
    and its config (attention_mode=cbam_se, 512x512 input, best IoU 0.837,
    best Dice 0.911 at epoch 111).
  - `best_classifier.pth` + `classifier_config.yml` — trained classifier
    checkpoint and its config (single-view EfficientNet-B0, 4 classes,
    best val F1 0.900 at epoch 12).

These checkpoints were originally saved from a Kaggle notebook's output
folder, but got synced to disk as unpacked PyTorch zip-archive folders
(`best_checkpoint_seg/`, `best_classifier_checkpoint/`) instead of single
`.pth` files. They were rebuilt into proper `.pth` files here and verified to
load with **zero missing/unexpected keys** against the model code in this
project.

## Class label mapping

Confirmed by the project owner against `train_labels.csv`:

| Label index | Class    |
|-------------|----------|
| 0           | Normal   |
| 1           | AMD      |
| 2           | DR       |
| 3           | Glaucoma |

## Notes / caveats

- This is a research/educational tool, not a diagnostic device.
- The classifier was trained with `manual_class_weights: {"0": 2.0}`
  (Normal upweighted 2x during training) — worth knowing if you see the
  model favor or disfavor "Normal" predictions.
- Preprocessing in `app.py` mirrors the original `infer.py` / `val_classifier.py`
  scripts: images are resized to `input_h`/`input_w` from each model's config
  (512x512 for both here) and normalized with ImageNet mean/std for the
  classifier; the segmentation model uses `albumentations.Normalize()`
  defaults, which are the same ImageNet stats.
