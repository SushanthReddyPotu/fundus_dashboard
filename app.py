"""
Streamlit dashboard: eye fundus image -> vessel segmentation mask + disease
classification (Normal / AMD / DR / Glaucoma).

Pipeline:
    1. UKAN segmentation model produces a vessel probability map from the raw
       fundus image.
    2. That vessel mask is fed together with the raw image into a
       segmentation-guided EfficientNet-B0 classifier, which predicts one of
       4 disease classes.

Run with:
    streamlit run app.py
"""
import os
import sys
import io

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archs
from classifier_model import SegGuidedEfficientNetB0, DualInputSegGuidedEfficientNet

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(APP_DIR, "checkpoints")
SEG_CONFIG_PATH = os.path.join(CKPT_DIR, "config_seg.yml")
SEG_CKPT_PATH = os.path.join(CKPT_DIR, "model_seg.pth")
CLS_CONFIG_PATH = os.path.join(CKPT_DIR, "classifier_config.yml")
CLS_CKPT_PATH = os.path.join(CKPT_DIR, "best_classifier.pth")

# Label mapping confirmed by the project owner (matches train_labels.csv).
CLASS_NAMES = {0: "Normal", 1: "AMD", 2: "DR", 3: "Glaucoma"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format.")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}


@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Segmentation model (UKAN) ---
    seg_config = load_yaml(SEG_CONFIG_PATH)
    seg_model = archs.__dict__[seg_config["arch"]](
        seg_config["num_classes"],
        seg_config["input_channels"],
        seg_config["deep_supervision"],
        embed_dims=seg_config["input_list"],
        no_kan=seg_config.get("no_kan", False),
        attention_mode=seg_config.get("attention_mode", "none"),
    )
    seg_ckpt = torch_load_compat(SEG_CKPT_PATH)
    seg_state = extract_state_dict(seg_ckpt)
    incompatible = seg_model.load_state_dict(seg_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Segmentation checkpoint does not match model config.\n"
            f"Missing: {incompatible.missing_keys}\nUnexpected: {incompatible.unexpected_keys}"
        )
    seg_model.to(device).eval()

    # --- Classifier model ---
    cls_config = load_yaml(CLS_CONFIG_PATH)
    model_type = cls_config.get("model_type", "single")
    num_classes = cls_config["num_classes"]
    dropout = cls_config.get("dropout", 0.3)
    if model_type == "dual_input":
        cls_model = DualInputSegGuidedEfficientNet(
            num_classes=num_classes, pretrained=False, dropout=dropout
        )
    else:
        cls_model = SegGuidedEfficientNetB0(num_classes=num_classes, pretrained=False)
    cls_ckpt = torch_load_compat(CLS_CKPT_PATH)
    cls_state = extract_state_dict(cls_ckpt)
    incompatible = cls_model.load_state_dict(cls_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Classifier checkpoint does not match model config.\n"
            f"Missing: {incompatible.missing_keys}\nUnexpected: {incompatible.unexpected_keys}"
        )
    cls_model.to(device).eval()

    return {
        "device": device,
        "seg_model": seg_model,
        "seg_config": seg_config,
        "cls_model": cls_model,
        "cls_config": cls_config,
    }


def run_segmentation(seg_model, seg_config, device, image_rgb):
    """image_rgb: HxWx3 uint8 RGB array. Returns prob_map (float32, HxW, original size, 0..1)."""
    h0, w0 = image_rgb.shape[:2]
    input_h = seg_config["input_h"]
    input_w = seg_config["input_w"]

    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = seg_model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # HxW at model resolution

    prob_full = cv2.resize(probs, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return prob_full


def run_classification(cls_model, cls_config, device, image_rgb, mask_prob):
    input_h = cls_config.get("input_h", 512)
    input_w = cls_config.get("input_w", 512)

    img_resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    mask_resized = cv2.resize(mask_prob, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = cls_model(img_tensor, mask_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


def overlay_mask(image_rgb, prob_map, threshold=0.5, alpha=0.45):
    binary = (prob_map >= threshold).astype(np.uint8)
    overlay = image_rgb.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255  # red channel
    mask3 = np.repeat(binary[..., None], 3, axis=2).astype(bool)
    blended = np.where(mask3, (overlay * (1 - alpha) + red * alpha).astype(np.uint8), overlay)
    return blended


def main():
    st.set_page_config(page_title="Fundus Vessel Segmentation & Disease Classifier", layout="wide")
    st.title("Eye Fundus Image Analysis")
    st.caption(
        "Upload a fundus image to get a predicted vessel segmentation mask and a "
        "disease classification across 4 classes: Normal, AMD, DR, Glaucoma."
    )

    with st.spinner("Loading models..."):
        try:
            models = load_models()
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.stop()

    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Vessel mask threshold", 0.0, 1.0, 0.5, 0.05)
    overlay_alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    device_name = "GPU" if models["device"].type == "cuda" else "CPU"
    st.sidebar.caption(f"Running inference on: {device_name}")

    uploaded_file = st.file_uploader(
        "Upload a fundus image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    if uploaded_file is None:
        st.info("Upload an image to run the pipeline.")
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read this file as an image. Please upload a valid image file.")
        return
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("Running vessel segmentation..."):
        prob_map = run_segmentation(
            models["seg_model"], models["seg_config"], models["device"], image_rgb
        )

    with st.spinner("Running disease classification..."):
        class_probs = run_classification(
            models["cls_model"], models["cls_config"], models["device"], image_rgb, prob_map
        )

    pred_idx = int(np.argmax(class_probs))
    pred_name = CLASS_NAMES.get(pred_idx, f"Class {pred_idx}")
    pred_conf = float(class_probs[pred_idx])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Vessel Mask")
        prob_display = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
        st.image(prob_display, use_container_width=True, clamp=True)
    with col3:
        st.subheader("Overlay")
        overlay_img = overlay_mask(image_rgb, prob_map, threshold=threshold, alpha=overlay_alpha)
        st.image(overlay_img, use_container_width=True)

    st.divider()
    st.subheader("Predicted Diagnosis")
    st.metric(label="Predicted class", value=pred_name, delta=f"{pred_conf*100:.1f}% confidence")

    st.write("Class probabilities:")
    prob_rows = {CLASS_NAMES.get(i, f"Class {i}"): float(p) for i, p in enumerate(class_probs)}
    st.bar_chart(prob_rows)

    with st.expander("Raw probability values"):
        for name, p in sorted(prob_rows.items(), key=lambda x: -x[1]):
            st.write(f"{name}: {p:.4f}")

    st.divider()
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    st.download_button(
        "Download binary vessel mask (PNG)",
        data=buf.getvalue(),
        file_name="vessel_mask.png",
        mime="image/png",
    )

    st.caption(
        "Note: this tool is for research/educational purposes only and is not a "
        "substitute for professional medical diagnosis."
    )


if __name__ == "__main__":
    main()
