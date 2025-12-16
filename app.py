import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import joblib
from skimage.feature import hog

# --- Configuration ---
MODEL_PATH = "model.pkl"
IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def extract_hog_features(image_array):
    features = hog(
        image_array,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"]
    )
    return features.reshape(1, -1)

def apply_color_overlay(image, color=(255, 0, 0), alpha=0.3):
    """
    Apply a semi-transparent color overlay to a grayscale PIL image.
    color: (R, G, B) — red=(255,0,0), green=(0,255,0)
    alpha: transparency (0 = invisible, 1 = opaque)
    """
    # Convert grayscale to RGB
    rgb_image = image.convert("RGB")
    # Create color overlay
    overlay = Image.new("RGB", rgb_image.size, color)
    # Blend
    highlighted = Image.blend(rgb_image, overlay, alpha)
    return highlighted

# --- UI ---
st.set_page_config(page_title="Lung Nodule Classifier", layout="centered")
st.title("🫁 Lung Nodule Classifier")
st.write("Upload a **128×128 grayscale CT patch**.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and resize
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Predict
    features = extract_hog_features(img_array)
    pred = model.predict(features)[0]
    
    # Apply overlay
    if pred == 1:
        annotated = apply_color_overlay(image, color=(255, 0, 0), alpha=0.4)  # Red tint
        label = "🔴 **Nodule Detected**"
    else:
        annotated = apply_color_overlay(image, color=(0, 255, 0), alpha=0.4)  # Green tint
        label = "🟢 **Non-Nodule**"
    
    # Display side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(annotated, caption="Highlighted", use_column_width=True)
    
    st.subheader(label)
