import streamlit as st
import numpy as np
from PIL import Image  # ← Only image library you need
from skimage.feature import hog
import joblib

# --- Configuration ---
MODEL_PATH = "model.pkl"
IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- HOG feature extraction ---
def extract_hog_features(image_array):
    """Extract HOG from a 128x128 grayscale numpy array."""
    features = hog(
        image_array,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"]
    )
    return features.reshape(1, -1)

# --- App UI ---
st.set_page_config(page_title="Lung Nodule Classifier", layout="centered")
st.title("🫁 Lung Nodule Classifier (HOG + SVM)")
st.write("Upload a **128×128 grayscale CT patch** to classify it as **Nodule** or **Non-Nodule**.")
st.info("ℹ️ Works best with LUNA16-style patches.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image with PIL → convert to grayscale
    image = Image.open(uploaded_file).convert("L")  # 'L' = grayscale
    st.image(image, caption="Uploaded Image", width=200)

    # Resize to 128x128 (use high-quality resampling)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)  # Shape: (128, 128)

    # Extract features and predict
    features = extract_hog_features(img_array)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = np.max(proba)

    # Display result
    label = "🟢 **Non-Nodule**" if pred == 0 else "🔴 **Nodule**"
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")
