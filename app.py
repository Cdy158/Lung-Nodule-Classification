import streamlit as st
import numpy as np
from PIL import Image
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

# --- UI ---
st.set_page_config(page_title="Lung Nodule Classifier", layout="centered")
st.title("🫁 Lung Nodule Classifier")
st.write("Upload a **128×128 grayscale CT patch**.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)
    
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    features = extract_hog_features(img_array)
    pred = model.predict(features)[0]  # ← Only .predict() used
    
    label = "🟢 **Non-Nodule**" if pred == 0 else "🔴 **Nodule**"
    st.subheader(f"Prediction: {label}")
