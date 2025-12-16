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
st.write("Upload a **128×128 grayscale CT patch** for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    
    # Resize to 128x128 (important!)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Extract HOG features (just like your code)
    features = extract_hog_features(img_array)
    
    # Predict
    pred = model.predict(features)[0]
    
    # Display result
    if pred == 1:
        st.markdown("### Prediction: 🔴 **Nodule**")
    else:
        st.markdown("### Prediction: 🟢 **Non-Nodule**")
    
    # Show image
    st.image(image, caption="Uploaded Patch (128x128)", use_column_width=True)
