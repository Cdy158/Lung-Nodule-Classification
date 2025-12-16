import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw
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
    # Load and resize image
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Run prediction
    features = extract_hog_features(img_array)
    pred = model.predict(features)[0]
    
    # Prepare annotated image
    image_annotated = image.copy()
    draw = ImageDraw.Draw(image_annotated)
    
    if pred == 1:  # Nodule → add red border
        draw.rectangle([(0, 0), (127, 127)], outline="red", width=4)
        label = "🔴 **Nodule Detected**"
    else:
        draw.rectangle([(0, 0), (127, 127)], outline="green", width=4)
        label = "🟢 **Non-Nodule**"
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Patch", use_column_width=True)
    with col2:
        st.image(image_annotated, caption="Classification Result", use_column_width=True)
    
    st.subheader(label)
