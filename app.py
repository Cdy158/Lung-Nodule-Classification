import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
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

# --- UI ---
st.set_page_config(page_title="Lung Nodule Classifier", layout="centered")
st.title("🫁 Lung Nodule Classifier")
st.write("Upload a **128×128 grayscale CT patch** (nodule expected near center).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and resize
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Predict
    features = extract_hog_features(img_array)
    pred = model.predict(features)[0]
    
    # Create annotated image
    image_annotated = image.convert("RGB")
    draw = ImageDraw.Draw(image_annotated)
    
    if pred == 1:  # Nodule → mark center
        center = (64, 64)  # center of 128x128
        radius = 10
        # Draw red circle
        draw.ellipse(
            (center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius),
            outline="red",
            width=2
        )
        # Draw red crosshair (optional)
        draw.line([(64, 54), (64, 74)], fill="red", width=2)  # vertical
        draw.line([(54, 64), (74, 64)], fill="red", width=2)  # horizontal
        
        label = "🔴 **Nodule Detected** (center assumed)"
    else:
        label = "🟢 **Non-Nodule**"
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(image_annotated, caption="Prediction", use_column_width=True)
    
    st.subheader(label)
