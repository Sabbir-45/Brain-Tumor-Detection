import os
import cv2
import gdown
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image

# =========================
# CONFIG
# =========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "vgg19_finetuned_best.h5")

GDRIVE_FILE_ID = "1ac_C4G6zsmkc5kA4yPLPpcZEZlHUaJb1"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# DOWNLOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# =========================
# BUILD MODEL
# =========================
def build_vgg19():
    base = VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.load_weights(MODEL_PATH)
    return model

model = build_vgg19()

# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_img(img):
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image_input):
    if image_input is None:
        return "❌ No image uploaded"

    arr = image_input.astype("float32")

    # Grayscale check
    R, G, B = arr[...,0], arr[...,1], arr[...,2]
    gray_dist = (np.mean(np.abs(R-G)) +
                 np.mean(np.abs(G-B)) +
                 np.mean(np.abs(R-B))) / 3

    if gray_dist > 7:
        return "❌ Invalid Image (Not MRI)"

    # Edge check
    gray = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    if edge_density < 0.008 or edge_density > 0.12:
        return "❌ Invalid MRI"

    # Prediction
    img_pp = preprocess_img(arr)
    prob = float(model.predict(img_pp, verbose=0)[0][0])

    if prob >= 0.5:
        return f"🧠 Tumor Detected ({prob*100:.2f}%)"
    else:
        return f"🧠 No Tumor Detected ({(1-prob)*100:.2f}%)"

# =========================
# GRADIO UI
# =========================
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Brain MRI"),
    outputs=gr.Textbox(label="Result"),
    title="🧠 Brain Tumor Detection using VGG19",
    description="Upload an MRI image to detect brain tumor using deep learning"
)

interface.launch()
