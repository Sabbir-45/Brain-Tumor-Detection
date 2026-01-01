import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

import gdown  # Google Drive download

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = "uploads"
MODEL_WEIGHTS_PATH = "models/vgg19_finetuned_best.h5"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Google Drive FILE_ID (from your link)
GDRIVE_FILE_ID = "1ac_C4G6zsmkc5kA4yPLPpcZEZlHUaJb1"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# Download model if not exists
# =========================
if not os.path.exists(MODEL_WEIGHTS_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_WEIGHTS_PATH, quiet=False)

# =========================
# HELPERS
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_img(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

# =========================
# BUILD VGG19 MODEL
# =========================
def build_clean_vgg19():
    base = VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base.trainable = False
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# =========================
# LOAD MODEL WEIGHTS
# =========================
try:
    tumor_model = build_clean_vgg19()
    tumor_model.load_weights(MODEL_WEIGHTS_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ ERROR loading weights:", e)
    tumor_model = None

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if tumor_model is None:
        return "❌ MODEL FAILED TO LOAD. CHECK YOUR WEIGHTS FILE."

    if "file" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Invalid file or no file selected.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # =========================
    # MRI VALIDATION
    # =========================
    img = tf.keras.utils.load_img(path)
    arr = tf.keras.utils.img_to_array(img).astype("float32")

    # Grayscale check
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    grayscale_distance = (np.mean(np.abs(R - G)) +
                          np.mean(np.abs(G - B)) +
                          np.mean(np.abs(R - B))) / 3
    if grayscale_distance > 7:
        return render_template("result.html",
                               result="❌ Invalid Image",
                               score=0, image_file=filename)

    # Aspect ratio check
    w, h = img.size
    if not (0.75 <= w / h <= 1.35):
        return render_template("result.html",
                               result="❌ Invalid MRI",
                               score=0, image_file=filename)

    # Edge density check
    gray = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.008 or edge_density > 0.12:
        return render_template("result.html",
                               result="❌ Invalid MRI.",
                               score=0, image_file=filename)

    # Circle detection (brain curvature)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                               param1=50, param2=28, minRadius=60, maxRadius=260)
    if circles is None:
        return render_template("result.html",
                               result="❌ Invalid MRI — brain structure not detected.",
                               score=0, image_file=filename)

    # =========================
    # TUMOR PREDICTION
    # =========================
    img_pp = preprocess_img(path)
    prob = float(tumor_model.predict(img_pp, verbose=0)[0][0])
    status = "🧠 Tumor Detected" if prob >= 0.5 else "🧠 No Tumor Detected"

    return render_template("result.html",
                           result=status,
                           score=round(prob*100, 2),
                           image_file=filename)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/health")
def health():
    return {"status": True}

# =========================
if __name__ == "__main__":
    # For Render deploy, debug=False in production
    app.run(debug=True)
