import base64
import io
import time
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
from flask import Blueprint, jsonify, render_template, request
import tensorflow as tf

api_bp = Blueprint("api", __name__)
_model = None


def load_digit_model(model_path: str) -> None:
    global _model
    _model = tf.keras.models.load_model(model_path)


def preprocess_digit_image(pil_image: Image.Image, invert: bool = True, is_upload: bool = False) -> tuple[np.ndarray, np.ndarray]:
    gray = np.array(pil_image.convert("L"), dtype=np.uint8)

    # Fast blank / uniform check (contrast dynamic range < 15)
    if int(np.max(gray)) - int(np.min(gray)) < 15:
        empty = np.zeros((28, 28), dtype=np.float32)
        return empty.reshape(1, 28, 28, 1), empty

    if invert:
        gray = 255 - gray

    if is_upload:
        # For uploaded photos/scans:
        # 1. Light Gaussian blur suppresses paper grain and sensor noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 2. Otsu thresholding dynamically separates ink strokes from paper shadows
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 3. Morphological dilation thickens ballpoint/thin pen strokes to match MNIST brush widths
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
    else:
        # For canvas: exact 25 threshold to remove soft brush edge artifacts
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_TOZERO)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        empty = np.zeros((28, 28), dtype=np.float32)
        return empty.reshape(1, 28, 28, 1), empty

    x, y, w, h = cv2.boundingRect(coords)
    roi = thresh[y:y + h, x:x + w]

    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    cy, cx = ndimage.center_of_mass(canvas)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = int(round(14.0 - cy))
        shift_x = int(round(14.0 - cx))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))

    normalized = np.clip(canvas / 255.0, 0.0, 1.0).astype(np.float32)
    return normalized.reshape(1, 28, 28, 1), normalized


@api_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@api_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None}), 200


@api_bp.route("/api/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "CNN model not loaded"}), 503

    start_time = time.perf_counter()
    invert = True
    is_upload = False

    if "file" in request.files and request.files["file"].filename:
        file_storage = request.files["file"]
        pil_img = Image.open(file_storage.stream)
        invert_flag = request.form.get("invert", "auto")
        if invert_flag == "auto":
            mean_val = float(np.mean(np.array(pil_img.convert("L"))))
            invert = mean_val > 127.0
        else:
            invert = invert_flag.lower() == "true"
        is_upload = True
    elif request.is_json and "image" in request.json:
        data_url = request.json["image"]
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode == "RGBA":
            bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg.convert("L")
        invert = True
        is_upload = False
    else:
        return jsonify({"error": "No image provided"}), 400

    tensor, _ = preprocess_digit_image(pil_img, invert=invert, is_upload=is_upload)
    if float(np.max(tensor)) == 0.0:
        return jsonify({"error": "Blank image detected. Please draw or upload a digit."}), 400

    probs = _model.predict(tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit])
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return jsonify({
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100.0, 2),
        "probabilities": [round(float(p) * 100.0, 2) for p in probs],
        "inference_time_ms": latency_ms
    }), 200
