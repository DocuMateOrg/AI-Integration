import io
import os
import tempfile
import base64
import json
import re
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
import numpy as np
import cv2
import requests

# ---------------- Load .env first ---------------- #
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "").strip()
PORT = int(os.getenv("PORT", 5001))


if not GEMINI_API_KEY or not GEMINI_API_URL:
    raise ValueError("GEMINI_API_KEY or GEMINI_API_URL not set in .env")

# ---------------- Initialize Flask ---------------- #
app = Flask(__name__)

# ---------------- Image Preprocessing Helpers ---------------- #
def read_image_from_bytes(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def write_image_to_bytes(img):
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes()

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def denoise_and_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
    return denoised

def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def preprocess_image_bytes(data: bytes, do_binarize=True):
    img = read_image_from_bytes(data)
    img = deskew_image(img)
    img = denoise_and_clahe(img)
    if do_binarize:
        img = binarize(img)
    return write_image_to_bytes(img)

# ---------------- Routes ---------------- #
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "ai_service (preprocess+analyze)",
        "routes": ["/preprocess (POST)", "/analyze (POST)"],
        "gemini_config_present": bool(GEMINI_API_KEY and GEMINI_API_URL)
    })

@app.route("/preprocess", methods=["POST"])
def preprocess_endpoint():
    if "file" in request.files:
        data = request.files["file"].read()
    else:
        payload = request.get_json() or {}
        if "file_b64" in payload:
            try:
                data = base64.b64decode(payload["file_b64"])
            except Exception as e:
                return jsonify({"error": "invalid base64", "details": str(e)}), 400
        else:
            return jsonify({"error": "no file provided"}), 400

    try:
        cleaned_bytes = preprocess_image_bytes(data)
    except Exception as e:
        return jsonify({"error": "preprocessing_failed", "details": str(e)}), 500

    cleaned_b64 = base64.b64encode(cleaned_bytes).decode("utf-8")
    return jsonify({"cleaned_image_b64": cleaned_b64})

@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    # Validate API config
    if not GEMINI_API_URL or not GEMINI_API_KEY:
        return jsonify({"error": "missing_gemini_config",
                        "details": "Set GEMINI_API_KEY and GEMINI_API_URL in .env"}), 500

    # Load input bytes
    if "file" in request.files:
        data = request.files["file"].read()
    else:
        payload = request.get_json() or {}
        if "cleaned_file_b64" in payload:
            try:
                data = base64.b64decode(payload["cleaned_file_b64"])
            except Exception as e:
                return jsonify({"error": "invalid base64", "details": str(e)}), 400
        elif "file_url" in payload:
            resp = requests.get(payload["file_url"])
            if resp.status_code != 200:
                return jsonify({"error": "failed_to_download_file_url",
                                "status_code": resp.status_code}), 400
            data = resp.content
        else:
            return jsonify({"error": "no file provided"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        img_b64 = base64.b64encode(data).decode("utf-8")

        prompt_instructions = (
            "You are an OCR+NLP assistant. Given the image, extract the full clean text in "
            "Sinhala and/or English. Output JSON with fields: text (UTF-8), language ('si' or 'en' or 'mixed'), "
            "confidence (0-1), tags (array of keywords), summary (1-3 short sentences). "
            "Return valid JSON only."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_instructions},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64
                            }
                        }
                    ]
                }
            ]
        }

        headers = {"Content-Type": "application/json"}

        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=120)

        if resp.status_code != 200:
            return jsonify({"error": "gemini_call_failed",
                            "status_code": resp.status_code,
                            "details": resp.text}), 500

        try:
            result = resp.json()
        except Exception as e:
            return jsonify({"error": "invalid_gemini_json",
                            "details": str(e),
                            "raw": resp.text}), 500

        generated_text = (
            result.get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text", "")
        )

        if not generated_text:
            return jsonify({"error": "empty_model_output", "raw": result}), 500

        clean_text = re.sub(r"```json|```", "", generated_text).strip()

        try:
            model_json = json.loads(clean_text)
        except Exception as e:
            return jsonify({
                "error": "failed_to_load_model_json",
                "details": str(e),
                "cleaned_text": clean_text,
                "raw_api_response": result
            }), 500

        return jsonify(model_json), 200

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ---------------- Run Flask ---------------- #
if __name__ == "__main__":
    print("GEMINI_API_KEY:", repr(GEMINI_API_KEY))
    print("GEMINI_API_URL:", repr(GEMINI_API_URL))

    print("GEMINI_API_URL set:", bool(GEMINI_API_URL))
    print("GEMINI_API_KEY set:", bool(GEMINI_API_KEY))
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
