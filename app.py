# ====================================================================
# DecodeLabs Project 4 — Image or Text Recognition (Basic)
# "Building the Machine's Optic Nerve"
#
# Path chosen: Path 1 — Optical Character Recognition (OCR)
#
# Pipeline (mirrors the Project 3 "4-step" architecture):
#   STEP 1 — INGESTION      : accept raw image bytes (upload or sample)
#   STEP 2 — PRE-PROCESSING : grayscale -> Gaussian blur -> adaptive
#                              thresholding (Otsu's method)
#   STEP 3 — RECOGNITION    : pytesseract (Tesseract engine) reads the
#                              pre-processed image using a selectable
#                              Page Segmentation Mode (PSM)
#   STEP 4 — VALIDATION     : the "80% Gatekeeper" — every recognized
#                              word's confidence score is checked; only
#                              words scoring >= 80% survive into the
#                              final machine-readable output
# ====================================================================

import base64
import io
import os

import base64
import io
import os

import cv2
import numpy as np
import pytesseract
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Windows: point pytesseract to the installed tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# The "80% Threshold" from the mission brief — the absolute minimum
# standard for Project 4. Confidence below this is dropped, never
# displayed as fact.
CONFIDENCE_GATE = 80

# Supported Page Segmentation Modes (from "Tuning the PSM" playbook slide)
PSM_MODES = {
    "3": "Fully automatic (default, varied layouts)",
    "6": "Single uniform block of text (book pages)",
    "7": "Single text line (number plates / headers)",
    "11": "Sparse, scattered text (invoices)",
}

SAMPLE_FILES = {
    "sample_invoice.png": "Invoice (sparse layout)",
    "sample_sign.png": "Sign / header (single line)",
    "sample_paragraph.png": "Paragraph (uniform block)",
}


# ---------------------------------------------------------------
# STEP 2 — PRE-PROCESSING
# ---------------------------------------------------------------
def preprocess_image(bgr_image):
    """
    Systematic image pre-processing, per 'The Logic Skeleton' playbook:
      1. Grayscale Conversion  -> collapses the 3D RGB matrix into a
         1D intensity matrix, removing distracting color data.
      2. Gaussian Blur         -> smooths micro-imperfections & noise.
      3. Adaptive Thresholding -> Otsu's method forces every pixel to
         a binary decision (pure black-and-white) for clean contours.
    Returns (gray, blurred, thresholded, otsu_cutoff)
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_cutoff, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return gray, blurred, thresholded, otsu_cutoff


# ---------------------------------------------------------------
# STEP 3 — RECOGNITION  +  STEP 4 — VALIDATION (80% Gate)
# ---------------------------------------------------------------
def run_ocr_pipeline(thresholded_image, psm_mode):
    """
    Runs pytesseract on the pre-processed binary image and applies the
    80% confidence gatekeeper to every detected word:

        if confidence >= 0.80:
            draw_box_and_label()
        else:
            drop_detection()
    """
    config = f"--oem 3 --psm {psm_mode}"
    raw_data = pytesseract.image_to_data(
        thresholded_image, config=config, output_type=pytesseract.Output.DICT
    )

    words = []
    kept_text_parts = []
    kept_confidences = []

    n = len(raw_data["text"])
    for i in range(n):
        text = raw_data["text"][i].strip()
        conf_raw = raw_data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0

        if text == "" or conf < 0:
            continue

        kept = conf >= CONFIDENCE_GATE
        words.append({
            "text": text,
            "confidence": round(conf, 1),
            "kept": kept,
            "bbox": {
                "x": int(raw_data["left"][i]),
                "y": int(raw_data["top"][i]),
                "w": int(raw_data["width"][i]),
                "h": int(raw_data["height"][i]),
            },
        })
        if kept:
            kept_text_parts.append(text)
            kept_confidences.append(conf)

    avg_confidence = (
        round(sum(kept_confidences) / len(kept_confidences), 1)
        if kept_confidences else 0.0
    )
    final_text = " ".join(kept_text_parts)

    return {
        "words": words,
        "final_text": final_text,
        "avg_confidence": avg_confidence,
        "total_words_detected": len(words),
        "words_kept": len(kept_confidences),
        "gate_passed": avg_confidence >= CONFIDENCE_GATE and len(kept_confidences) > 0,
    }


def _encode_png(np_image_gray_or_bgr):
    ok, buf = cv2.imencode(".png", np_image_gray_or_bgr)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_boxes(bgr_image, words):
    """Draws green boxes for kept words, faint red for dropped ones —
    the 'Visual Confirmation' deliverable."""
    canvas = bgr_image.copy()
    for w in words:
        b = w["bbox"]
        pt1 = (b["x"], b["y"])
        pt2 = (b["x"] + b["w"], b["y"] + b["h"])
        color = (46, 125, 91) if w["kept"] else (43, 90, 232)  # BGR: green / soft red-orange
        thickness = 2 if w["kept"] else 1
        cv2.rectangle(canvas, pt1, pt2, color, thickness)
    return canvas


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples")
def api_samples():
    return jsonify({
        "samples": [{"file": f, "label": lbl} for f, lbl in SAMPLE_FILES.items()],
        "psm_modes": PSM_MODES,
        "confidence_gate": CONFIDENCE_GATE,
    })


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    psm_mode = request.form.get("psm", "3")
    if psm_mode not in PSM_MODES:
        psm_mode = "3"

    # STEP 1 — INGESTION
    if "image" in request.files and request.files["image"].filename:
        file_bytes = request.files["image"].read()
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    else:
        sample_name = request.form.get("sample")
        if sample_name not in SAMPLE_FILES:
            return jsonify({"error": "No image uploaded and no valid sample selected."}), 400
        pil_img = Image.open(os.path.join(DATA_DIR, sample_name)).convert("RGB")

    bgr_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # STEP 2 — PRE-PROCESSING
    gray, blurred, thresholded, otsu_cutoff = preprocess_image(bgr_image)

    # STEP 3 + 4 — RECOGNITION + VALIDATION
    result = run_ocr_pipeline(thresholded, psm_mode)

    annotated = _draw_boxes(bgr_image, result["words"])

    response = {
        "psm_mode": psm_mode,
        "psm_description": PSM_MODES[psm_mode],
        "confidence_gate": CONFIDENCE_GATE,
        "otsu_cutoff": round(float(otsu_cutoff), 1),
        "final_text": result["final_text"],
        "avg_confidence": result["avg_confidence"],
        "total_words_detected": result["total_words_detected"],
        "words_kept": result["words_kept"],
        "gate_passed": result["gate_passed"],
        "words": result["words"],
        "images": {
            "original": _encode_png(bgr_image),
            "grayscale": _encode_png(gray),
            "thresholded": _encode_png(thresholded),
            "annotated": _encode_png(annotated),
        },
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
