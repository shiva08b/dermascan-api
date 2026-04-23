import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

# ── Constants ─────────────────────────────────────────────────
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ACNE_CLASSES = ["cystic", "open_comedone", "closed_comedone"]

RECOMMENDATION_TAGS = {
    "cystic":          ["benzoyl_peroxide", "retinoid", "avoid_oils", "salicylic_acid"],
    "open_comedone":   ["salicylic_acid", "niacinamide", "exfoliant", "retinol"],
    "closed_comedone": ["salicylic_acid", "retinol", "non_comedogenic", "glycolic_acid"],
}

SEVERITY_THRESHOLDS = {
    "severe":   0.85,
    "moderate": 0.60,
}

# ── Load ONNX models once at startup ──────────────────────────
screener_session   = ort.InferenceSession("models/screener.onnx")
classifier_session = ort.InferenceSession("models/classifier.onnx")

def preprocess(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes → normalized numpy tensor"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)           # HWC → CHW
    arr = np.expand_dims(arr, axis=0)      # add batch dim
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

def estimate_severity(confidence: float, acne_type: str) -> str:
    if acne_type == "cystic":
        if confidence > SEVERITY_THRESHOLDS["severe"]:   return "severe"
        elif confidence > SEVERITY_THRESHOLDS["moderate"]: return "moderate"
        else: return "mild"
    else:
        if confidence > 0.90: return "moderate"
        else: return "mild"

def predict(image_bytes: bytes) -> dict:
    """
    Two-stage DermaScan inference pipeline.
    Returns structured dict ready for skincare API.
    """
    tensor = preprocess(image_bytes)
    input_name = screener_session.get_inputs()[0].name

    # ── Stage 1: Screener ──────────────────────────────────────
    screen_logits = screener_session.run(None, {input_name: tensor})[0][0]
    screen_probs  = softmax(screen_logits)
    screen_pred   = int(np.argmax(screen_probs))
    screen_conf   = float(screen_probs[screen_pred])

    if screen_pred == 0:
        return {
            "is_acne":            False,
            "acne_type":          None,
            "confidence":         round(screen_conf, 4),
            "severity":           None,
            "recommendation_tags": ["no_acne_detected"],
            "raw_scores": {
                "screener": screen_probs.tolist()
            }
        }

    # ── Stage 2: Classifier ────────────────────────────────────
    input_name    = classifier_session.get_inputs()[0].name
    cls_logits    = classifier_session.run(None, {input_name: tensor})[0][0]
    cls_probs     = softmax(cls_logits)
    cls_pred      = int(np.argmax(cls_probs))
    cls_conf      = float(cls_probs[cls_pred])
    acne_type     = ACNE_CLASSES[cls_pred]
    severity      = estimate_severity(cls_conf, acne_type)

    return {
        "is_acne":             True,
        "acne_type":           acne_type,
        "confidence":          round(cls_conf, 4),
        "severity":            severity,
        "recommendation_tags": RECOMMENDATION_TAGS[acne_type],
        "raw_scores": {
            "screener":   screen_probs.tolist(),
            "classifier": cls_probs.tolist()
        }
    }