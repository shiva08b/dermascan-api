"""
vit_inference.py
─────────────────────────────────────────────────────────────────
Optional inference module using google/vit-base-patch16-224.

The ViT is a general ImageNet classifier and has no native concept
of "acne". We use two heuristics to produce a binary acne/no-acne
output suitable for demo and multi-model comparison:

  1. Keyword match – if any of the top-5 predicted ImageNet labels
     contain skin-adjacent words (e.g. "acne", "skin", "pore",
     "lesion", "tick"), we flag the image as possible acne.

  2. Low-confidence heuristic – close-up photos of human skin do
     not map well to standard ImageNet categories. A top-1
     confidence < 0.30 indicates the model is uncertain, which
     is a soft signal that the image may be a skin close-up.

This module is intentionally isolated: it does NOT touch or import
anything from inference.py (the EfficientNet pipeline).
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

from PIL import Image

logger = logging.getLogger("vit_inference")

# ── Lazy-load (models download on first call, cached after that) ─
@lru_cache(maxsize=1)
def _get_model():
    """Download / cache ViT processor + model on first invocation."""
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        import torch  # noqa: F401  — presence check

        logger.info("Loading google/vit-base-patch16-224 …")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        model.eval()
        logger.info("ViT model ready.")
        return processor, model
    except ImportError:
        raise RuntimeError(
            "ViT requires torch and transformers. "
            "Run: pip install transformers torch torchvision"
        )
    except Exception as exc:
        logger.error("Failed to load ViT model: %s", exc)
        raise


# ── Label keywords that suggest a skin / dermatological subject ──
_SKIN_KEYWORDS = {
    "acne", "skin", "pore", "lesion", "tick", "mite",
    "rash", "blotch", "freckle", "pimple", "face",
    "eczema", "derma", "boil", "pustule",
}


def _has_skin_label(labels: list[str]) -> bool:
    for label in labels:
        for kw in _SKIN_KEYWORDS:
            if kw in label.lower():
                return True
    return False


# ── Public inference function ────────────────────────────────────

def predict_vit(image_bytes: bytes) -> dict:
    """
    Run ViT binary classification on raw image bytes.

    Returns a dict that is structurally compatible with the main
    pipeline's output so the API can merge / compare them easily.
    The `acne_type` and `severity` fields are always None because
    the ViT only supports binary detection in this pipeline.
    """
    import torch

    processor, model = _get_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    top_probs, top_ids = torch.topk(probs, 5)
    top_probs_list: list[float] = top_probs.tolist()
    top_labels: list[str] = [
        model.config.id2label[idx.item()] for idx in top_ids
    ]

    top1_conf = top_probs_list[0]
    skin_label_match = _has_skin_label(top_labels)

    # Binary decision
    is_acne = skin_label_match or (top1_conf < 0.30)

    return {
        "is_acne":             is_acne,
        "acne_type":           None,
        "confidence":          round(top1_conf, 4),
        "severity":            None,
        "recommendation_tags": ["possible_acne"] if is_acne else ["no_acne_detected"],
        "vision_model":        "vit",
        "vit_top_predictions": [
            {"label": label, "confidence": round(conf, 4)}
            for label, conf in zip(top_labels, top_probs_list)
        ],
        "raw_scores": {
            "vit_top5": top_probs_list
        },
    }
