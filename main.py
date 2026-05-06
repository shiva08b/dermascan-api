import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from pydantic import BaseModel
from inference import predict
from vit_inference import predict_vit
from skincare import get_skincare_routine, get_ai_products, get_ai_remedies

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="DermaScan API",
    description="AI-powered acne detection and skincare recommendation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Provider enum ──────────────────────────────────────────────
class AIProvider(str, Enum):
    gemini      = "gemini"
    groq        = "groq"
    cohere      = "cohere"
    huggingface = "huggingface"
    openrouter  = "openrouter"
    combined    = "combined"
    consensus   = "consensus"

# ── Vision model enum ─────────────────────────────────────────────────────
class VisionModel(str, Enum):
    efficientnet = "efficientnet"   # default: full 2-stage pipeline
    vit          = "vit"            # experimental: ViT binary classifier

# ── Request bodies for standalone endpoints ───────────────────────────
class ProductsRequest(BaseModel):
    acne_type:   str = "unknown"
    severity:    str = "mild"
    skin_type:   str = "combination"
    tags:        list = []
    ai_provider: str = "gemini"

class RemediesRequest(BaseModel):
    acne_type:   str = "unknown"
    skin_type:   str = "combination"
    ai_provider: str = "gemini"

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "DermaScan API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze",
            "health":  "GET /health",
            "docs":    "GET /docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(
    file:         UploadFile = File(...),
    ai_provider:  AIProvider  = AIProvider.gemini,
    vision_model: VisionModel = VisionModel.efficientnet,
    skin_type:    str         = "combination",
):
    # Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB")

    # ── ViT path (experimental, binary only) ─────────────────────────────────
    if vision_model == VisionModel.vit:
        vit_result = predict_vit(image_bytes)
        if not vit_result["is_acne"]:
            return {
                **vit_result,
                "skincare_routine": {
                    "provider": "none",
                    "routine": "No acne detected. Maintain a basic gentle cleanser + moisturizer + SPF routine.",
                    "status": "skipped"
                }
            }
        # ViT detected possible acne — get routine + products + remedies in parallel
        routine, products, remedies = await asyncio.gather(
            get_skincare_routine("unknown", "mild", vit_result["recommendation_tags"], ai_provider.value),
            get_ai_products("unknown", "mild", skin_type, vit_result["recommendation_tags"], ai_provider.value),
            get_ai_remedies("unknown", skin_type, ai_provider.value),
            return_exceptions=True,
        )
        response = {**vit_result, "skincare_routine": routine if isinstance(routine, dict) else {"provider": "none", "routine": "", "status": "error"}}
        if isinstance(products, list) and products:
            response["product_recommendations"] = {"sponsored": [p for p in products if p.get("sponsored")], "organic": [p for p in products if not p.get("sponsored")], "all": products}
        if isinstance(remedies, list) and remedies:
            response["home_remedies"] = remedies
        return response

    # ── EfficientNet path (default, full 2-stage pipeline) ────────────────────
    scan_result = predict(image_bytes)

    # If no acne detected, skip AI routine
    if not scan_result["is_acne"]:
        return {
            **scan_result,
            "skincare_routine": {
                "provider": "none",
                "routine": "No acne detected. Maintain a basic gentle cleanser + moisturizer + SPF routine.",
                "status": "skipped"
            }
        }

    # Stage 3 — All three AI calls in parallel
    acne_type = scan_result["acne_type"]
    severity  = scan_result["severity"]
    tags      = scan_result["recommendation_tags"]
    provider  = ai_provider.value

    routine, products, remedies = await asyncio.gather(
        get_skincare_routine(acne_type, severity, tags, provider),
        get_ai_products(acne_type, severity, skin_type, tags, provider),
        get_ai_remedies(acne_type, skin_type, provider),
        return_exceptions=True,
    )

    response = {
        **scan_result,
        "skincare_routine": routine if isinstance(routine, dict) else {"provider": "none", "routine": "", "status": "error"},
    }
    if isinstance(products, list) and products:
        response["product_recommendations"] = {
            "sponsored": [p for p in products if p.get("sponsored")],
            "organic":   [p for p in products if not p.get("sponsored")],
            "all":       products,
        }
    if isinstance(remedies, list) and remedies:
        response["home_remedies"] = remedies

    return response


# ── Standalone products endpoint ──────────────────────────────────────
@app.post("/products")
async def products_endpoint(req: ProductsRequest):
    items = await get_ai_products(
        acne_type = req.acne_type,
        severity  = req.severity,
        skin_type = req.skin_type,
        tags      = req.tags,
        provider  = req.ai_provider,
    )
    return {
        "sponsored": [p for p in items if p.get("sponsored")],
        "organic":   [p for p in items if not p.get("sponsored")],
        "all":       items,
    }


# ── Standalone remedies endpoint ──────────────────────────────────────
@app.post("/remedies")
async def remedies_endpoint(req: RemediesRequest):
    return await get_ai_remedies(
        acne_type = req.acne_type,
        skin_type = req.skin_type,
        provider  = req.ai_provider,
    )