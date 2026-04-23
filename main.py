from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from pydantic import BaseModel
from inference import predict
from skincare import get_skincare_routine

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
    file: UploadFile = File(...),
    ai_provider: AIProvider = AIProvider.gemini
):
    # Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB")

    # Stage 1 & 2 — DermaScan inference
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

    # Stage 3 — Skincare routine from chosen AI
    routine = await get_skincare_routine(
        acne_type = scan_result["acne_type"],
        severity  = scan_result["severity"],
        tags      = scan_result["recommendation_tags"],
        provider  = ai_provider.value
    )

    return {**scan_result, "skincare_routine": routine}