"""
DR Screening Web App — FastAPI Backend

Orchestrates the full pipeline:
  Upload → Quality Check → Preprocessing → VLM Classification → Report

Serves the frontend as static files from ../frontend/.

This is a NON-CLINICAL PROTOTYPE — not a validated diagnostic tool.
"""

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from quality_check import check_quality
from preprocessing import preprocess
from vlm_client import analyze_fundus
from report import generate_report_html

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DR Screening Prototype",
    description="Diabetic Retinopathy screening using a Vision-Language Model. PROTOTYPE ONLY.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found. Place index.html in ../frontend/</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Full DR screening pipeline:
    1. Read uploaded image
    2. Quality check
    3. Preprocessing (crop, resize, CLAHE)
    4. VLM classification
    5. Return all results as JSON
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (JPEG, PNG, etc.).")

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 20 MB).")

    # --- Step 1: Quality Check ---
    quality_result = check_quality(image_bytes)

    # --- Step 2: Preprocessing ---
    try:
        enhanced_bytes, preprocess_meta = preprocess(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # --- Step 3: VLM Classification ---
    vlm_result = analyze_fundus(enhanced_bytes)

    # --- Build response ---
    # Encode images for frontend display
    original_b64 = base64.b64encode(image_bytes).decode("utf-8")
    enhanced_b64 = base64.b64encode(enhanced_bytes).decode("utf-8")

    response = {
        "quality": quality_result,
        "preprocessing": preprocess_meta,
        "original_image": original_b64,
        "enhanced_image": enhanced_b64,
        "vlm_result": {
            "dr_grade": vlm_result.dr_grade,
            "grade_label": vlm_result.grade_label,
            "confidence": vlm_result.confidence,
            "key_findings": vlm_result.key_findings,
            "approx_regions": vlm_result.approx_regions,
            "reasoning": vlm_result.reasoning,
            "image_quality_note": vlm_result.image_quality_note,
            "is_referable": vlm_result.is_referable,
            "referable_label": vlm_result.referable_label,
            "model_used": vlm_result.model_used,
            "error": vlm_result.error,
        },
    }

    return JSONResponse(content=response)


@app.post("/api/report")
async def generate_report(file: UploadFile = File(...)):
    """
    Run the full pipeline and return an HTML report (for printing to PDF).
    """
    # Validate
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Pipeline
    quality_result = check_quality(image_bytes)

    try:
        enhanced_bytes, _ = preprocess(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

    vlm_result = analyze_fundus(enhanced_bytes)

    original_b64 = base64.b64encode(image_bytes).decode("utf-8")
    enhanced_b64 = base64.b64encode(enhanced_bytes).decode("utf-8")

    # Generate HTML report
    html = generate_report_html(
        quality_status=quality_result["status"],
        quality_reasons=quality_result["reasons"],
        dr_grade=vlm_result.dr_grade,
        grade_label=vlm_result.grade_label,
        is_referable=vlm_result.is_referable,
        referable_label=vlm_result.referable_label,
        confidence=vlm_result.confidence,
        key_findings=vlm_result.key_findings,
        reasoning=vlm_result.reasoning,
        image_quality_note=vlm_result.image_quality_note,
        model_used=vlm_result.model_used,
        original_image_b64=original_b64,
        enhanced_image_b64=enhanced_b64,
    )

    return HTMLResponse(content=html)


@app.get("/api/health")
async def health_check():
    """Simple health check."""
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    return {"status": "ok", "deepseek_api_key_configured": has_key}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Add backend dir to sys.path so local imports work
    backend_dir = str(Path(__file__).resolve().parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    print("\n" + "=" * 60)
    print("  DR Screening Prototype — Starting Server")
    print("  PROTOTYPE ONLY — NOT FOR CLINICAL USE")
    print("=" * 60)
    print(f"\n  Open http://localhost:8000 in your browser\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[backend_dir],
    )
