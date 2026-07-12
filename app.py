"""
app.py - FastAPI app untuk sentiment analysis komentar "rupiah melemah"

Cara jalanin lokal:
    pip install -r requirements.txt
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Lalu buka http://localhost:8000/docs untuk coba lewat Swagger UI,
atau test lewat curl:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "kebijakan pemerintah tidak bagus untuk rakyat"}'
"""

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from preprocessing import MODEL_INFO, predict_with_confidence

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Sentiment Analysis API - Rupiah Melemah",
    description=(
        "Klasifikasi sentimen komentar YouTube berbahasa Indonesia "
        "(topik pelemahan rupiah) beserta confidence score per kelas."
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks komentar yang mau dianalisis")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Daftar teks komentar")


class PredictResult(BaseModel):
    text: str
    preprocessed_text: str
    label: str
    confidence: float
    all_scores: Dict[str, float]


@app.get("/")
def home():
    """Halaman web sederhana: input teks -> hasil sentimen + confidence score."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api")
def api_info():
    return {
        "message": "Sentiment Analysis API aktif",
        "model": MODEL_INFO.get("model_name"),
        "classes": MODEL_INFO.get("classes"),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResult)
def predict(payload: PredictRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong")
    return predict_with_confidence(payload.text)


@app.post("/predict-batch", response_model=List[PredictResult])
def predict_batch(payload: BatchPredictRequest):
    cleaned = [t for t in payload.texts if t and t.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Semua teks kosong")
    return [predict_with_confidence(t) for t in cleaned]
