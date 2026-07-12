"""
app_flask.py - Alternatif Flask untuk sentiment analysis komentar "rupiah melemah"

Cara jalanin lokal:
    pip install -r requirements.txt
    python app_flask.py

Server default di http://localhost:5000
Test lewat curl:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "kebijakan pemerintah tidak bagus untuk rakyat"}'
"""

import os

from flask import Flask, jsonify, request, send_from_directory

from preprocessing import MODEL_INFO, predict_with_confidence

app = Flask(__name__, static_folder="static")


@app.get("/")
def home():
    """Halaman web sederhana: input teks -> hasil sentimen + confidence score."""
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api")
def api_info():
    return jsonify(
        {
            "message": "Sentiment Analysis API aktif",
            "model": MODEL_INFO.get("model_name"),
            "classes": MODEL_INFO.get("classes"),
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Teks tidak boleh kosong"}), 400
    return jsonify(predict_with_confidence(text))


@app.post("/predict-batch")
def predict_batch():
    data = request.get_json(silent=True) or {}
    texts = [t for t in data.get("texts", []) if t and t.strip()]
    if not texts:
        return jsonify({"error": "texts tidak boleh kosong"}), 400
    return jsonify([predict_with_confidence(t) for t in texts])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
