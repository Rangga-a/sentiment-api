# Dockerfile - default menjalankan versi FastAPI.
# Untuk pakai Flask, ganti baris CMD paling bawah (lihat komentar).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# FastAPI (default). Render/Railway/Fly.io menginjeksikan variabel PORT saat runtime,
# jadi kita baca dari situ (fallback ke 8000 kalau dijalankan manual via `docker run`).
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Flask (uncomment baris di bawah, comment baris di atas):
# CMD ["sh", "-c", "python app_flask.py"]
