# Deployment - Sentiment Analysis "Rupiah Melemah"

API untuk klasifikasi sentimen komentar YouTube berbahasa Indonesia, lengkap dengan
**confidence score** (probabilitas per kelas), tersedia dalam 2 versi: **FastAPI** dan **Flask**.

## 1. Struktur folder

```
deployment/
├── app.py                 # FastAPI app
├── app_flask.py            # Flask app (alternatif)
├── preprocessing.py        # preprocessing + loader model (dipakai app.py & app_flask.py)
├── requirements.txt
├── Dockerfile
├── README.md
└── model_artifacts/         # <-- HARUS diisi manual, lihat langkah 2
    ├── model.pkl
    ├── tfidf_vectorizer.pkl
    ├── preprocessing_assets.pkl
    └── model_info.pkl
```

## 2. Siapkan model_artifacts/

File-file di `model_artifacts/` dihasilkan dari notebook `Sentiment_Analysis_Final.ipynb`
(Bagian 10.1 dan Bagian 13):

1. Jalankan seluruh notebook sampai selesai di Google Colab.
2. Bagian 10.1 otomatis memilih model dengan F1-macro terbaik. Kalau model itu
   `LinearSVC` (tidak punya `predict_proba` bawaan), notebook otomatis mengkalibrasinya
   dengan `CalibratedClassifierCV` supaya tetap bisa menghasilkan confidence score.
3. Bagian 13 menyimpan `model.pkl`, `tfidf_vectorizer.pkl`, `preprocessing_assets.pkl`,
   `model_info.pkl` ke folder `model_artifacts/`, lalu men-zip-nya jadi
   `model_artifacts.zip` (otomatis ke-download kalau di Colab).
4. Ekstrak `model_artifacts.zip` ke folder `deployment/model_artifacts/` ini (folder yang
   sudah disediakan di sini masih kosong, isi manual dengan 4 file di atas).

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Jalankan lokal

**FastAPI:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Buka `http://localhost:8000/docs` untuk coba lewat Swagger UI interaktif.

**Flask:**
```bash
python app_flask.py
```
Server jalan di `http://localhost:5000`.

## 5. Contoh request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "kebijakan pemerintah tidak bagus untuk rakyat"}'
```

Contoh response (label + confidence score + skor tiap kelas):

```json
{
  "text": "kebijakan pemerintah tidak bagus untuk rakyat",
  "preprocessed_text": "bijak perintah tidak_bagus rakyat",
  "label": "negative",
  "confidence": 0.87,
  "all_scores": {
    "negative": 0.87,
    "positive": 0.13
  }
}
```

Endpoint yang tersedia (sama di kedua versi):
- `GET /` — info model yang sedang aktif
- `GET /health` — health check
- `POST /predict` — `{"text": "..."}` → satu prediksi + confidence score
- `POST /predict-batch` — `{"texts": ["...", "..."]}` → list prediksi (endpoint `/predict-batch` khusus FastAPI, di Flask juga tersedia dengan payload sama)

## 6. Deploy ke internet lewat GitHub (gratis, pakai Render)

### Langkah A — Push project ke GitHub

```bash
cd sentiment-api
git init
git add .
git commit -m "Sentiment analysis API - FastAPI + Flask"
```

Buat repo baru di [github.com](https://github.com) (kosong, tanpa README/gitignore bawaan),
lalu:
```bash
git remote add origin https://github.com/<username-kamu>/sentiment-rupiah-api.git
git branch -M main
git push -u origin main
```

### Langkah B — Deploy ke Render (free tier)

1. Buka [render.com](https://render.com), sign up/login pakai akun GitHub.
2. Klik **New +** → **Web Service**.
3. Connect ke repo `sentiment-rupiah-api` yang barusan di-push.
4. Render otomatis mendeteksi `Dockerfile` di root repo dan memilih **Docker** sebagai
   environment build.
5. Isi konfigurasi:
   - **Name**: bebas, misal `sentiment-rupiah-api`
   - **Region**: pilih yang terdekat (Singapore kalau tersedia)
   - **Instance Type**: **Free**
6. Klik **Create Web Service**.
7. Render build image dari `Dockerfile`, lalu jalankan container. Progress bisa dipantau
   di tab **Events**.
8. Setelah build sukses (~2-5 menit), Render kasih URL publik seperti
   `https://sentiment-rupiah-api.onrender.com`.

**Penting soal port:** Render otomatis meng-inject environment variable `PORT` (default
`10000`) dan container **wajib** listen di port itu. `Dockerfile` di project ini sudah
disesuaikan untuk baca `PORT` secara otomatis (`--port ${PORT:-8000}`), jadi tidak perlu
setting manual — cukup deploy seperti langkah di atas.

### Langkah C — Test dari internet

```bash
curl -X POST https://sentiment-rupiah-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "rupiah makin hancur"}'
```

Atau langsung buka `https://sentiment-rupiah-api.onrender.com/` di browser — tampilan web
yang sama seperti di lokal.

### Auto-deploy setelah update

Setiap kali kamu `git push` ke branch `main`, Render otomatis build ulang dan deploy versi
terbaru — tidak perlu klik apa-apa lagi di dashboard.

### Catatan free tier

- Free tier Render akan **sleep** setelah idle beberapa menit tanpa traffic. Request
  pertama setelah idle butuh ~30-60 detik untuk "bangun" (cold start) — ini normal,
  bukan error.
- Free tier juga punya batas jam aktif per bulan; untuk project belajar/portofolio
  biasanya lebih dari cukup.
- Alternatif gratis serupa kalau Render penuh/di-suspend: **Railway** (flow hampir sama,
  auto-detect Dockerfile) atau **Fly.io** (perlu install CLI, `fly launch` → `fly deploy`).

## 7. Catatan penting

- Preprocessing di `preprocessing.py` **harus identik** dengan yang dipakai saat training
  (lihat Bagian 6 di notebook). Kalau nanti preprocessing di notebook diubah, `preprocessing.py`
  di sini juga wajib disesuaikan, kalau tidak hasil prediksi bisa meleset.
- `confidence` adalah probabilitas kelas dengan skor tertinggi (`predict_proba().max()`).
  `all_scores` berisi probabilitas untuk semua kelas, berguna kalau mau tampilkan bar/progress
  confidence di frontend.
- Model berbasis TF-IDF + klasik ini **tidak mendeteksi sarkasme** (lihat catatan di Bagian 12
  notebook) — confidence score tinggi bukan berarti makna sindiran ikut terbaca benar.
