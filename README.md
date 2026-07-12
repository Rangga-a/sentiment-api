# Sentiment Analysis "Rupiah Melemah" — Deployment

**Live demo:** [sentiment-api-six.vercel.app](https://sentiment-api-six.vercel.app/)

## Tentang project ini

Project ini adalah sistem klasifikasi sentimen untuk komentar YouTube berbahasa Indonesia
yang membahas topik pelemahan nilai tukar rupiah. Tujuannya: dari sebuah komentar bebas,
sistem menebak apakah nada komentar itu **Negative**, **Neutral**, atau **Positive**,
lengkap dengan **confidence score** (seberapa yakin model terhadap prediksinya).

Alur project secara garis besar:

1. **Pengumpulan data** — 9.927 komentar YouTube mentah (kolom teks, author, waktu, jumlah
   like, dll.) terkait video-video seputar pelemahan rupiah.
2. **Auto-labeling** — karena data mentah tidak berlabel, tiap komentar diberi label
   sentimen awal secara otomatis menggunakan model transformer (HuggingFace, berbasis
   BERTweet Indonesia) sebagai *pseudo-labeling*.
3. **Preprocessing teks** — pipeline khusus Bahasa Indonesia: pembersihan simbol/emoji/URL,
   normalisasi kata gaul (kamus 128 entri), penggabungan kata negasi (mis. "tidak_bagus"
   supaya maknanya tidak terbalik saat dihitung TF-IDF), penghapusan stopword (843 kata),
   dan stemming (Sastrawi).
4. **Ekstraksi fitur** — teks bersih diubah jadi vektor numerik dengan **TF-IDF**.
5. **Pemodelan** — 6 model klasik dibandingkan: Naive Bayes, Decision Tree, Random Forest,
   AdaBoost, Linear SVM, dan Logistic Regression (tuned). Model terbaik dipilih otomatis
   berdasarkan skor **F1-macro** tertinggi di data uji.
6. **Confidence score** — kalau model terpilih tidak punya `predict_proba` bawaan (mis.
   Linear SVM), model dikalibrasi otomatis dengan `CalibratedClassifierCV` supaya tetap
   bisa mengeluarkan probabilitas per kelas, bukan cuma label mentah.
7. **Deployment** — model final diekspor (`model.pkl`, `tfidf_vectorizer.pkl`,
   `preprocessing_assets.pkl`) lalu dibungkus jadi API (FastAPI/Flask) dengan tampilan web
   sederhana di atasnya, dan di-deploy ke internet.

## Hasil akhir

- **Model final yang dipakai:** `LogisticRegression` (terpilih otomatis sebagai model
  dengan F1-macro terbaik dari 6 kandidat yang dibandingkan).
- **Kelas output:** `Negative`, `Neutral`, `Positive`.
- **Confidence score:** tersedia native dari model ini (`predict_proba`), tidak perlu
  kalibrasi tambahan.
- Angka metrik lengkap (accuracy, precision, recall, F1 per kelas, dan tabel perbandingan
  ke-6 model) ada di output notebook `Sentiment_Analysis_Final.ipynb` Bagian 10, karena
  nilainya tergantung random seed/split saat notebook dijalankan.

**Keterbatasan yang perlu diketahui:** karena berbasis TF-IDF + model klasik (bukan
transformer end-to-end), model ini **tidak bisa membaca sarkasme/sindiran** — kalimat
positif secara harfiah pun kadang salah diklasifikasi kalau polanya mirip komentar negatif
di data training. Ini sudah dicatat sebagai batasan di Bagian 12 notebook.

## Struktur folder

```
deployment/
├── app.py                   # FastAPI app
├── app_flask.py              # Flask app (alternatif)
├── preprocessing.py          # preprocessing + loader model (dipakai app.py & app_flask.py)
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
├── static/
│   ├── index.html            # tampilan web (dark theme)
│   └── style.css
└── model_artifacts/
    ├── model.pkl
    ├── tfidf_vectorizer.pkl
    └── preprocessing_assets.pkl
```

## Cara menjalankan lokal

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**FastAPI:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Buka `http://localhost:8000/` — tampilan web langsung muncul: input teks, klik tombol,
hasil (label + confidence score per kelas) tampil di bawahnya. Dokumentasi API teknis
(Swagger) ada di `/docs`.

**Flask:**
```bash
python app_flask.py
```
Buka `http://localhost:5000/` — tampilan yang sama.

## Contoh request API

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "kebijakan pemerintah tidak bagus untuk rakyat"}'
```

Contoh response:
```json
{
  "text": "kebijakan pemerintah tidak bagus untuk rakyat",
  "preprocessed_text": "bijak perintah tidak_bagus rakyat",
  "label": "Negative",
  "confidence": 0.8557,
  "all_scores": {
    "Negative": 0.8557,
    "Neutral": 0.0708,
    "Positive": 0.0735
  }
}
```

Endpoint yang tersedia (sama di kedua versi):
- `GET /` — halaman web sederhana (input teks → hasil sentimen + confidence score)
- `GET /api` — info model yang sedang aktif (JSON)
- `GET /health` — health check
- `POST /predict` — `{"text": "..."}` → satu prediksi + confidence score
- `POST /predict-batch` — `{"texts": ["...", "..."]}` → list prediksi (khusus FastAPI)

## Deploy ke internet

Project ini sudah live di **Vercel**: https://sentiment-api-six.vercel.app/

Alternatif lain yang juga bisa dipakai (gratis, deploy dari GitHub):

### Render (via Docker)

1. Push project ke GitHub.
2. [render.com](https://render.com) → **New +** → **Web Service** → connect repo.
3. Render otomatis pakai `Dockerfile`. Pilih **Instance Type: Free**.
4. **Create Web Service** → tunggu build ~2-5 menit → dapat URL publik.

`Dockerfile` sudah disesuaikan untuk baca environment variable `PORT` otomatis
(`--port ${PORT:-8000}`), sesuai kebutuhan Render/Railway/Fly.io.

Catatan: Render kadang meminta info kartu untuk verifikasi akun meski tetap pakai
instance Free (bukan ditagih — cuma verifikasi $1 yang di-refund). Kalau tidak mau
memasukkan kartu, alternatif tanpa kartu sama sekali: **Hugging Face Spaces** (SDK: Docker)
atau **Vercel** seperti yang sudah dipakai di project ini.

### Auto-deploy

Baik Render maupun Vercel akan otomatis build ulang & deploy versi terbaru setiap kali
`git push` ke branch utama — tidak perlu langkah manual tambahan.

## Catatan penting

- Preprocessing di `preprocessing.py` **harus identik** dengan yang dipakai saat training
  (lihat Bagian 6 di notebook). Kalau preprocessing di notebook diubah, `preprocessing.py`
  di sini wajib disesuaikan juga, kalau tidak hasil prediksi bisa meleset.
- `confidence` adalah probabilitas kelas dengan skor tertinggi (`predict_proba().max()`).
  `all_scores` berisi probabilitas semua kelas, dipakai untuk menampilkan progress bar
  skor di tampilan web.
- Model ini **tidak mendeteksi sarkasme** — confidence tinggi tidak menjamin makna
  sindiran ikut terbaca benar (lihat bagian "Hasil akhir" di atas).
