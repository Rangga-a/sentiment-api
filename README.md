# 🇮🇩 Sentiment Analysis — Rupiah Melemah

Klasifikasi sentimen komentar YouTube berbahasa Indonesia seputar isu pelemahan rupiah, lengkap dengan confidence score per kelas. Dibangun dengan TF-IDF + model Machine Learning klasik, dan web app sederhana.

**🔗 Live demo:** [sentiment-api-six.vercel.app](https://sentiment-api-six.vercel.app/)

---

## 📖 Tentang Project

Project ini menjawab pertanyaan: dari sebuah komentar bebas soal rupiah, apakah nadanya **Negative**, **Neutral**, atau **Positive** — dan seberapa yakin model terhadap prediksi itu?

Alur pengerjaannya:

1. **Pengumpulan data** — 9.927 komentar YouTube mentah dari video-video seputar pelemahan rupiah.
2. **Auto-labeling** — data mentah tidak berlabel, jadi tiap komentar diberi label sentimen awal secara otomatis menggunakan model transformer (HuggingFace, berbasis BERTweet Indonesia) sebagai *pseudo-labeling*.
3. **Preprocessing teks** — pipeline khusus Bahasa Indonesia: pembersihan simbol/emoji/URL, normalisasi kata gaul (128 entri), penggabungan kata negasi (mis. "tidak_bagus" agar maknanya tidak terbalik), penghapusan stopword (843 kata), dan stemming (Sastrawi).
4. **Ekstraksi fitur** — teks bersih diubah jadi vektor numerik dengan **TF-IDF**.
5. **Pemodelan** — 6 model klasik dibandingkan: Naive Bayes, Decision Tree, Random Forest, AdaBoost, Linear SVM, dan Logistic Regression (tuned). Model terbaik dipilih otomatis berdasarkan skor **F1-macro** tertinggi.
6. **Confidence score** — kalau model terpilih tidak punya `predict_proba` bawaan (mis. Linear SVM), otomatis dikalibrasi dengan `CalibratedClassifierCV` supaya tetap bisa mengeluarkan probabilitas per kelas.
7. **Deployment** — model diekspor lalu dibungkus jadi API (FastAPI/Flask) dengan web UI sederhana di atasnya, di-deploy ke **Vercel**.

## ✅ Hasil Akhir

| | |
|---|---|
| **Model final** | `LogisticRegression` (terpilih otomatis, F1-macro terbaik dari 6 kandidat) |
| **Kelas output** | `Negative`, `Neutral`, `Positive` |
| **Confidence score** | Native dari `predict_proba`, tanpa kalibrasi tambahan |

Angka metrik lengkap (accuracy, precision, recall, F1 per kelas, tabel perbandingan 6 model) ada di output notebook `Sentiment_Analysis_Final.ipynb` bagian perbandingan model.

> ⚠️ **Keterbatasan:** karena berbasis TF-IDF + model klasik (bukan transformer end-to-end), model ini **tidak bisa membaca sarkasme/sindiran**. Kalimat positif secara harfiah pun kadang salah diklasifikasi kalau polanya mirip komentar negatif di data training.

## 🛠️ Tech Stack

- **Model & preprocessing:** scikit-learn, TF-IDF, Sastrawi
- **Backend:** FastAPI
- **Frontend:** HTML, CSS
- **Deployment:** Vercel

## 📂 Struktur Folder

## Struktur folder

```
deployment/
├── app.py                           
├── preprocessing.py         
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
├── notebook/
│   ├── sentiment_analysis.ipynb          
│   └── rupiah_melemah.csv
├── static/
│   ├── index.html          
│   └── style.css
└── model_artifacts/
    ├── model.pkl
    ├── tfidf_vectorizer.pkl
    └── preprocessing_assets.pkl
```

## 🚀 Menjalankan Lokal

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**FastAPI:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Buka `http://localhost:8000/` — dokumentasi API teknis (Swagger) ada di `/docs`.

**Flask:**
```bash
python app_flask.py
```
Buka `http://localhost:5000/`.

## 📡 API Reference

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Tampilan web (input teks → hasil sentimen) |
| `GET` | `/api` | Info model yang sedang aktif |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Prediksi satu teks |
| `POST` | `/predict-batch` | Prediksi banyak teks sekaligus |

**Contoh request:**
```bash
curl -X POST https://sentiment-api-six.vercel.app/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "kebijakan pemerintah tidak bagus untuk rakyat"}'
```

**Contoh response:**
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

## 👤 Author

Dibuat oleh [Rangga-a](https://github.com/Rangga-a)
