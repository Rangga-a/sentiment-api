import os
import re
import pickle

import emoji
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")


def load_artifacts():
    try:
        model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
        tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"))
        with open(os.path.join(ARTIFACT_DIR, "preprocessing_assets.pkl"), "rb") as f:
            assets = pickle.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Artefak model tidak ditemukan di folder model_artifacts/. "
            "Pastikan model.pkl, tfidf_vectorizer.pkl, dan preprocessing_assets.pkl "
            "sudah ditaruh di folder deployment/model_artifacts/."
        ) from e

    # model_info.pkl bersifat opsional (mis. kalau export dari notebook versi lama
    # belum menyimpannya). Kalau tidak ada, info model diturunkan dari objek model itu sendiri.
    model_info_path = os.path.join(ARTIFACT_DIR, "model_info.pkl")
    if os.path.exists(model_info_path):
        with open(model_info_path, "rb") as f:
            model_info = pickle.load(f)
    else:
        model_info = {
            "model_name": type(model).__name__,
            "classes": list(model.classes_) if hasattr(model, "classes_") else None,
        }

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Model yang di-load tidak mendukung predict_proba, sehingga confidence "
            "score tidak bisa dihitung. Pastikan Bagian 10.1 di notebook (kalibrasi "
            "model dengan CalibratedClassifierCV bila perlu) sudah dijalankan sebelum export."
        )

    return model, tfidf, assets, model_info


MODEL, TFIDF, ASSETS, MODEL_INFO = load_artifacts()

SLANG_DICT = ASSETS["slang_dict"]
NEGATION_WORDS = ASSETS["negation_words"]
STOPWORD_ID = ASSETS["stopword_id"]

_stemmer = StemmerFactory().create_stemmer()
_stem_cache = {}


def remove_emoji(text: str) -> str:
    return emoji.replace_emoji(str(text), replace=" ")


def cleaning(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.replace(".", " ").replace("_", " ")
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return text.split()


def normalize_slang(tokens):
    return [SLANG_DICT.get(word, word) for word in tokens]


def merge_negation(tokens):
    merged = []
    skip = False
    for i, tok in enumerate(tokens):
        if skip:
            skip = False
            continue
        if tok in NEGATION_WORDS and i + 1 < len(tokens):
            merged.append(f"{tok}_{tokens[i + 1]}")
            skip = True
        else:
            merged.append(tok)
    return merged


def remove_stopwords(tokens):
    return [w for w in tokens if w not in STOPWORD_ID]


def _stem_word(word: str) -> str:
    if "_" in word:
        parts = word.split("_")
        return "_".join(_stem_cache.setdefault(p, _stemmer.stem(p)) for p in parts)
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer.stem(word)
    return _stem_cache[word]


def stemming(tokens) -> str:
    return " ".join(_stem_word(w) for w in tokens)


def preprocess(text: str) -> str:
    """Pipeline preprocessing lengkap: emoji -> cleaning -> tokenize -> slang -> negasi -> stopword -> stemming."""
    text = remove_emoji(text)
    text = cleaning(text)
    tokens = tokenize(text)
    tokens = normalize_slang(tokens)
    tokens = merge_negation(tokens)
    tokens = remove_stopwords(tokens)
    return stemming(tokens)


def predict_with_confidence(text: str) -> dict:
    """Preprocess teks, prediksi label, dan hitung confidence score (probabilitas per kelas)."""
    clean_text = preprocess(text)
    vec = TFIDF.transform([clean_text])
    proba = MODEL.predict_proba(vec)[0]
    classes = MODEL.classes_
    idx = int(proba.argmax())

    return {
        "text": text,
        "preprocessed_text": clean_text,
        "label": str(classes[idx]),
        "confidence": float(proba[idx]),
        "all_scores": {str(cls): float(p) for cls, p in zip(classes, proba)},
    }
