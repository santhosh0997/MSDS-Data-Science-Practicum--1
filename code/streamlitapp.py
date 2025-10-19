import streamlit as st
import numpy as np
import joblib
import re
import nltk
import textstat
import torch
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# -------------------------
# NLTK setup
# -------------------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except:
    nltk.download('averaged_perceptron_tagger', quiet=True)

STOP = set(stopwords.words("english"))

# -------------------------
# Load artifacts
# -------------------------
pca    = joblib.load("artifacts/models/pca_embeddings.joblib")
scaler = joblib.load("artifacts/models/scaler_ling.joblib")
clf    = joblib.load("artifacts/models/hybrid_logistic.joblib")

# Optional rare word set (from training)
try:
    RARE_WORDS = joblib.load("artifacts/models/rare_words_set.joblib")
except:
    RARE_WORDS = None

# SBERT and GPT2
sbert = SentenceTransformer("all-MiniLM-L6-v2")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_gpt2_tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
_gpt2_tok.pad_token = _gpt2_tok.eos_token
_gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2").to(_device).eval()

# -------------------------
# Linguistic feature functions
# -------------------------
def _safe_div(a, b):
    return a / b if b else 0.0

def perplexity(text: str, max_length=512) -> float:
    if not text.strip():
        return 0.0
    enc = _gpt2_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(_device)
    with torch.no_grad():
        out = _gpt2(**enc, labels=enc["input_ids"])
        loss = float(out.loss.detach().cpu().item())
    return float(np.exp(loss))

def entropy(text: str) -> float:
    words = word_tokenize(text.lower())
    counts = Counter(words)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))

def avg_sentence_len(text: str) -> float:
    sents = sent_tokenize(text)
    if not sents:
        return 0.0
    return float(np.mean([len(word_tokenize(s)) for s in sents]))

def repetition_ratio(text: str) -> float:
    words = word_tokenize(text.lower())
    return float(len(words) / (len(set(words)) + 1e-9)) if words else 0.0

def _pos_ratios(text: str):
    tokens = word_tokenize(text.lower())
    if not tokens:
        return 0.0, 0.0, 0.0
    try:
        tags = pos_tag(tokens, tagset=None, lang='eng')
    except:
        tags = pos_tag(tokens)
    total = len(tags)
    nouns = sum(1 for _, t in tags if t.startswith('NN'))
    verbs = sum(1 for _, t in tags if t.startswith('VB'))
    prons = sum(1 for _, t in tags if t in ('PRP','PRP$'))
    return _safe_div(nouns,total), _safe_div(verbs,total), _safe_div(prons,total)

def flesch_reading_ease(text: str) -> float:
    return float(textstat.flesch_reading_ease(text or ""))

def gunning_fog(text: str) -> float:
    return float(textstat.gunning_fog(text or ""))

def rare_word_ratio(text: str) -> float:
    tokens = [w.lower() for w in word_tokenize(text)]
    if not tokens:
        return 0.0
    if RARE_WORDS is None:
        return 0.0
    rare = sum(1 for w in tokens if w in RARE_WORDS)
    return _safe_div(rare, len(tokens))

def stopword_ratio(text: str) -> float:
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    if not tokens:
        return 0.0
    sw = sum(1 for w in tokens if w in STOP)
    return _safe_div(sw, len(tokens))

def compute_linguistic_vector_11(text: str) -> np.ndarray:
    noun_r, verb_r, pron_r = _pos_ratios(text)
    feats = [
        perplexity(text),           # 1
        entropy(text),              # 2
        avg_sentence_len(text),     # 3
        repetition_ratio(text),     # 4
        noun_r,                     # 5
        verb_r,                     # 6
        pron_r,                     # 7
        flesch_reading_ease(text),  # 8
        gunning_fog(text),          # 9
        rare_word_ratio(text),      # 10
        stopword_ratio(text),       # 11
    ]
    return np.array(feats, dtype=np.float32)

# -------------------------
# Feature pipeline
# -------------------------
def text_to_hybrid_features(text: str):
    emb = sbert.encode([text])
    emb_pca = pca.transform(emb)
    ling = compute_linguistic_vector_11(text).reshape(1, -1)
    ling_scaled = scaler.transform(ling)
    hybrid = np.hstack([emb_pca, ling_scaled])
    return hybrid

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Human vs AI Text Classifier", layout="centered")
st.title(" Human vs AI — Hybrid Logistic Classifier")
st.caption("Embeddings (SBERT) + 11 Linguistic features → Logistic Regression (Hybrid Logistic).")

example = (
    "This product exceeded my expectations. The build quality is solid and the battery lasts all day. "
    "I would recommend it to anyone looking for reliability."
)

with st.form("inference"):
    text = st.text_area("Paste text to classify:", value=example, height=200)
    submit = st.form_submit_button("Classify")

if submit:
    text = text.strip()
    if len(text) == 0:
        st.warning("Please paste some text.")
    else:
        with st.spinner("Scoring..."):
            X = text_to_hybrid_features(text)
            proba = None
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)[0]
            y_pred = clf.predict(X)[0]

        label_map = {0: "Human", 1: "AI"}
        pred_name = label_map.get(int(y_pred), str(y_pred))

        st.subheader(f"Prediction: {pred_name}")
        if proba is not None and len(proba) == 2:
            st.write("Confidence")
            st.progress(float(proba[1]))
            st.write(f"AI probability: **{proba[1]:.3f}**")
            st.write(f"Human probability: **{proba[0]:.3f}**")

        with st.expander("Show intermediate features"):
            ling_vec = compute_linguistic_vector_11(text)
            st.json({
                "perplexity": float(ling_vec[0]),
                "entropy": float(ling_vec[1]),
                "avg_sentence_len": float(ling_vec[2]),
                "repetition_ratio": float(ling_vec[3]),
                "noun_ratio": float(ling_vec[4]),
                "verb_ratio": float(ling_vec[5]),
                "pronoun_ratio": float(ling_vec[6]),
                "flesch_reading_ease": float(ling_vec[7]),
                "gunning_fog": float(ling_vec[8]),
                "rare_word_ratio": float(ling_vec[9]),
                "stopword_ratio": float(ling_vec[10]),
            })
            if RARE_WORDS is None:
                st.warning("Rare-word set not found; rare_word_ratio defaults to 0.")
st.markdown("---")
st.caption("Ensure PCA, scaler, and hybrid logistic artifacts come from the same training run.")
