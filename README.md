# NetGuardian

**Web Attack Detection** pipeline combining:

* **Binary Classification** (benign vs. malicious)
* **Anomaly Detection** (One-Class SVM, LOF, etc.)
* **Multi-class Classification** (Normal, SQLi, XSS, Other)
* **Streamlit Dashboard** & **Flask API** for real-time inference

---

## 🚀 Features

1. **Anomaly Detection & Binary Classification**

   * TF-IDF + SVM / Random Forest / XGBoost
   * One-Class SVM tuned on “normal” traffic

2. **Multi-Class Classification**

   * Advanced feature engineering (regex heuristics, URL decoding, entropy, char/word n-grams, numerical features)
   * LightGBM / XGBoost / RandomForest / LogisticRegression

3. **End-to-end Demo**

   * `dashboard.py`: Streamlit UI
   * `flask_app.py`: REST API serving models
   * Random sampling of real CSIC requests (benign, SQLi, XSS, Other)

---


## 🛠️ Training & Model Generation

1. **Binary & Anomaly**

   ```bash
   python csic_classifier_and_anomaly_model.py
   ```

   * Trains TF-IDF + SVM and One-Class SVM
   * Saves `vectorizers/vectorizer.pkl`, `models/classifier.pkl`, `models/anomaly_detector.pkl`
   

2. **Multi-class**

   ```bash
   python multiclass_attack_classifier.py
   ```

   * Extracts advanced features, trains LightGBM (or other)
   * Saves `vectorizers/{tfidf_char,tfidf_word,count_vec,scaler}.pkl` and `models/lightgbm_model.pkl`

---

## 🔍 CSIC Dataset

* **csic\_database.csv**: real HTTP requests (normal & attacks)
* Used for sampling examples and model training

---
