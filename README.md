# 🛡️ SENTINEL-AI: Advanced Financial Fraud Detection System

> An enterprise-grade, high-performance machine learning system designed to detect fraudulent financial transactions with exceptional accuracy, featuring an interactive dual-mode web dashboard.

---

## 🌟 Overview
**SENTINEL-AI** is a cutting-edge fraud detection solution built using advanced machine learning algorithms (Random Forest) optimized for high precision, recall, and reliability. Designed with a modular architecture and a user-friendly Streamlit interface, it supports both real-time single transaction analysis and bulk CSV batch processing.

---

## ✨ Key Features
- **High-Performance ML Model:** Achieved stellar metrics (Precision, Recall, F1-Score, and ROC-AUC) during rigorous model evaluation.
- **Dual-Mode Interface:** 
  - 💳 **Single Transaction Mode:** Real-time risk assessment with automated decision explainability.
  - 📁 **Bulk CSV Processing Mode:** Enterprise-grade batch analysis capable of evaluating hundreds of transactions simultaneously with downloadable reports.
- **Clean Modular Architecture:** Structured professionally for scalability, testing, and academic excellence.

---

## 📂 Project Structure
```text
SENTINEL_AI/
│
├── DATA/              # Dataset files (e.g., transactions.csv)
├── MODELS/            # Serialized machine learning models (.pkl)
├── SRC/               # Core scripts for data processing and training
├── TESTS/             # Unit tests and stability checks
├── app.py             # Main Streamlit web application
├── evaluate.py        # Model evaluation and metrics verification
└── REQUIREMENTS.txt   # Project dependencies
