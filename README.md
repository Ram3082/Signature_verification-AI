[README.md](https://github.com/user-attachments/files/27375443/README.md)
# 🖊️ Signature Verification AI

## What It Does

A full-stack web app that takes a handwritten signature image as input and autonomously classifies it as **genuine or forged** using a CNN — with a Grad-CAM heatmap showing exactly what the model looked at.

---

## Architecture

```
User Upload (Image)
       │
       ▼
┌─────────────────┐
│  Flask REST API  │  ← app.py
└────────┬────────┘
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
CNN Model    Classic ML
(cnn_model   (model.pkl)
 .keras)     [Render deploy]
   │
   ▼
Grad-CAM Heatmap Generator
   │
   ▼
SQLite DB  ←  Stores result + confidence score
   │
   ▼
HTML/CSS/JS Frontend (templates/ + static/)
```

> **Local:** uses `cnn_model.keras` (TensorFlow/Keras CNN)  
> **Render:** uses `model.pkl` (scikit-learn) due to memory constraints

---

## Results

| Metric    | Value |
|-----------|-------|
| Accuracy  | ~93%  |
| Precision | ~91%  |
| Recall    | ~95%  |
| F1 Score  | ~93%  |

> ⚠️ Replace with actual numbers from your `metrics.json`

---

## Technical Decisions

**Why `model.pkl` on Render instead of TensorFlow?**  
Render's free tier caps memory at ~512MB. TensorFlow alone exceeds this at runtime, so the deployed version swaps to a lightweight scikit-learn model. The full CNN is available for local use.

**Why Grad-CAM for explainability?**  
Black-box classifiers are hard to trust for security-critical tasks like signature verification. Grad-CAM generates class activation heatmaps directly from the CNN's convolutional layers — making every decision human-auditable without any retraining.

**Why SQLite over PostgreSQL?**  
The app has low concurrent write requirements. SQLite keeps the deployment self-contained, eliminates infrastructure overhead, and is fast enough for the use case.

---

## Quick Start

```bash
git clone https://github.com/Ram3082/Signature_verification-AI.git
cd Signature_verification-AI
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

---

## Stack

`Python` · `Flask` · `TensorFlow/Keras` · `scikit-learn` · `Grad-CAM` · `SQLite` · `HTML/CSS/JS` · `Render`

---

*Built by [Ram3082 — D. Uday Kiran](https://github.com/Ram3082)*
