# 📄 Privacy-First Smart Document Scanner Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**100% Offline. Global Multi-Language. AI Context Awareness. Professional Grade.**

This is the **Pro Version** of the Smart Document Scanner. It is designed to be a private, secure alternative to commercial giants like Adobe Scan and CamScanner, providing enterprise-level intelligence without sacrificing your data sovereignty.

---

## 🚀 Pro Features

### 🧠 AI Context Engine
The tool automatically identifies the document type by analyzing the OCR text:
- **Invoices**: Detects amounts, tax labels, and merchants.
- **Identity Docs**: Identifies Passports and National IDs.
- **Receipts**: Finds merchant names and subtotal data.

### 🛡️ PII Shield (Redaction Ready)
Automatically scans for and flags sensitive Personal Identifiable Information (PII):
- **Credit Cards** (13-16 digit pattern detection)
- **Social Security Numbers** (US/Global formats)
- **Emails & Phone Numbers**
- **Note:** Data is flagged in the UI so you can ensure safety before sharing.

### 🌍 Global Multi-Language Support
Supports **40+ major languages** including:
- **East Asian**: Chinese (Simplified/Traditional), Japanese, Korean, Thai, Vietnamese.
- **Indic**: Sinhalese, Tamil, Hindi, Bengali.
- **European**: German, French, Spanish, Russian, Portuguese, Italian, Polish, Dutch.
- **Middle Eastern**: Arabic, Hebrew, Turkish.

### 🛠️ 11-Step Computer Vision Pipeline
- **AI Hand Removal**: Erases fingers from the document.
- **Perspective Flattening**: Fixes tilts and rotations.
- **Shadow Subtraction**: Removes phone shadows.
- **Table Detection**: Automatically finds grid structures.

---

## 🔒 The Privacy Manifesto

1. **Zero Bytes Uploaded**: Processing happens in memory; no files are stored.
2. **Bring Your Own Privacy**: Use the Docker version to run 100% air-gapped (no internet).
3. **No Tracking**: No analytics, no cookies, no user accounts.
4. **Local Engine**: Uses a local Tesseract binary and OpenCV logic.

---

## 📦 Installation

### 1. Install Tesseract
Follow the [Official Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html). Make sure to install the **Language Packs** you need.

### 2. Run the App
```bash
git clone https://github.com/rimashrimsan/OCR_Document_Processing.git
pip install -r requirements.txt
streamlit run app.py
```

---

## 🗺️ Roadmap
- [x] AI Hand Removal
- [x] Searchable PDF Support
- [x] QR/Barcode Extraction
- [x] **Smart Context Identification** (NEW)
- [x] **PII Detection Shield** (NEW)
- [ ] **Table-to-Excel Export** (In Progress)
- [ ] **Batch Redaction** (Coming Soon)

---

## ⚖️ License
Distributed under the **GNU GPL v3 License**. 
Copyright (c) 2024 **rimashrimsan**.
