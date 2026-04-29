# 📄 Privacy-First Smart Document Scanner

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**100% Offline. 100% Private. Zero Cloud Uploads. Professional Grade.**

Most document scanner apps (like CamScanner or Adobe Scan) harvest your data, require subscriptions, and upload your sensitive documents to remote servers for processing. This tool is built on a different philosophy: **Your data never leaves your device.**

Built with **OpenCV**, **MediaPipe AI**, **PyMuPDF**, and **Tesseract OCR**, this is a professional grade scanner that runs entirely in your browser.

---

## ✨ Key Features

### 🛠️ 11-Step Computer Vision Pipeline
- **Perspective Flattening**: Automatically detects page corners and fixes tilts/angles.
- **AI Hand Removal**: Uses MediaPipe to detect fingers and erases them from the paper.
- **Shadow Subtraction**: Computes the background light field to remove phone shadows.
- **Auto White Balance**: Fixes yellow/blue color casts from indoor lighting.
- **Adaptive Contrast**: Boosts text readability on aged or damaged paper.

### 📝 Professional OCR & Output
- **Searchable PDFs**: Generates PDFs with invisible text layers for searching/copying.
- **Multilingual Support**: Supports English, Sinhalese, Tamil, Hindi, German, and more.
- **QR/Barcode Detection**: Automatically extracts data from codes found on documents.
- **High Compression**: Uses Deflate and Garbage Collection to keep file sizes small.

### 📱 Mobile Optimized
- **Installable PWA Feel**: Optimized CSS for full-screen use on iOS and Android.
- **In-Browser Camera**: Use your phone's camera directly without any app store download.

---

## 🔒 The Privacy Manifesto

1. **Zero Bytes Uploaded**: All image processing happens in the server's ephemeral memory or your browser. No files are stored.
2. **No Tracking**: No Google Analytics, no cookies, no user accounts.
3. **Open Source**: The code is public so you can verify our privacy claims.
4. **Local OCR**: Uses a local Tesseract binary instead of cloud-based OCR APIs.

---

## 🚀 Installation & Local Run

If you want to run this on your own machine:

### 1. Install System Dependencies
- **Tesseract OCR**: [Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- **Poppler/LibGL**: (Required for PDF and OpenCV support)

### 2. Clone and Install Python Packages
```bash
git clone https://github.com/rimashrimsan/OCR_Document_Processing.git
cd OCR_Document_Processing
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 🗺️ Roadmap
- [x] AI Hand Removal
- [x] Searchable PDF Support
- [x] QR/Barcode Extraction
- [x] **Table Detection & Extraction** (NEW)
- [ ] **Digital Signatures** (Coming Soon)
- [ ] **Batch Redaction** (Coming Soon)

---

## ⚖️ License & Credits
Distributed under the **GNU GPL v3 License**. 
Copyright (c) 2024 **rimashrimsan**.

Built with ❤️ for a more private internet.
