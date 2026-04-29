# Privacy First Smart Document Scanner

A powerful, fully offline document scanning and OCR tool built with computer vision.
No cloud APIs, no data collection, no watermarks, no subscriptions.
Your files never leave your device.

## Live Demo
Try it now on Streamlit Cloud (link in repository description)

## Features

### Page Detection and Geometry
Detects the page boundaries using K Means color clustering.
Flattens perspective using a 4 point homography transform.
Auto rotates skewed text using Hough Line analysis.
Removes dark borders left after perspective correction.

### Intelligent Cleanup
Detects skin tones (hands, fingers, thumbs) across light, medium, and dark skin types using HSV color segmentation.
Erases detected hands using the Telea Fast Marching inpainting algorithm.
Removes phone shadows and back page bleed through by estimating the background illumination field.

### Color and Lighting
Auto white balance correction using the Gray World algorithm to fix color casts from artificial lighting.
Adaptive contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).

### Text Enhancement
Non Local Means denoising for old or damaged documents.
Unsharp mask sharpening for crisp text edges.
Black and white adaptive thresholding mode for print ready output.

### Offline OCR (No Internet Required)
Extracts text from cleaned documents using Tesseract OCR.
Supports 100+ languages with zero cloud dependencies.
Generates searchable PDFs with invisible text layers.
Download extracted text as plain text files.

### Batch Processing and Export
Upload multiple files at once (PDFs and images mixed).
Download cleaned images individually as PNG or PDF.
Download all cleaned images as a single ZIP archive.
Combine all images into a single multi page PDF.
Adjustable output quality from 72 to 300 DPI.
Full PDF processing with per page preview gallery.

### What Makes This Different

| Feature | CamScanner | Adobe Scan | This Tool |
|---|---|---|---|
| No watermarks | Paid | Free | Free |
| Perspective flattening | Paid | Paid | Free |
| Hand and finger removal | No | No | Free |
| Shadow removal | Paid | Paid | Free |
| Bleed through removal | No | No | Free |
| Auto white balance | No | Paid | Free |
| Auto text rotation | Paid | Free | Free |
| Denoising old documents | No | No | Free |
| Text sharpening | No | No | Free |
| Black and white mode | Paid | Free | Free |
| Offline OCR | No (cloud) | No (cloud) | Free |
| Searchable PDF | Paid | Paid | Free |
| Batch multi file upload | Paid | Paid | Free |
| ZIP download | No | No | Free |
| Combined PDF export | No | Paid | Free |
| DPI quality control | No | No | Free |
| Privacy guarantee | No | No | Yes |
| Upload limit | 10MB | 25MB | 1GB |

## Code Structure
```
src/
  smart_scanner.py   # 11 step computer vision pipeline
  pdf_tools.py       # PDF page manipulation utilities
  vision_ocr.py      # Google Cloud Vision wrapper (optional, not used in the app)
app.py               # Streamlit web interface
packages.txt         # System dependencies for Streamlit Cloud (Tesseract)
requirements.txt     # Python dependencies
.streamlit/
  config.toml        # Dark theme and upload limit configuration
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

For OCR features, also install Tesseract:
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Technology
OpenCV, NumPy, scikit learn, PyMuPDF, Tesseract OCR, Streamlit
