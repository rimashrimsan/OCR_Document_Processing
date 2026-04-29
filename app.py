import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import sys
import os
import io
import zipfile
import gc
import datetime
import shutil
import re

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.smart_scanner import smart_scan_document

# ──────────────────────────────────────────
# TESSERACT INITIALIZATION
# ──────────────────────────────────────────
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    def find_tesseract():
        path = shutil.which("tesseract")
        if path: return path
        for p in ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]:
            if os.path.exists(p): return p
        for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "user"))]:
            if os.path.exists(p): return p
        return None
    tess_path = find_tesseract()
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
        TESSERACT_AVAILABLE = True
    elif os.name == 'posix':
        TESSERACT_AVAILABLE = True
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
except ImportError:
    TESSERACT_AVAILABLE = False

# ──────────────────────────────────────────
# CONFIGURATION & CSS
# ──────────────────────────────────────────
st.set_page_config(page_title="Smart Document Scanner Pro", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 12px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #0056b3; transform: scale(1.01); }
    @media (max-width: 640px) { .stImage > img { width: 100% !important; } .main .block-container { padding-top: 1rem; } }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .badge { padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; background-color: #28a745; color: white; display: inline-block; margin-right: 5px; }
    .badge-red { background-color: #dc3545; }
    .badge-blue { background-color: #007bff; }
</style>
""", unsafe_allow_html=True)

QR_DETECTOR = cv2.QRCodeDetector()

if "cam_active" not in st.session_state:
    st.session_state.cam_active = False

# ──────────────────────────────────────────
# SMART CONTEXT & REDACTION ENGINE
# ──────────────────────────────────────────
RE_PATTERNS = {
    "Credit Card": r"\b(?:\d[ -]*){13,16}\b",
    "SSN/ID": r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
    "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "Phone": r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"
}

KEYWORDS = {
    "Invoice": ["invoice", "bill to", "tax invoice", "amount due", "total payable"],
    "Passport/ID": ["passport", "identity card", "license", "national id", "birth date", "expiry"],
    "Receipt": ["receipt", "order #", "merchant", "subtotal", "cashier"]
}

def analyze_document_context(text):
    text_lower = text.lower()
    found_type = "Document"
    for doc_type, words in KEYWORDS.items():
        if any(w in text_lower for w in words):
            found_type = doc_type
            break
    suggested_name = found_type
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        first = lines[0].strip()
        if 3 < len(first) < 30: suggested_name = f"{found_type}_{first.replace(' ', '_')}"
    return found_type, suggested_name

def detect_pii(text):
    found = []
    for label, pattern in RE_PATTERNS.items():
        if re.search(pattern, text): found.append(label)
    return found

# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
st.sidebar.title("⚙️ Scanner Settings")
st.sidebar.markdown("**Pro Features**")
pii_redaction_setting = st.sidebar.checkbox("PII Security Detection", value=False)
smart_naming_setting = st.sidebar.checkbox("Smart Filenaming", value=True)

st.sidebar.markdown("**Page Detection**")
deskew_setting = st.sidebar.checkbox("Perspective Flattening", value=True)
crop_tol = st.sidebar.slider("Crop Tolerance", 10, 100, 50)
auto_rotate_setting = st.sidebar.checkbox("Auto Rotate Text", value=True)
border_cleanup_setting = st.sidebar.checkbox("Remove Edge Borders", value=True)

st.sidebar.markdown("**Cleanup**")
remove_hands_setting = st.sidebar.checkbox("Remove Hands (AI)", value=True)
table_detection_setting = st.sidebar.checkbox("Detect Tables", value=False)

st.sidebar.markdown("**Lighting**")
white_balance_setting = st.sidebar.checkbox("White Balance", value=True)
shadows_setting = st.sidebar.checkbox("Remove Shadows", value=True)
enhance_text_setting = st.sidebar.checkbox("Enhance Contrast", value=True)
bw_setting = st.sidebar.checkbox("Black & White", value=False)

st.sidebar.markdown("***")
st.sidebar.markdown("**Language**")
if TESSERACT_AVAILABLE:
    lang_options = {
        "English": "eng", "Sinhalese": "sin", "Tamil": "tam", "Hindi": "hin", "Arabic": "ara", 
        "Chinese (Simp)": "chi_sim", "French": "fra", "German": "deu", "Russian": "rus", "Spanish": "spa"
    }
    selected_lang = st.sidebar.selectbox("OCR Language", list(lang_options.keys()))
    ocr_lang = lang_options[selected_lang]
    ocr_enabled = st.sidebar.checkbox("Extract Text", value=True)
    searchable_pdf_enabled = st.sidebar.checkbox("Searchable PDF", value=False)
else:
    ocr_enabled = searchable_pdf_enabled = False
    st.sidebar.warning("⚠️ OCR Engine missing.")

st.sidebar.markdown("***")
output_dpi = st.sidebar.select_slider("Quality", [72, 100, 150, 200, 300], 150)
max_pages_to_process = st.sidebar.slider("Max Pages", 1, 200, 50)

# ──────────────────────────────────────────
# CACHED FUNCTIONS
# ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_single_image_cached(img_bgr, settings_dict):
    return smart_scan_document(img_bgr, **settings_dict)

@st.cache_data(show_spinner=False)
def run_ocr_cached(pil_img, lang):
    if not TESSERACT_AVAILABLE: return ""
    try: return pytesseract.image_to_string(pil_img, lang=lang)
    except Exception: return ""

@st.cache_data(show_spinner=False)
def make_searchable_pdf_page_cached(pil_img, lang):
    if not TESSERACT_AVAILABLE: return None
    try: return pytesseract.image_to_pdf_or_hocr(np.array(pil_img), extension='pdf', lang=lang)
    except Exception: return None

def pil_to_bgr(pil_img):
    if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr_img):
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

def detect_qr(img_bgr):
    try:
        data, _, _ = QR_DETECTOR.detectAndDecode(img_bgr)
        return data if data else None
    except Exception: return None

# ──────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────
st.title("📄 Smart Document Scanner Pro")
st.markdown("Global Support | AI Context | 100% Private")

st.subheader("📷 Camera Scan")
if not st.session_state.cam_active:
    if st.button("🚀 Open Camera Scanner", type="secondary"):
        st.session_state.cam_active = True
        st.rerun()
else:
    camera_photo = st.camera_input("Capture document")
    if st.button("❌ Close Camera"):
        st.session_state.cam_active = False
        st.rerun()

st.subheader("📁 Upload Files")
uploaded_files = st.file_uploader("Images or PDFs", type=["jpg", "jpeg", "png", "pdf", "tiff", "webp"], accept_multiple_files=True)

current_settings = {
    "crop_tolerance": crop_tol, "remove_hands": remove_hands_setting, "enhance_contrast": enhance_text_setting,
    "deskew": deskew_setting, "fix_shadows": shadows_setting, "auto_rotate_enabled": auto_rotate_setting,
    "denoise_enabled": False, "sharpen_enabled": False, "bw_mode": bw_setting,
    "white_balance_enabled": white_balance_setting, "border_cleanup": border_cleanup_setting,
    "detect_tables": table_detection_setting,
}

final_image_list = []
if 'camera_photo' in locals() and camera_photo: final_image_list.append(("camera_shot.jpg", Image.open(camera_photo)))
if uploaded_files:
    for f in uploaded_files:
        if not f.name.lower().endswith(".pdf"):
            try: final_image_list.append((f.name, Image.open(f)))
            except Exception: st.error(f"Error: {f.name}")
        else:
            try:
                doc = fitz.open(stream=f.read(), filetype="pdf")
                for i in range(min(len(doc), max_pages_to_process)):
                    pix = doc[i].get_pixmap(dpi=output_dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    final_image_list.append((f"{f.name}_P{i+1}.jpg", img))
                doc.close()
            except Exception: st.error(f"PDF Error: {f.name}")

if final_image_list:
    if st.button("✨ Process All Documents", type="primary"):
        all_text = []
        qr_results = []
        table_results = []
        processed_results = []
        
        total = len(final_image_list)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, image) in enumerate(final_image_list):
            status_text.text(f"Processing ({i+1}/{total}): {name}...")
            
            img_bgr = pil_to_bgr(image)
            scan_res = process_single_image_cached(img_bgr, current_settings)
            
            if table_detection_setting:
                scanned_bgr, table = scan_res
                if table is not None and np.any(table): table_results.append(f"{name}: Table Found")
            else:
                scanned_bgr, table = scan_res, None
            
            qr = detect_qr(scanned_bgr)
            if qr: qr_results.append(f"{name}: {qr}")
            
            scanned_pil = bgr_to_pil(scanned_bgr)
            text = run_ocr_cached(scanned_pil, ocr_lang) if ocr_enabled else ""
            dtype, sname = analyze_document_context(text)
            pii = detect_pii(text) if pii_redaction_setting else []
            
            processed_results.append((name, image, scanned_pil, text, qr, table, dtype, sname, pii))
            progress_bar.progress((i + 1) / total)
            del img_bgr, scanned_bgr
            gc.collect()

        status_text.text("✅ Processing Complete!")

        for name, original, cleaned, text, qr, table, dtype, sname, pii in processed_results:
            st.markdown(f"### {name}")
            badges = f"<span class='badge badge-blue'>{dtype}</span>"
            if qr: badges += f"<span class='badge'>🔍 QR Found</span>"
            if table is not None and np.any(table): badges += f"<span class='badge'>📊 Table Found</span>"
            if pii: badges += f"<span class='badge badge-red'>⚠️ PII Found: {', '.join(pii)}</span>"
            st.markdown(badges, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col1.image(original, caption="Original")
            col2.image(cleaned, caption="Cleaned")

            if smart_naming_setting: st.caption(f"Suggested: **{sname}.pdf**")

            dl_cols = st.columns(3)
            buf_p = io.BytesIO(); cleaned.save(buf_p, format="PNG")
            dl_cols[0].download_button("⬇️ PNG", buf_p.getvalue(), f"{sname}.png", "image/png", key=f"png_{name}")
            buf_d = io.BytesIO(); cleaned.save(buf_d, format="PDF", resolution=output_dpi)
            dl_cols[1].download_button("⬇️ PDF", buf_d.getvalue(), f"{sname}.pdf", "application/pdf", key=f"pdf_{name}")
            if text: all_text.append(f"--- {name} ({dtype}) ---\n{text}")

        if total > 1:
            st.divider()
            st.markdown("**Batch Downloads**")
            b_cols = st.columns(2)
            with b_cols[0]:
                z_buf = io.BytesIO()
                with zipfile.ZipFile(z_buf, "w") as zf:
                    for name, _, cleaned, _, _, _, _, sname, _ in processed_results:
                        img_buf = io.BytesIO()
                        cleaned.save(img_buf, format="PNG")
                        zf.writestr(f"{sname}.png", img_buf.getvalue())
                st.download_button("📦 All Images (ZIP)", z_buf.getvalue(), "scans.zip", "application/zip")
            with b_cols[1]:
                c_pdf = fitz.open()
                for _, _, cleaned, _, _, _, _, _, _ in processed_results:
                    buf = io.BytesIO()
                    cleaned.save(buf, format="PDF", resolution=output_dpi)
                    temp = fitz.open("pdf", buf.getvalue())
                    c_pdf.insert_pdf(temp)
                    temp.close()
                st.download_button("📄 All as Single PDF", c_pdf.write(deflate=True), "combined_scan.pdf", "application/pdf")

        if qr_results or table_results:
            with st.expander("🔍 Findings Log"):
                for r in qr_results: st.write(f"✅ QR: {r}")
                for t in table_results: st.write(f"📊 Table: {t}")

        if all_text:
            with st.expander("📝 All Extracted Text"):
                full = "\n\n".join(all_text)
                st.text_area("Results", full, height=300)
                st.download_button("📝 Download TXT", full, "data.txt")

st.markdown("***")
st.markdown("<div style='text-align: center;'><span class='badge'>🛡️ Privacy Verified: 100% Offline Processing</span></div>", unsafe_allow_html=True)
st.caption(f"Engine: {tess_path if 'tess_path' in locals() else 'System'}")
