import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
from pytesseract import Output
import sys
import os
import io
import zipfile
import gc
import datetime
import shutil
import re
import json

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import from root directly
try:
    from smart_scanner import smart_scan_document
except ImportError:
    from src.smart_scanner import smart_scan_document

# ──────────────────────────────────────────
# OCR, NLP & LLM INITIALIZATION
# ──────────────────────────────────────────
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    # Initialize PaddleOCR
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

PRESIDIO_AVAILABLE = False
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

OLLAMA_AVAILABLE = False
try:
    import ollama
    # Test if Ollama is running
    try:
        ollama.list()
        OLLAMA_AVAILABLE = True
    except Exception:
        OLLAMA_AVAILABLE = False
except ImportError:
    OLLAMA_AVAILABLE = False

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
# SMART CONTEXT & EXTRACTION ENGINE
# ──────────────────────────────────────────
def analyze_document_context_llm(text):
    if not OLLAMA_AVAILABLE or not text.strip():
        return "Document", "Document", {}
    
    prompt = f"""
    Analyze the following OCR text and extract structured information.
    1. Identify the Document Type (e.g., Invoice, Receipt, Passport, ID Card, Form).
    2. Extract key entities (e.g., Merchant Name, Total Amount, Date, ID Number) as a JSON object.
    
    Text:
    ---
    {text[:2000]}
    ---
    Respond ONLY with a valid JSON in this exact format:
    {{
        "document_type": "string",
        "suggested_filename": "string",
        "extracted_data": {{}}
    }}
    """
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content']
        # Try to parse JSON from response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            return data.get("document_type", "Document"), data.get("suggested_filename", "Document"), data.get("extracted_data", {})
    except Exception as e:
        pass
    
    # Fallback to simple keyword logic if LLM fails
    text_lower = text.lower()
    found_type = "Document"
    for doc_type, words in {"Invoice": ["invoice", "tax"], "Receipt": ["receipt", "total"], "ID": ["passport", "id"]}.items():
        if any(w in text_lower for w in words): found_type = doc_type
    return found_type, found_type, {}

# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
st.sidebar.title("⚙️ Scanner Settings")
st.sidebar.markdown("**Pro Features**")
pii_redaction_setting = st.sidebar.checkbox("Physical PII Blackout", value=False)
smart_naming_setting = st.sidebar.checkbox("Smart Filenaming", value=True)

st.sidebar.markdown("**Page Detection**")
deskew_setting = st.sidebar.checkbox("Perspective Flattening", value=True)
crop_tol = st.sidebar.slider("Crop Tolerance", 10, 100, 50)
auto_rotate_setting = st.sidebar.checkbox("Auto Rotate Text", value=True)
border_cleanup_setting = st.sidebar.checkbox("Remove Edge Borders", value=True)

st.sidebar.markdown("**Cleanup**")
remove_hands_setting = st.sidebar.checkbox("Remove Hands (Auto)", value=True)
table_detection_setting = st.sidebar.checkbox("Detect Tables", value=False)

st.sidebar.markdown("**Lighting**")
white_balance_setting = st.sidebar.checkbox("White Balance", value=True)
shadows_setting = st.sidebar.checkbox("Remove Shadows", value=True)
enhance_text_setting = st.sidebar.checkbox("Enhance Contrast", value=True)
bw_setting = st.sidebar.checkbox("Black & White", value=False)

st.sidebar.markdown("***")
st.sidebar.markdown("**AI Capabilities**")
if PADDLEOCR_AVAILABLE:
    ocr_enabled = st.sidebar.checkbox("Extract Text (PaddleOCR)", value=True)
else:
    ocr_enabled = False
    st.sidebar.warning("⚠️ PaddleOCR missing.")

if PRESIDIO_AVAILABLE:
    pii_redaction_setting = st.sidebar.checkbox("Physical PII Blackout (NLP)", value=True)
else:
    pii_redaction_setting = False
    st.sidebar.warning("⚠️ Presidio NLP missing.")

if OLLAMA_AVAILABLE:
    use_llm_extraction = st.sidebar.checkbox("LLM Data Extraction", value=True)
else:
    use_llm_extraction = False
    st.sidebar.warning("⚠️ Ollama (Llama 3) not running.")

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
def run_ocr_and_redact_cached(img_bgr, do_redact):
    if not PADDLEOCR_AVAILABLE: return bgr_to_pil(img_bgr), "", []
    try:
        results = ocr_engine.ocr(img_bgr, cls=True)
        if not results or not results[0]:
            return bgr_to_pil(img_bgr), "", []
            
        lines = results[0]
        text_full = "\n".join([line[1][0] for line in lines])
        pil_img = bgr_to_pil(img_bgr)
        
        if not do_redact or not PRESIDIO_AVAILABLE:
            return pil_img, text_full, []
            
        # Presidio Context-Aware Detection
        analyzer_results = analyzer.analyze(text=text_full, language="en")
        if not analyzer_results:
            return pil_img, text_full, []
            
        found_labels = list(set([res.entity_type for res in analyzer_results]))
        
        # We need to map Presidio text matches back to PaddleOCR bounding boxes
        img_copy = pil_img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        for res in analyzer_results:
            sensitive_word = text_full[res.start:res.end].strip()
            if not sensitive_word: continue
            
            # Find which bounding box this word belongs to
            for line in lines:
                bbox, (line_text, score) = line
                if sensitive_word in line_text:
                    box = np.array(bbox).astype(np.int32)
                    xmin, ymin = min(box[:, 0]), min(box[:, 1])
                    xmax, ymax = max(box[:, 0]), max(box[:, 1])
                    draw.rectangle([xmin, ymin, xmax, ymax], fill="black")
                    
        return img_copy, text_full, found_labels
    except Exception as e:
        print(f"OCR Error: {e}")
        return bgr_to_pil(img_bgr), "", []

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
    "bw_mode": bw_setting, "white_balance_enabled": white_balance_setting, "border_cleanup": border_cleanup_setting,
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

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
    st.session_state.final_pdf_bytes = None
    st.session_state.final_text_str = None
    st.session_state.qr_results = []
    st.session_state.table_results = []

if final_image_list:
    if st.button("✨ Process All Documents", type="primary"):
        all_text_list = []
        qr_results = []
        table_results = []
        processed_results = []
        
        total = len(final_image_list)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        c_pdf = fitz.open() # Create a single PDF document
        
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
            
            if ocr_enabled:
                scanned_pil, text, pii = run_ocr_and_redact_cached(scanned_bgr, pii_redaction_setting)
            else:
                scanned_pil = bgr_to_pil(scanned_bgr)
                text, pii = "", []
                
            if use_llm_extraction:
                dtype, sname, extracted_json = analyze_document_context_llm(text)
            else:
                dtype, sname, extracted_json = "Document", "Document", {}
            
            processed_results.append((name, image, scanned_pil, text, qr, table, dtype, sname, pii, extracted_json))
            
            if text:
                all_text_list.append(f"--- {name} ({dtype}) ---\n{text}\nData: {json.dumps(extracted_json)}")
                
            # Add to the combined PDF
            buf = io.BytesIO()
            scanned_pil.save(buf, format="PDF", resolution=output_dpi)
            temp = fitz.open("pdf", buf.getvalue())
            c_pdf.insert_pdf(temp)
            temp.close()
            
            progress_bar.progress((i + 1) / total)
            del img_bgr, scanned_bgr
            gc.collect()

        status_text.text("✅ Processing Complete!")
        
        # Save finalized strings and bytes to session state so they don't rebuild on rerun
        st.session_state.processed_data = processed_results
        st.session_state.final_pdf_bytes = c_pdf.write(deflate=True)
        st.session_state.final_text_str = "\n\n".join(all_text_list) if all_text_list else None
        st.session_state.qr_results = qr_results
        st.session_state.table_results = table_results
        
        c_pdf.close()

if st.session_state.processed_data:
    for name, original, cleaned, text, qr, table, dtype, sname, pii, extracted_json in st.session_state.processed_data:
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
        
        if extracted_json:
            with st.expander("🤖 AI Data Extraction"):
                st.json(extracted_json)

    st.divider()
    st.markdown("**📥 Downloads**")
    
    dl_cols = st.columns(2)
    
    # Complete PDF Download
    with dl_cols[0]:
        if st.session_state.final_pdf_bytes:
            st.download_button("📄 Download Complete PDF", st.session_state.final_pdf_bytes, "combined_scan.pdf", "application/pdf", use_container_width=True)

    # All Text Download
    with dl_cols[1]:
        if st.session_state.final_text_str:
            st.download_button("📝 Download All Text (TXT)", st.session_state.final_text_str, "extracted_text.txt", "text/plain", use_container_width=True)
        else:
            st.button("📝 No Text Extracted", disabled=True, use_container_width=True)

    if st.session_state.qr_results or st.session_state.table_results:
        with st.expander("🔍 Findings Log"):
            for r in st.session_state.qr_results: st.write(f"✅ QR: {r}")
            for t in st.session_state.table_results: st.write(f"📊 Table: {t}")

    if st.session_state.final_text_str:
        with st.expander("📝 View Extracted Text"):
            st.text_area("Results", st.session_state.final_text_str, height=300)

st.markdown("***")
st.markdown("<div style='text-align: center;'><span class='badge'>🛡️ Privacy Verified: 100% Offline Processing</span></div>", unsafe_allow_html=True)
st.caption(f"Engine: {tess_path if 'tess_path' in locals() else 'System'}")
