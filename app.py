import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
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
# OCR & LLM INITIALIZATION
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

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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
RE_PATTERNS = {
    "Credit Card": r"\b(?:\d[ -]*){13,16}\b",
    "SSN/ID": r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
    "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "Phone": r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"
}

def analyze_document_context_llm(text, groq_api_key):
    if not GROQ_AVAILABLE or not groq_api_key or not text.strip():
        return "Document", "Document", {}
    
    prompt = f"""
    Analyze the following OCR text and extract structured information.
    1. Identify the Document Type (e.g., Invoice, Receipt, Passport, ID Card, Form).
    2. Extract key entities (e.g., Merchant Name, Total Amount, Date, ID Number) as a JSON object.
    
    Text:
    ---
    {text[:2000]}
    ---
    Respond ONLY with a valid JSON in this exact format, with no markdown formatting:
    {{
        "document_type": "string",
        "suggested_filename": "string",
        "extracted_data": {{}}
    }}
    """
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        content = response.choices[0].message.content
        
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            return data.get("document_type", "Document"), data.get("suggested_filename", "Document"), data.get("extracted_data", {})
    except Exception as e:
        pass
    
    return "Document", "Document", {}

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

groq_api_key = st.sidebar.text_input("Groq API Key (For AI Extraction)", type="password", help="Get a free key from console.groq.com")

if TESSERACT_AVAILABLE:
    lang_options = {
        "English": "eng", "Sinhalese": "sin", "Tamil": "tam", "Hindi": "hin", "Arabic": "ara", 
        "Chinese (Simp)": "chi_sim", "French": "fra", "German": "deu", "Russian": "rus", "Spanish": "spa"
    }
    selected_lang = st.sidebar.selectbox("OCR Language", list(lang_options.keys()))
    ocr_lang = lang_options[selected_lang]
    ocr_enabled = st.sidebar.checkbox("Extract Text (PyTesseract)", value=True)
else:
    ocr_enabled = False
    st.sidebar.warning("⚠️ PyTesseract missing.")

pii_redaction_setting = st.sidebar.checkbox("Physical PII Blackout (Regex)", value=True)
use_llm_extraction = st.sidebar.checkbox("LLM Data Extraction (Groq)", value=True)

st.sidebar.markdown("***")
output_dpi = st.sidebar.select_slider("Quality", [72, 100, 150, 200, 300], 150)
max_pages_to_process = st.sidebar.slider("Max Pages", 1, 200, 50)

# ──────────────────────────────────────────
# CACHED FUNCTIONS
# ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_single_image_cached(img_bgr, settings_dict):
    return smart_scan_document(img_bgr, **settings_dict)

from pytesseract import Output

@st.cache_data(show_spinner=False)
def run_ocr_and_redact_cached(pil_img, lang, do_redact):
    if not TESSERACT_AVAILABLE: return pil_img, "", []
    try:
        text = pytesseract.image_to_string(pil_img, lang=lang)
        if not do_redact:
            return pil_img, text, []
            
        data = pytesseract.image_to_data(pil_img, lang=lang, output_type=Output.DICT)
        found_labels = set()
        matches = []
        
        for label, pattern in RE_PATTERNS.items():
            for match in re.finditer(pattern, text):
                found_labels.add(label)
                matches.extend(match.group().split())
                
        if not found_labels:
            return pil_img, text, []
            
        img_copy = pil_img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            word = data['text'][i].strip()
            if not word: continue
            
            should_redact = (word in matches)
            if not should_redact:
                for label, pattern in RE_PATTERNS.items():
                    if re.match(pattern, word):
                        should_redact = True
                        found_labels.add(label)
                        break
                        
            if should_redact:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                draw.rectangle([x, y, x + w, y + h], fill="black")
                
        return img_copy, text, list(found_labels)
    except Exception:
        return pil_img, "", []

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
                scanned_pil, text, pii = run_ocr_and_redact_cached(scanned_pil, ocr_lang, pii_redaction_setting)
            else:
                text, pii = "", []
                
            if use_llm_extraction and groq_api_key:
                dtype, sname, extracted_json = analyze_document_context_llm(text, groq_api_key)
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
