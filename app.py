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
from concurrent.futures import ThreadPoolExecutor

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
st.set_page_config(page_title="Smart Document Scanner", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 12px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #0056b3; transform: scale(1.01); }
    @media (max-width: 640px) { .stImage > img { width: 100% !important; } .main .block-container { padding-top: 1rem; } }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

QR_DETECTOR = cv2.QRCodeDetector()

# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
st.sidebar.title("⚙️ Scanner Settings")
st.sidebar.markdown("**Page Detection**")
deskew_setting = st.sidebar.checkbox("Perspective Flattening", value=True)
crop_tol = st.sidebar.slider("Crop Tolerance", 10, 100, 50)
auto_rotate_setting = st.sidebar.checkbox("Auto Rotate Text", value=True)
border_cleanup_setting = st.sidebar.checkbox("Remove Edge Borders", value=True)

st.sidebar.markdown("**Cleanup**")
remove_hands_setting = st.sidebar.checkbox("Remove Hands (AI Powered)", value=True)

st.sidebar.markdown("**Lighting and Quality**")
white_balance_setting = st.sidebar.checkbox("Auto White Balance", value=True)
shadows_setting = st.sidebar.checkbox("Remove Shadows", value=True)
enhance_text_setting = st.sidebar.checkbox("Enhance Contrast", value=True)
denoise_setting = st.sidebar.checkbox("Denoise", value=False)
sharpen_setting = st.sidebar.checkbox("Sharpen", value=False)
bw_setting = st.sidebar.checkbox("Black and White Mode", value=False)

st.sidebar.markdown("***")
st.sidebar.markdown("**Text Extraction (OCR)**")
if TESSERACT_AVAILABLE:
    ocr_enabled = st.sidebar.checkbox("Extract Text (OCR)", value=False)
    lang_map = {"English": "eng", "Sinhalese": "sin", "Tamil": "tam", "Hindi": "hin", "German": "deu", "French": "fra", "Spanish": "spa", "Arabic": "ara", "Custom": "custom"}
    selected_lang_name = st.sidebar.selectbox("Language", list(lang_map.keys()))
    ocr_lang = st.sidebar.text_input("Code", "eng") if selected_lang_name == "Custom" else lang_map[selected_lang_name]
    searchable_pdf_enabled = st.sidebar.checkbox("Searchable PDF", value=False)
else:
    ocr_enabled = searchable_pdf_enabled = False
    st.sidebar.warning("⚠️ OCR Engine not found. Reboot in Dashboard.")

st.sidebar.markdown("***")
file_prefix = st.sidebar.text_input("File Prefix", "Scan")
add_date = st.sidebar.checkbox("Add Date", value=True)
final_prefix = f"{file_prefix}_{datetime.datetime.now().strftime('%Y-%m-%d')}_" if add_date else f"{file_prefix}_"
output_dpi = st.sidebar.select_slider("DPI", [72, 100, 150, 200, 300], 150)
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
    except Exception: return "OCR Error"

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
st.title("📄 Privacy First Smart Scanner")
st.markdown("100% Offline. 100% Private. Zero cloud uploads.")

st.subheader("📷 Camera Scan")
show_camera = st.checkbox("Open Camera", value=False)
camera_photo = st.camera_input("Take photo") if show_camera else None

st.subheader("📁 Upload")
uploaded_files = st.file_uploader("Upload images/PDFs", type=["jpg", "jpeg", "png", "pdf", "tiff", "webp"], accept_multiple_files=True)

current_settings = {
    "crop_tolerance": crop_tol, "remove_hands": remove_hands_setting, "enhance_contrast": enhance_text_setting,
    "deskew": deskew_setting, "fix_shadows": shadows_setting, "auto_rotate_enabled": auto_rotate_setting,
    "denoise_enabled": denoise_setting, "sharpen_enabled": sharpen_setting, "bw_mode": bw_setting,
    "white_balance_enabled": white_balance_setting, "border_cleanup": border_cleanup_setting,
}

final_image_list = []
if camera_photo: final_image_list.append(("camera_shot.jpg", Image.open(camera_photo)))
if uploaded_files:
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
    image_files = [f for f in uploaded_files if not f.name.lower().endswith(".pdf")]
    for img_file in image_files:
        try: final_image_list.append((img_file.name, Image.open(img_file)))
        except Exception: st.error(f"Error opening {img_file.name}")

if (uploaded_files or camera_photo):
    if st.button("🚀 Start Scanning / Apply Settings", type="primary"):
        all_extracted_text = []
        qr_results = []

        # PROCESS PDFs
        if 'pdf_files' in locals() and pdf_files:
            for pdf_file in pdf_files:
                st.subheader(f"📑 {pdf_file.name}")
                try:
                    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                    pages_to_process = min(len(doc), max_pages_to_process)
                    searchable_pdf_parts = [] if (searchable_pdf_enabled and TESSERACT_AVAILABLE) else None
                    output_pdf = fitz.open() if not searchable_pdf_parts else None
                    progress = st.progress(0)
                    pdf_text_parts = []
                    
                    for i in range(pages_to_process):
                        pix = doc[i].get_pixmap(dpi=output_dpi)
                        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img_bgr = pil_to_bgr(img_pil)
                        scanned_bgr = process_single_image_cached(img_bgr, current_settings)
                        scanned_pil = bgr_to_pil(scanned_bgr)
                        
                        qr_data = detect_qr(scanned_bgr)
                        if qr_data: qr_results.append(f"{pdf_file.name} P{i+1}: {qr_data}")

                        if ocr_enabled:
                            text = run_ocr_cached(scanned_pil, ocr_lang)
                            pdf_text_parts.append(f"--- Page {i+1} ---\n{text}")

                        if searchable_pdf_parts is not None:
                            pdf_page_bytes = make_searchable_pdf_page_cached(scanned_pil, ocr_lang)
                            if pdf_page_bytes: searchable_pdf_parts.append(pdf_page_bytes)
                        else:
                            buf = io.BytesIO()
                            scanned_pil.save(buf, format="PDF", resolution=output_dpi)
                            temp_pdf = fitz.open("pdf", buf.getvalue())
                            output_pdf.insert_pdf(temp_pdf)
                            temp_pdf.close()
                        progress.progress((i+1)/pages_to_process)
                    
                    if searchable_pdf_parts:
                        merged = fitz.open()
                        for p in searchable_pdf_parts: 
                            t = fitz.open("pdf", p)
                            merged.insert_pdf(t)
                            t.close()
                        final_bytes = merged.write(deflate=True, garbage=3)
                        merged.close()
                    else:
                        final_bytes = output_pdf.write(deflate=True, garbage=3)
                        output_pdf.close()
                    
                    st.download_button(f"⬇️ Download PDF ({pdf_file.name})", final_bytes, f"{final_prefix}{pdf_file.name}", "application/pdf")
                    if ocr_enabled: all_extracted_text.append("\n\n".join(pdf_text_parts))
                    doc.close()
                except Exception as e: st.error(f"Error: {str(e)}")

        # PROCESS IMAGES
        if final_image_list:
            st.subheader(f"🖼️ Images")
            
            def process_image_task(item):
                name, image = item
                img_bgr = pil_to_bgr(image)
                scanned_bgr = process_single_image_cached(img_bgr, current_settings)
                scanned_pil = bgr_to_pil(scanned_bgr)
                qr_data = detect_qr(scanned_bgr)
                text = run_ocr_cached(scanned_pil, ocr_lang) if ocr_enabled else ""
                return (name, image, scanned_pil, text, qr_data)

            # Parallel processing for images
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_image_task, final_image_list))

            for name, original, cleaned, text, qr in results:
                st.markdown(f"**{name}**")
                if qr: st.success(f"🔍 Found QR Code: {qr}")
                col1, col2 = st.columns(2)
                col1.image(original, caption="Original")
                col2.image(cleaned, caption="Cleaned")
                
                dl_cols = st.columns(3)
                with dl_cols[0]:
                    buf = io.BytesIO()
                    cleaned.save(buf, format="PNG")
                    st.download_button(f"⬇️ PNG", buf.getvalue(), f"cleaned_{name}", "image/png")
                with dl_cols[1]:
                    buf_pdf = io.BytesIO()
                    cleaned.save(buf_pdf, format="PDF", resolution=output_dpi)
                    st.download_button(f"⬇️ PDF", buf_pdf.getvalue(), f"cleaned_{os.path.splitext(name)[0]}.pdf", "application/pdf")
                
                if text: all_extracted_text.append(f"--- {name} ---\n{text}")

        # Batch Downloads for multiple images
        if len(final_image_list) > 1:
            st.divider()
            st.markdown("**Batch Downloads**")
            b_cols = st.columns(2)
            with b_cols[0]:
                z_buf = io.BytesIO()
                with zipfile.ZipFile(z_buf, "w") as zf:
                    for name, _, cleaned, _, _ in results:
                        img_buf = io.BytesIO()
                        cleaned.save(img_buf, format="PNG")
                        zf.writestr(f"cleaned_{name}", img_buf.getvalue())
                st.download_button("📦 Download All Images (ZIP)", z_buf.getvalue(), "cleaned_images.zip", "application/zip")
            with b_cols[1]:
                c_pdf = fitz.open()
                for _, _, cleaned, _, _ in results:
                    buf = io.BytesIO()
                    cleaned.save(buf, format="PDF", resolution=output_dpi)
                    temp = fitz.open("pdf", buf.getvalue())
                    c_pdf.insert_pdf(temp)
                    temp.close()
                st.download_button("📄 Download All as Single PDF", c_pdf.write(deflate=True), "batch_scan.pdf", "application/pdf")

        if qr_results:
            with st.expander("🔍 Found QR/Barcodes"):
                for r in qr_results: st.write(r)

        if all_extracted_text:
            with st.expander("📝 Extracted Text"):
                full = "\n\n".join(all_extracted_text)
                st.text_area("Text", full, height=300)
                st.download_button("📝 Download Text File", full, "scanned_text.txt")

st.markdown("***")
st.caption(f"Engine Info: {tess_path if 'tess_path' in locals() else 'System Path'}")
