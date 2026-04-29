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

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.smart_scanner import smart_scan_document

# Try to import Tesseract for offline OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
st.sidebar.title("⚙️ Scanner Settings")

st.sidebar.markdown("**Page Detection**")
deskew_setting = st.sidebar.checkbox(
    "Perspective Flattening", value=True,
    help="Detects the 4 corners of the page and flattens any tilt or angle into a perfect rectangle."
)
crop_tol = st.sidebar.slider(
    "Crop Tolerance", min_value=10, max_value=100, value=50,
    help="Controls how aggressively the algorithm removes background around the page."
)
auto_rotate_setting = st.sidebar.checkbox(
    "Auto Rotate Text", value=True,
    help="Detects the angle of text lines and rotates the page so text is perfectly horizontal."
)
border_cleanup_setting = st.sidebar.checkbox(
    "Remove Edge Borders", value=True,
    help="Trims dark borders that appear after perspective correction or from scanner glass edges."
)

st.sidebar.markdown("**Cleanup**")
remove_hands_setting = st.sidebar.checkbox(
    "Remove Hands (AI Powered)", value=True,
    help="Uses MediaPipe AI to detect hand landmarks for all skin tones and erases them seamlessly."
)

st.sidebar.markdown("**Lighting and Quality**")
white_balance_setting = st.sidebar.checkbox(
    "Auto White Balance", value=True,
    help="Corrects color casts from yellow lamps or blue screens using the Gray World algorithm."
)
shadows_setting = st.sidebar.checkbox(
    "Remove Shadows and Bleed Through", value=True,
    help="Computes the background light field and subtracts phone shadows or text bleeding through from the back page."
)
enhance_text_setting = st.sidebar.checkbox(
    "Enhance Text Contrast", value=True,
    help="Uses Adaptive Histogram Equalization to boost the difference between text and paper."
)
denoise_setting = st.sidebar.checkbox(
    "Denoise Old Documents", value=False,
    help="Removes speckle noise from aged, photocopied, or damaged pages using Non Local Means filtering."
)
sharpen_setting = st.sidebar.checkbox(
    "Sharpen Text", value=False,
    help="Applies an unsharp mask to make text edges crisper."
)
bw_setting = st.sidebar.checkbox(
    "Black and White Mode", value=False,
    help="Converts to a clean monochrome scan using adaptive thresholding. Ideal for printing or faxing."
)

st.sidebar.markdown("***")

# OCR Section
st.sidebar.markdown("**Text Extraction (OCR)**")
if TESSERACT_AVAILABLE:
    ocr_enabled = st.sidebar.checkbox(
        "Extract Text (Tesseract OCR)", value=False,
        help="Runs offline OCR on each page to extract readable text. No internet required."
    )
    
    # User friendly language selection
    lang_map = {
        "English": "eng",
        "Sinhalese": "sin",
        "Tamil": "tam",
        "Hindi": "hin",
        "German": "deu",
        "French": "fra",
        "Spanish": "spa",
        "Arabic": "ara",
        "Custom Code": "custom"
    }
    selected_lang_name = st.sidebar.selectbox("OCR Language", options=list(lang_map.keys()), index=0)
    
    if selected_lang_name == "Custom Code":
        ocr_lang = st.sidebar.text_input("Enter Language Code(s)", value="eng", help="Use codes like 'eng+sin'")
    else:
        ocr_lang = lang_map[selected_lang_name]

    searchable_pdf_enabled = st.sidebar.checkbox(
        "Generate Searchable PDF", value=False,
        help="Creates a PDF with an invisible text layer so you can search and copy text from the document."
    )
else:
    ocr_enabled = False
    searchable_pdf_enabled = False
    st.sidebar.warning("Install pytesseract and Tesseract OCR for offline text extraction.")

st.sidebar.markdown("***")

# Output and Safety
st.sidebar.markdown("**File Naming & Quality**")
file_prefix = st.sidebar.text_input("File Prefix", value="Scan", help="Prefix for your saved files.")
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
add_date = st.sidebar.checkbox("Add Date to Filename", value=True)
final_prefix = f"{file_prefix}_{current_date}_" if add_date else f"{file_prefix}_"

output_dpi = st.sidebar.select_slider(
    "Output Quality (DPI)", options=[72, 100, 150, 200, 300], value=150,
    help="Higher DPI = better quality but larger file size. 150 is good for reading, 300 for printing."
)
max_pages_to_process = st.sidebar.slider(
    "Max Pages (Safety)", min_value=1, max_value=200, value=50,
    help="Limits the number of pages processed per PDF to prevent memory crashes on the server."
)

st.sidebar.markdown("***")
st.sidebar.info("💡 If the algorithm crops too much, lower the Crop Tolerance. If colors look wrong, try toggling Auto White Balance.")

# ──────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────
st.title("📄 Privacy First Smart Document Scanner")
st.markdown("""
**100% Offline. 100% Private. Zero cloud uploads. No watermarks. No subscriptions.**

Most document scanner apps upload your files to remote servers, add watermarks to free versions, 
and lock advanced features behind paywalls. This tool processes everything locally in your browser.
Your files are never stored, never logged, and never sent to any external service.
""")

# Feature highlights
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    st.markdown("**🔧 11 Step Pipeline**")
    st.caption("Perspective, rotation, shadows, hands, contrast, and more")
with col_f2:
    st.markdown("**🖐️ AI Hand Removal**")
    st.caption("Advanced MediaPipe detection for all skin tones")
with col_f3:
    st.markdown("**📸 Mobile Camera**")
    st.caption("Scan directly from your phone's camera in browser")
with col_f4:
    st.markdown("**🔒 100% Privacy**")
    st.caption("Zero data collection or cloud processing")

st.divider()

# ──────────────────────────────────────────
# INPUTS (CAMERA + UPLOAD)
# ──────────────────────────────────────────
st.subheader("📷 Scan with Camera")
show_camera = st.checkbox("Open Camera Scanner", value=False, help="Enable this to take a photo directly from your device.")

camera_photo = None
if show_camera:
    camera_photo = st.camera_input("Take a photo of a document", help="Opens your phone or webcam camera directly.")

st.subheader("📁 Upload Files")
uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["jpg", "jpeg", "png", "pdf", "tiff", "tif", "bmp", "webp"],
    accept_multiple_files=True,
    help="Drag and drop multiple files. Supports PDF, JPG, PNG, TIFF, BMP, and WebP."
)


def process_single_image(img_bgr):
    """Run the full scanner pipeline on a single BGR image."""
    return smart_scan_document(
        img_bgr,
        crop_tolerance=crop_tol,
        remove_hands=remove_hands_setting,
        enhance_contrast=enhance_text_setting,
        deskew=deskew_setting,
        fix_shadows=shadows_setting,
        auto_rotate_enabled=auto_rotate_setting,
        denoise_enabled=denoise_setting,
        sharpen_enabled=sharpen_setting,
        bw_mode=bw_setting,
        white_balance_enabled=white_balance_setting,
        border_cleanup=border_cleanup_setting,
    )


def pil_to_bgr(pil_img):
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_img):
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


def run_ocr(pil_img):
    if not TESSERACT_AVAILABLE:
        return ""
    return pytesseract.image_to_string(pil_img, lang=ocr_lang)


def make_searchable_pdf_page(pil_img):
    """Generate a searchable PDF page with invisible text layer."""
    if not TESSERACT_AVAILABLE:
        return None
    return pytesseract.image_to_pdf_or_hocr(np.array(pil_img), extension='pdf', lang=ocr_lang)


# Combine camera and uploaded images
final_image_list = []
if camera_photo:
    final_image_list.append(("camera_shot.jpg", Image.open(camera_photo)))

if uploaded_files:
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
    image_files = [f for f in uploaded_files if not f.name.lower().endswith(".pdf")]
    
    # Process Images First
    for img_file in image_files:
        try:
            final_image_list.append((img_file.name, Image.open(img_file)))
        except Exception:
            st.error(f"Could not open image: {img_file.name}")

    all_extracted_text = []

    # ──────────────────────────────────────
    # PROCESS PDFs
    # ──────────────────────────────────────
    for pdf_file in pdf_files:
        st.subheader(f"📑 {pdf_file.name}")
        try:
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception:
            st.error(f"Could not open {pdf_file.name}. The file may be corrupted or password protected.")
            continue

        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages_to_process)
        
        if total_pages > max_pages_to_process:
            st.warning(f"File has {total_pages} pages. Processing only the first {max_pages_to_process} to prevent memory crash.")

        # Choose output method
        if searchable_pdf_enabled and TESSERACT_AVAILABLE:
            searchable_pdf_parts = []
        else:
            output_pdf = fitz.open()

        progress = st.progress(0, text="Starting...")

        # Show preview columns (up to 6 pages)
        num_preview = min(pages_to_process, 6)
        preview_cols = st.columns(num_preview) if num_preview > 0 else []
        pdf_text_parts = []

        for i in range(pages_to_process):
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=output_dpi)

                mode = "RGBA" if pix.alpha else "RGB"
                img_pil = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if mode == "RGBA":
                    img_pil = img_pil.convert("RGB")

                img_bgr = pil_to_bgr(img_pil)
                scanned_bgr = process_single_image(img_bgr)
                scanned_pil = bgr_to_pil(scanned_bgr)

                # Preview
                if i < num_preview:
                    with preview_cols[i]:
                        st.image(scanned_pil, caption=f"Page {i+1}", use_container_width=True)

                # OCR text extraction
                if ocr_enabled:
                    page_text = run_ocr(scanned_pil)
                    pdf_text_parts.append(f"--- Page {i+1} ---\n{page_text}")

                # Build output PDF
                if searchable_pdf_enabled and TESSERACT_AVAILABLE:
                    pdf_page_bytes = make_searchable_pdf_page(scanned_pil)
                    if pdf_page_bytes:
                        searchable_pdf_parts.append(pdf_page_bytes)
                else:
                    buf = io.BytesIO()
                    scanned_pil.save(buf, format="PDF", resolution=output_dpi)
                    temp_pdf = fitz.open("pdf", buf.getvalue())
                    output_pdf.insert_pdf(temp_pdf)
                    temp_pdf.close()
                
                # Explicit memory cleanup per page
                del img_bgr, scanned_bgr, scanned_pil
                gc.collect()

            except Exception as e:
                st.error(f"Error on page {i+1}: {str(e)}")

            progress.progress((i+1) / pages_to_process, text=f"Processed page {i+1} of {pages_to_process}")

        # Finalize PDF
        if searchable_pdf_enabled and TESSERACT_AVAILABLE and searchable_pdf_parts:
            # Merge searchable PDF pages
            merged_pdf = fitz.open()
            for part in searchable_pdf_parts:
                temp = fitz.open("pdf", part)
                merged_pdf.insert_pdf(temp)
                temp.close()
            final_pdf_bytes = merged_pdf.write(deflate=True, garbage=3)
            merged_pdf.close()
        else:
            final_pdf_bytes = output_pdf.write(deflate=True, garbage=3)
            output_pdf.close()

        doc.close()
        st.success(f"Finished processing {pages_to_process} pages.")

        # Download buttons row
        dl_cols = st.columns(3)
        with dl_cols[0]:
            label = "⬇️ Download Searchable PDF" if (searchable_pdf_enabled and TESSERACT_AVAILABLE) else "⬇️ Download Cleaned PDF"
            st.download_button(
                label=label,
                data=final_pdf_bytes,
                file_name=f"{final_prefix}{os.path.splitext(pdf_file.name)[0]}.pdf",
                mime="application/pdf",
                key=f"pdf_dl_{pdf_file.name}"
            )

        if ocr_enabled and pdf_text_parts:
            full_text = "\n\n".join(pdf_text_parts)
            all_extracted_text.append(full_text)
            with dl_cols[1]:
                st.download_button(
                    label="📝 Download as Text",
                    data=full_text,
                    file_name=f"{os.path.splitext(pdf_file.name)[0]}.txt",
                    mime="text/plain",
                    key=f"txt_dl_{pdf_file.name}"
                )

        st.divider()

# ──────────────────────────────────────────
# PROCESS IMAGES (Including Camera)
# ──────────────────────────────────────────
if final_image_list:
    st.subheader(f"🖼️ Processing {len(final_image_list)} Image(s)")
    cleaned_images = []
    all_extracted_text_images = []

    for name, image in final_image_list:
        try:
            img_bgr = pil_to_bgr(image)
            scanned_bgr = process_single_image(img_bgr)
            scanned_pil = bgr_to_pil(scanned_bgr)
            cleaned_images.append((name, image, scanned_pil))

            if ocr_enabled:
                page_text = run_ocr(scanned_pil)
                all_extracted_text_images.append(f"--- {name} ---\n{page_text}")
            
            # Memory cleanup
            del img_bgr, scanned_bgr
            gc.collect()
        except Exception:
            st.error(f"Could not process {name}")

    # Before/After display
    for name, original, cleaned in cleaned_images:
        st.markdown(f"**{name}**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original", use_container_width=True)
        with col2:
            st.image(cleaned, caption="Cleaned", use_container_width=True)

        # Per image downloads
        dl_cols = st.columns(3)
        with dl_cols[0]:
            buf = io.BytesIO()
            cleaned.save(buf, format="PNG")
            st.download_button(
                label=f"⬇️ Download PNG",
                data=buf.getvalue(),
                file_name=f"{final_prefix}{name}",
                mime="image/png",
                key=f"img_png_{name}"
            )
        with dl_cols[1]:
            buf_pdf = io.BytesIO()
            cleaned.save(buf_pdf, format="PDF", resolution=output_dpi)
            st.download_button(
                label=f"⬇️ Download as PDF",
                data=buf_pdf.getvalue(),
                file_name=f"{final_prefix}{os.path.splitext(name)[0]}.pdf",
                mime="application/pdf",
                key=f"img_pdf_{name}"
            )

    # Batch ZIP download
    if len(cleaned_images) > 1:
        st.markdown("**Batch Download**")
        zip_cols = st.columns(2)

        # ZIP of PNGs
        with zip_cols[0]:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, _, cleaned in cleaned_images:
                    img_buf = io.BytesIO()
                    cleaned.save(img_buf, format="PNG")
                    zf.writestr(f"cleaned_{name}", img_buf.getvalue())
            st.download_button(
                label="📦 Download All Images (ZIP)",
                data=zip_buf.getvalue(),
                file_name="cleaned_images.zip",
                mime="application/zip",
                key="zip_png_dl"
            )

        # Combined PDF of all images
        with zip_cols[1]:
            combined_pdf = fitz.open()
            for name, _, cleaned in cleaned_images:
                buf = io.BytesIO()
                cleaned.save(buf, format="PDF", resolution=output_dpi)
                temp = fitz.open("pdf", buf.getvalue())
                combined_pdf.insert_pdf(temp)
                temp.close()
            combined_bytes = combined_pdf.write(deflate=True, garbage=3)
            combined_pdf.close()
            st.download_button(
                label="📄 Download All as Single PDF",
                data=combined_bytes,
                file_name=f"{final_prefix}batch_all.pdf",
                mime="application/pdf",
                key="combined_pdf_dl"
            )

    st.divider()

    # ──────────────────────────────────────
    # COMBINED OCR OUTPUT
    # ──────────────────────────────────────
    if ocr_enabled and (all_extracted_text or all_extracted_text_images):
        with st.expander("📝 View All Extracted Text", expanded=False):
            combined_text = "\n\n".join(all_extracted_text + all_extracted_text_images)
            st.text_area("Extracted Text", combined_text, height=400)
            st.download_button(
                label="📝 Download Complete Text File",
                data=combined_text,
                file_name="extracted_text.txt",
                mime="text/plain",
                key="all_txt_dl"
            )

# ──────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────
st.markdown("***")
st.caption("Built with OpenCV, NumPy, MediaPipe AI, PyMuPDF, and Tesseract OCR. No data leaves your browser.")
