import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import sys
import os
import io
import zipfile

sys.path.append(os.path.dirname(__file__))
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
    "Remove Hands and Fingers", value=True,
    help="Detects skin tones using HSV color analysis and seamlessly erases hands holding the book."
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
    ocr_lang = st.sidebar.text_input(
        "OCR Language Code", value="eng",
        help="Examples: eng, deu, fra, sin, ara. Combine with + like eng+deu."
    )
    searchable_pdf_enabled = st.sidebar.checkbox(
        "Generate Searchable PDF", value=False,
        help="Creates a PDF with an invisible text layer so you can search and copy text from the document."
    )
else:
    ocr_enabled = False
    searchable_pdf_enabled = False
    st.sidebar.warning("Install pytesseract and Tesseract OCR for offline text extraction.")

st.sidebar.markdown("***")

# Output Format
st.sidebar.markdown("**Output Format**")
output_dpi = st.sidebar.select_slider(
    "Output Quality (DPI)", options=[72, 100, 150, 200, 300], value=150,
    help="Higher DPI = better quality but larger file size. 150 is good for reading, 300 for printing."
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
    st.markdown("**🖐️ Hand Removal**")
    st.caption("Erases fingers holding the book using skin detection")
with col_f3:
    st.markdown("**📝 Offline OCR**")
    st.caption("Extract text in 100+ languages without internet")
with col_f4:
    st.markdown("**📦 Batch Processing**")
    st.caption("Upload multiple files and download as ZIP")

st.divider()

# ──────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────
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


if uploaded_files:
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
    image_files = [f for f in uploaded_files if not f.name.lower().endswith(".pdf")]

    all_extracted_text = []

    # ──────────────────────────────────────
    # PROCESS PDFs
    # ──────────────────────────────────────
    for pdf_file in pdf_files:
        st.subheader(f"📑 {pdf_file.name}")

        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        # Choose output method
        if searchable_pdf_enabled and TESSERACT_AVAILABLE:
            searchable_pdf_parts = []
        else:
            output_pdf = fitz.open()

        progress = st.progress(0, text="Starting...")

        # Show preview columns (up to 6 pages)
        num_preview = min(total_pages, 6)
        preview_cols = st.columns(num_preview) if num_preview > 0 else []
        pdf_text_parts = []

        for i in range(total_pages):
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

            progress.progress((i+1) / total_pages, text=f"Processed page {i+1} of {total_pages}")

        # Finalize PDF
        if searchable_pdf_enabled and TESSERACT_AVAILABLE and searchable_pdf_parts:
            # Merge searchable PDF pages
            merged_pdf = fitz.open()
            for part in searchable_pdf_parts:
                temp = fitz.open("pdf", part)
                merged_pdf.insert_pdf(temp)
                temp.close()
            final_pdf_bytes = merged_pdf.write()
            merged_pdf.close()
        else:
            final_pdf_bytes = output_pdf.write()
            output_pdf.close()

        doc.close()
        st.success(f"Finished processing {total_pages} pages.")

        # Download buttons row
        dl_cols = st.columns(3)
        with dl_cols[0]:
            label = "⬇️ Download Searchable PDF" if (searchable_pdf_enabled and TESSERACT_AVAILABLE) else "⬇️ Download Cleaned PDF"
            st.download_button(
                label=label,
                data=final_pdf_bytes,
                file_name=f"cleaned_{pdf_file.name}",
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

    # ──────────────────────────────────────
    # PROCESS IMAGES
    # ──────────────────────────────────────
    if image_files:
        st.subheader(f"🖼️ Processing {len(image_files)} Image(s)")
        cleaned_images = []

        for img_file in image_files:
            image = Image.open(img_file)
            img_bgr = pil_to_bgr(image)
            scanned_bgr = process_single_image(img_bgr)
            scanned_pil = bgr_to_pil(scanned_bgr)
            cleaned_images.append((img_file.name, image, scanned_pil))

            if ocr_enabled:
                page_text = run_ocr(scanned_pil)
                all_extracted_text.append(f"--- {img_file.name} ---\n{page_text}")

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
                    file_name=f"cleaned_{name}",
                    mime="image/png",
                    key=f"img_png_{name}"
                )
            with dl_cols[1]:
                buf_pdf = io.BytesIO()
                cleaned.save(buf_pdf, format="PDF", resolution=output_dpi)
                st.download_button(
                    label=f"⬇️ Download as PDF",
                    data=buf_pdf.getvalue(),
                    file_name=f"{os.path.splitext(name)[0]}.pdf",
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
                combined_bytes = combined_pdf.write()
                combined_pdf.close()
                st.download_button(
                    label="📄 Download All as Single PDF",
                    data=combined_bytes,
                    file_name="cleaned_all_pages.pdf",
                    mime="application/pdf",
                    key="combined_pdf_dl"
                )

        st.divider()

    # ──────────────────────────────────────
    # COMBINED OCR OUTPUT
    # ──────────────────────────────────────
    if ocr_enabled and all_extracted_text:
        with st.expander("📝 View All Extracted Text", expanded=False):
            combined_text = "\n\n".join(all_extracted_text)
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
st.caption("Built with OpenCV, NumPy, scikit learn, PyMuPDF, and Tesseract OCR. No data leaves your browser.")
