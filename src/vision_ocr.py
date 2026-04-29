# Legacy research module
# This file was used during early research with the Google Cloud Vision API.
# It is NOT used by the Streamlit app (app.py), which runs 100% offline.
# Kept for reference only.

def detect_text_google(client, image_content):
    """
    Extract text from an image using Google Cloud Vision API.
    Requires a valid Google Cloud credentials JSON file.
    This function is not called by the main application.
    """
    try:
        from google.cloud import vision
        image = vision.Image(content=image_content)
        response = client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(response.error.message)
        return response.full_text_annotation.text if response.full_text_annotation else ''
    except ImportError:
        raise ImportError(
            "google-cloud-vision is not installed. "
            "This module is not used by the main app. "
            "The app uses Tesseract OCR for fully offline text extraction."
        )
