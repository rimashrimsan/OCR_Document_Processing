import os
from google.cloud import vision
from pdf2image import convert_from_path
import io
import time

def detect_text_google(client, image_content):
    image = vision.Image(content=image_content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'{response.error.message}')
    return response.full_text_annotation.text if response.full_text_annotation else ''
