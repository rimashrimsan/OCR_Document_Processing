import fitz
import os
import shutil

def insert_pdf_pages(main_pdf_path, scan_pdf_path, output_pdf_path, insert_at_index):
    doc = fitz.open(main_pdf_path)
    scan = fitz.open(scan_pdf_path)
    doc.insert_pdf(scan, start_at=insert_at_index)
    scan.close()
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()

def remove_pdf_pages(main_pdf_path, output_pdf_path, pages_to_remove_1_based):
    doc = fitz.open(main_pdf_path)
    indices = sorted([p - 1 for p in pages_to_remove_1_based], reverse=True)
    for idx in indices:
        doc.delete_page(idx)
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()
