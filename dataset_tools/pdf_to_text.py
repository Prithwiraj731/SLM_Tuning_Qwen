"""
Script 1: pdf_to_text.py
Purpose: Convert PDF documents into clean text.
Requirements: Read PDFs from data/pdf/, use PyMuPDF (fitz) for extraction, clean text.
Save output as data/text/combined_dataset.txt
"""

import os
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    import subprocess
    import sys
    print("PyMuPDF not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
    import fitz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdf")
TEXT_DIR = os.path.join(BASE_DIR, "data", "text")

def clean_text(text: str) -> str:
    """Cleans text of broken lines, excessive whitespace, and normalizes unicode."""
    # Remove broken line breaks (e.g., word- \n part)
    text = text.replace("-\n", "")
    
    # Normalize unicode to avoid strange characters
    text = text.encode("ascii", "ignore").decode("utf-8")

    # Replace multiple newlines (paragraphs) with a placeholder
    text = re.sub(r'\n{2,}', '<PARAGRAPH_BREAK>', text)
    
    # Replace single newlines with spaces (merging broken sentence lines)
    text = text.replace('\n', ' ')
    
    # Restore the paragraph breaks
    text = text.replace('<PARAGRAPH_BREAK>', '\n\n')
    
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def process_pdfs():
    os.makedirs(TEXT_DIR, exist_ok=True)
    
    if not os.path.exists(PDF_DIR) or not os.listdir(PDF_DIR):
        print(f"[!] No PDFs found in {PDF_DIR}.")
        print("[!] Please place your PDF documents inside the 'QWEN/data/pdf/' directory.")
        return

    print(f"[*] Reading PDFs from {PDF_DIR}...")
    
    all_extracted_text = []

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            
            try:
                # Open the document
                doc = fitz.open(pdf_path)
                
                # Extract text across all pages
                for page in doc:
                    all_extracted_text.append(page.get_text())
                
                doc.close()
                print(f"[+] Processed: {filename}")
                
            except Exception as e:
                print(f"[-] Error processing {filename}: {e}")

    if not all_extracted_text:
        print("[!] No text could be extracted from any PDFs.")
        return

    print("\n[*] Cleaning and combining text...")
    # Clean the aggregated text
    cleaned = clean_text(" ".join(all_extracted_text))
    
    # Save to a single text file
    text_path = os.path.join(TEXT_DIR, "combined_dataset.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
        
    print(f"[+] Output saved -> data/text/combined_dataset.txt")

if __name__ == "__main__":
    process_pdfs()
    print("[*] Extraction complete.")
