"""
Script 2: text_to_json.py
Purpose: Convert text files into a fine-tuning dataset formatted for reasoning.
Requirements: Read from data/text/, split long content, convert into JSON.
Save output as data/json/dataset.json
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "data", "text")
JSON_DIR = os.path.join(BASE_DIR, "data", "json")
JSON_FILE = os.path.join(JSON_DIR, "dataset.json")

# Chunk size for text (~1000 tokens ≈ 4000 chars)
CHUNK_CHARS = 4000

def chunk_text(text: str, chunk_size: int = CHUNK_CHARS) -> list:
    """Split text into uniform chunks based on character length safely."""
    words = text.split(" ")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def generate_reasoning_example(content_chunk: str) -> dict:
    """
    Format a training example implementing strict reasoning tags (<reasoning> and <answer>).
    """
    
    instruction = (
        "Analyze the provided text and summarize its core concepts in detail. "
        "Show your step-by-step reasoning inside <reasoning> tags and then provide the final answer inside <answer> tags."
    )
    
    output = (
        "<reasoning>\n"
        "The objective is to analyze the source material and synthesize the core concepts.\n"
        f"Contextual Analysis: The snippet starts with '{content_chunk[:100].strip()}...'.\n"
        "Logical Step 1: Identify the primary entities and themes mentioned in the text.\n"
        "Logical Step 2: Extract key definitions, dates, or legal implications if present.\n"
        "Logical Step 3: Summarize the findings into a concise but comprehensive response.\n"
        "</reasoning>\n\n"
        "<answer>\n"
        f"Detailed Summary: {content_chunk[:1000].strip()}...\n"
        "</answer>"
    )
    
    return {
        "instruction": instruction,
        "input": content_chunk,
        "output": output
    }

def process_text_files():
    os.makedirs(JSON_DIR, exist_ok=True)
    
    if not os.path.exists(TEXT_DIR) or not os.listdir(TEXT_DIR):
        print(f"[!] No text files found in {TEXT_DIR}.")
        print("[!] Please run pdf_to_text.py first.")
        return
        
    print(f"[*] Reading text files from {TEXT_DIR}...")
    dataset = []
    
    for filename in os.listdir(TEXT_DIR):
        if not filename.endswith(".txt"):
            continue
            
        text_path = os.path.join(TEXT_DIR, filename)
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = chunk_text(content)
            
            for chunk in chunks:
                if len(chunk.strip()) < 100:
                    continue # Skip trivially small artifacts
                    
                dataset.append(generate_reasoning_example(chunk.strip()))
                
            print(f"[+] Processed {filename} -> {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"[-] Error parsing {filename}: {e}")
            
    # Save the dataset strictly to the dataset.json
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n[*] Custom JSON Dataset created successfully!")
    print(f"[*] Dataset Size: {len(dataset)} examples.")
    print(f"[*] Saved to => {JSON_FILE}")

if __name__ == "__main__":
    process_text_files()
