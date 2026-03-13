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
    Format a generic training example implementing reasoning style output.
    Uses the Instruction format as standard practice.
    """
    
    instruction = (
        "Analyze the provided text and summarize its core concepts. "
        "Show your step-by-step reasoning before providing the final answer."
    )
    
    # Generic structured reasoning demonstrating chain-of-thought
    output = (
        "Step 1: understand the problem\n"
        "The objective is to analyze the provided text excerpt and extract its fundamental concepts into a clear summary.\n\n"
        "Step 2: analyze the information\n"
        f"Scanning the input context, the primary focus surrounds the following context: '{content_chunk[:150].strip()}...'. "
        "The text discusses various topics which require synthesizing into a coherent explanation.\n\n"
        "Step 3: derive the solution\n"
        "By structuring the extracted core points from the broader context, we can construct the final answer reflecting the source text.\n\n"
        f"Final Answer: {content_chunk[:500].strip()}... (Content summarized for brevity)"
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
