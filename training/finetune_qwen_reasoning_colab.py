"""
Script 3: finetune_qwen_reasoning_colab.py
Purpose: Complete Colab pipeline for Qwen2.5-0.5B fine-tuning with reasoning via Unsloth.
"""

import os
import sys
import json
import subprocess
import shutil

# ==============================================================================
# 1. INSTALL AND IMPORT REQUIRED LIBRARIES
# ==============================================================================
print("[*] Installing required libraries...")
try:
    import unsloth
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "git+https://github.com/unslothai/unsloth.git",
                           "transformers>=4.51.3", "datasets<4.0.0", "trl==0.7.3", 
                           "peft", "accelerate", "bitsandbytes", 
                           "huggingface_hub", "sentencepiece", "tqdm"])

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from huggingface_hub import HfApi, login

# ==============================================================================
# 2. CONFIGURATION AND DRIVE MOUNTING
# ==============================================================================
# Google Colab specific logic
try:
    from google.colab import drive
    print("[*] Mounting Google Drive...")
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    print("[!] Not running in Colab. Using local execution paths.")
    IN_COLAB = False

HF_TOKEN = "hf_..." # <--- REPLACE WITH YOUR HF WRITE TOKEN
REPO_ID = "YourUsername/Qwen2.5-0.5B-Reasoning-Model" # <--- REPLACE WITH YOUR HF REPO

# Paths
BASE_DIR = "/content" if IN_COLAB else os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(BASE_DIR, "dataset.json") # Ensure uploaded here
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/QWEN/final_model/"

# Setup fresh clean final directory
if os.path.exists(FINAL_MODEL_DIR):
    shutil.rmtree(FINAL_MODEL_DIR)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARN_RATE = 2e-4
MAX_EPOCHS = 3
MAX_SEQ_LEN = 2048

# ==============================================================================
# 3. LOAD THE DATASET
# ==============================================================================
print(f"[*] Loading dataset from {DATA_JSON_PATH}...")
if not os.path.exists(DATA_JSON_PATH):
    raise FileNotFoundError(f"[!] Please upload your 'dataset.json' to {DATA_JSON_PATH}")

dataset = load_dataset("json", data_files={"train": DATA_JSON_PATH})["train"]

# ChatML Template (Standard for Qwen2.5)
def format_chatml(examples, tokenizer):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        user_msg = f"{instruction}\n\nContext:\n{input_text}"
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output}
        ]
        # Using tokenizer.apply_chat_template is safer for training
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }

# ==============================================================================
# 4. LOAD QWEN2.5-0.5B USING UNSLOTH
# ==============================================================================
print("[*] Loading Qwen2.5-0.5B base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.5B",
    max_seq_length = MAX_SEQ_LEN,
    dtype = None, 
    load_in_4bit = True,
)

# Crucial: Set the ChatML template if not present
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

# Format dataset using the tokenizer's template
dataset = dataset.map(lambda x: format_chatml(x, tokenizer), batched=True)

# ==============================================================================
# 5. ENABLE QLORA TRAINING
# ==============================================================================
print("[*] Enabling QLoRA Optimization...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==============================================================================
# 6. CONFIGURE SFTTRAINER
# ==============================================================================
print("[*] Configuring SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LEN,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps = 10,
        num_train_epochs = MAX_EPOCHS,
        learning_rate = LEARN_RATE,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

# ==============================================================================
# 7. EXECUTE TRAINING
# ==============================================================================
print("[*] Starting QLoRA Fine-Tuning...")
trainer.train()

# ==============================================================================
# 8. SAVE LORA ADAPTERS & BASE ARTIFACTS
# ==============================================================================
print(f"[*] Saving artifacts to {FINAL_MODEL_DIR}...")
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Save training state
trainer.save_state()
if os.path.exists("outputs/trainer_state.json"):
    shutil.copy2("outputs/trainer_state.json", os.path.join(FINAL_MODEL_DIR, "trainer_state.json"))

# ==============================================================================
# 9. EXPORT MERGED BASE FP16 MODEL (Safetensors)
# ==============================================================================
print("[*] Exporting Merged FP16 version...")
model.save_pretrained_merged(FINAL_MODEL_DIR, tokenizer, save_method = "merged_16bit")

# ==============================================================================
# 10. EXPORT GGUF ARCHITECTURES (Consolidated Logic)
# ==============================================================================
print("[*] Quantizing model into GGUF representations...")

quant_methods = ["q4_k_m", "q5_k_m", "q8_0"]
for method in quant_methods:
    print(f"[*] Exporting quantization: {method}...")
    # This might create subfolders, we will clean them up after
    model.save_pretrained_gguf(FINAL_MODEL_DIR, tokenizer, quantization_method = method)

# Cleanup: Unsloth typically puts GGUFs in FINAL_MODEL_DIR (or suffixed folders)
# We move all .gguf files found in FINAL_MODEL_DIR or its children to the root.
for root, dirs, files in os.walk(FINAL_MODEL_DIR):
    for file in files:
        if file.endswith(".gguf"):
            src = os.path.join(root, file)
            # Give specific names if they don't have them
            dest_name = file
            if "q4_k_m" in file.lower(): dest_name = "model-Q4_K_M.gguf"
            elif "q5_k_m" in file.lower(): dest_name = "model-Q5_K_M.gguf"
            elif "q8_0" in file.lower(): dest_name = "model-Q8_0.gguf"
            
            # Avoid target same as source
            if src != os.path.join(FINAL_MODEL_DIR, dest_name):
                shutil.move(src, os.path.join(FINAL_MODEL_DIR, dest_name))

# Remove the extra subdirectories often created by Unsloth
extra_dirs = [d for d in os.listdir(FINAL_MODEL_DIR) if os.path.isdir(os.path.join(FINAL_MODEL_DIR, d))]
for ed in extra_dirs:
    shutil.rmtree(os.path.join(FINAL_MODEL_DIR, ed))

# Check for the suffixed folder as well
GGUF_TEMP_DIR = FINAL_MODEL_DIR + "_gguf"
if os.path.exists(GGUF_TEMP_DIR):
    for file in os.listdir(GGUF_TEMP_DIR):
        if file.endswith(".gguf"):
            shutil.move(os.path.join(GGUF_TEMP_DIR, file), os.path.join(FINAL_MODEL_DIR, file))
    shutil.rmtree(GGUF_TEMP_DIR)

# ==============================================================================
# 11. EXPORT ENTIRE FOLDER TO GOOGLE DRIVE
# ==============================================================================
if IN_COLAB:
    print(f"[*] Copying artifacts to Drive -> {DRIVE_OUTPUT_DIR}")
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    for item in os.listdir(FINAL_MODEL_DIR):
        s = os.path.join(FINAL_MODEL_DIR, item)
        d = os.path.join(DRIVE_OUTPUT_DIR, item)
        if not os.path.isdir(s):
            shutil.copy2(s, d)
    print("[+] Drive Transfer Complete.")

# ==============================================================================
# 12. HUGGING FACE AUTOMATIC UPLOAD
# ==============================================================================
if HF_TOKEN.startswith("hf_"):
    try:
        login(token=HF_TOKEN)
        api = HfApi()
        api.create_repo(repo_id=REPO_ID, private=False, exist_ok=True)
        api.upload_folder(
            folder_path=FINAL_MODEL_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Export models via Colab Unsloth QLoRA Pipeline (ChatML + Tags)"
        )
        print(f"[SUCCESS] Hub Export: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"[!] HF Export issue: {e}")

print("[*] All processes terminated successfully.")
