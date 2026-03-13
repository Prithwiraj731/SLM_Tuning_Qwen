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

# Format instruction prompts
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_prompts(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    
    # We append EOS later using the tokenizer
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = prompt_template.format(instruction, input_text, output)
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True)

# ==============================================================================
# 4. LOAD QWEN2.5-0.5B USING UNSLOTH
# ==============================================================================
print("[*] Loading Qwen2.5-0.5B base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.5B",
    max_seq_length = MAX_SEQ_LEN,
    dtype = None, # Autodetect precision
    load_in_4bit = True, # QLoRA Memory Optimization
)

# Append specialized EOS token
def append_eos(examples):
    examples["text"] = [t + tokenizer.eos_token for t in examples["text"]]
    return examples

dataset = dataset.map(append_eos, batched=True)

# ==============================================================================
# 5. ENABLE QLORA TRAINING
# ==============================================================================
print("[*] Enabling QLoRA Optimization...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth optimization requires 0
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==============================================================================
# 6. CONFIGURE SFTTRAINER
# ==============================================================================
print("[*] Configuring SFTTrainer & Hyperparameters...")
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
# This automatically saves adapter_config.json and adapter_model.safetensors
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Custom training logs mapping
trainer.save_state()
try:
    with open("outputs/trainer_state.json", "r") as f:
        ts = json.load(f)
    with open(os.path.join(FINAL_MODEL_DIR, "trainer_state.json"), "w") as f:
        json.dump(ts, f, indent=4)
except Exception:
    pass

torch.save(trainer.args, os.path.join(FINAL_MODEL_DIR, "training_args.bin"))

with open(os.path.join(FINAL_MODEL_DIR, "training_log.txt"), "w") as f:
    for log in trainer.state.log_history:
        f.write(json.dumps(log) + "\n")

shutil.copy2(DATA_JSON_PATH, os.path.join(FINAL_MODEL_DIR, "dataset.json"))

# ==============================================================================
# 9. EXPORT MERGED BASE FP16 MODEL (Safetensors)
# ==============================================================================
print("[*] Exporting Merged FP16 version of the model...")
model.save_pretrained_merged(FINAL_MODEL_DIR, tokenizer, save_method = "merged_16bit")

# ==============================================================================
# 10. EXPORT GGUF ARCHITECTURES TO ROOT FOLDER
# ==============================================================================
print("[*] Quantizing model into GGUF representations...")
model.save_pretrained_gguf(FINAL_MODEL_DIR, tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf(FINAL_MODEL_DIR, tokenizer, quantization_method = "q5_k_m")
model.save_pretrained_gguf(FINAL_MODEL_DIR, tokenizer, quantization_method = "q8_0")

# Unsloth creates a parallel folder suffixed with '_gguf'. We move files to root.
GGUF_TEMP_DIR = FINAL_MODEL_DIR + "_gguf"
if os.path.exists(GGUF_TEMP_DIR):
    for file_name in os.listdir(GGUF_TEMP_DIR):
        src_path = os.path.join(GGUF_TEMP_DIR, file_name)
        # Rename and move to root FINAL_MODEL_DIR
        dest_name = file_name
        if "q4_k_m" in file_name.lower(): dest_name = "model-Q4_K_M.gguf"
        elif "q5_k_m" in file_name.lower(): dest_name = "model-Q5_K_M.gguf"
        elif "q8_0" in file_name.lower(): dest_name = "model-Q8_0.gguf"
        
        shutil.move(src_path, os.path.join(FINAL_MODEL_DIR, dest_name))
    shutil.rmtree(GGUF_TEMP_DIR)

# ==============================================================================
# 12. EXPORT ENTIRE FOLDER TO GOOGLE DRIVE
# ==============================================================================
if IN_COLAB:
    print(f"[*] Copying {FINAL_MODEL_DIR} to Drive -> {DRIVE_OUTPUT_DIR}")
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    for item in os.listdir(FINAL_MODEL_DIR):
        s = os.path.join(FINAL_MODEL_DIR, item)
        d = os.path.join(DRIVE_OUTPUT_DIR, item)
        if os.path.isdir(s):
            continue # We enforce single depth (no subfolders)
        shutil.copy2(s, d)
    print("[+] Drive Transfer Complete.")

# ==============================================================================
# 13. HUGGING FACE AUTOMATIC UPLOAD
# ==============================================================================
print("[*] Deploying configuration to Hugging Face Hub...")
if HF_TOKEN.startswith("hf_"):
    try:
        login(token=HF_TOKEN)
        api = HfApi()
        api.create_repo(repo_id=REPO_ID, private=False, exist_ok=True)
        
        # Uploading contents strictly from the root of the folder
        api.upload_folder(
            folder_path=FINAL_MODEL_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Export models via Colab Unsloth QLoRA Pipeline"
        )
        print(f"[SUCCESS] Public repository exported at: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"[!] Deployment encountered an issue: {e}")
else:
    print("[!] Warning: HF_TOKEN not instantiated correctly. Proceed directly with drive files.")

print("[*] All processes terminated successfully.")
