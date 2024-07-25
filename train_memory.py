import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os
from transformers import BitsAndBytesConfig


print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "Not available")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Load the model and tokenizer
model_path="/mnt/fs1/lroc/llama2/hf_model"
# data_dir_path = "/mnt/fs1/lroc/llama2/data/wikitext-2/"
data_dir_path = "/mnt/fs1/lroc/llama2/data/"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
# # Configure quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# # Load the model with quantization
# model = LlamaForCausalLM.from_pretrained(
#     model_path,
#     quantization_config=bnb_config,
#     device_map="auto",
# )

model = LlamaForCausalLM.from_pretrained(model_path)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load local dataset
dataset = load_dataset("text", data_files={"train": os.path.join(data_dir_path, "train.txt")}, split="train[:10]")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset in batches to save memory
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1,  # Process 1 example at a time to minimize memory usage
    num_proc=1,     # Use only one process to minimize memory usage
    remove_columns=dataset.column_names
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="no",
    evaluation_strategy="no",
    dataloader_num_workers=0,  # Disable multi-process data loading
    remove_unused_columns=False,  # Avoid unnecessary memory operations
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()
