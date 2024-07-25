import argparse
# from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

import os
# import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
# from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from peft import get_peft_model, LoraConfig, TaskType



data_dir_path="/mnt/fs1/lroc/llama2/data/wikitext-2/"
output_dir_path="/mnt/fs1/lroc/llama2/output"

model_path="/mnt/fs1/lroc/llama2/hf_model"
tokenizer_path="/mnt/fs1/lroc/llama2/hf_model"

# Read TRANSFORMERS_CACHE from environment variable
transformers_cache = os.environ.get('TRANSFORMERS_CACHE', '/tmp/hf_cache')

# path validation
if not os.path.exists(model_path):
    raise ValueError(f"Model path does not exist: {model_path}")
if not os.path.exists(tokenizer_path):
    raise ValueError(f"Tokenizer path does not exist: {tokenizer_path}")
if not os.path.exists(data_dir_path):
    raise ValueError(f"Data directory does not exist: {data_dir_path}")

print(f"Model path contents: {os.listdir(model_path)}")
print(f"Tokenizer path contents: {os.listdir(tokenizer_path)}")
print(f"Data directory contents: {os.listdir(data_dir_path)}")

parser = argparse.ArgumentParser(description='LLaMa2 Training Script')
parser.add_argument('--data_dir', type=str, default=data_dir_path, help='Directory containing the training data')
parser.add_argument('--output_dir', type=str, default=output_dir_path, help='Directory to save output files')
args = parser.parse_args()

# # Load the tokenizer and model
# tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
# model = LlamaForCausalLM.from_pretrained(model_path)

# # Enable gradient checkpointing
# model.gradient_checkpointing_enable()

# # Set padding token
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.pad_token_id

# # Load the dataset from the local disk
# dataset_path = args.data_dir
# print(f"Loading dataset from {dataset_path}...")
# dataset = load_from_disk(dataset_path)
# print("Dataset loaded successfully.")

# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)

# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir=args.output_dir,
#     per_device_train_batch_size=1, # reduce batch size from 2
#     num_train_epochs=1,
#     logging_dir="./logs",
#     logging_steps=10,
#     fp16=True,  # Use mixed precision training
#     gradient_accumulation_steps=4,  # Accumulate gradients over multiple
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets,
# )

# # Train the model
# trainer.train() 


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, local_files_only=True, cache_dir=transformers_cache)

# Load the model in 8-bit precision
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto",
    local_files_only=True,
    cache_dir=transformers_cache
)

# # Prepare the model for int8 training
# model = prepare_model_for_int8_training(model)

# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Get the PEFT model
model = get_peft_model(model, lora_config)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Load the dataset from the local disk
dataset_path = args.data_dir
print(f"Loading dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path)
print("Dataset loaded successfully.")

# Set a temporary directory for the datasets library
os.environ['TMPDIR'] = '/tmp'

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset['train'].map(tokenize_function, batched=True, remove_columns=['text'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="/output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    optim="adamw_torch",
    learning_rate=2e-4,
    warmup_ratio=0.05,
    dataloader_num_workers=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("/output/final_model")
