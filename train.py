import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

import os

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

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Load the dataset from the local disk
dataset_path = args.data_dir
print(f"Loading dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path)
print("Dataset loaded successfully.")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Define training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1, # reduce batch size from 2
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Use mixed precision training
    gradient_accumulation_steps=4,  # Accumulate gradients over multiple
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train() 