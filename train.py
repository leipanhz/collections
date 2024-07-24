import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

import os
from transformers import LlamaTokenizer, LlamaForCausalLM

data_dir_path="/mnt/fs1/lroc/llama2/data/wikitext-2/"
output_dir_path="/mnt/fs1/lroc/llama2/output"

model_path="/mnt/fs1/lroc/llama2/hf_model"
tokenizer_path="/mnt/fs1/lroc/llama2/hf_model"

parser = argparse.ArgumentParser(description='LLaMa2 Training Script')
parser.add_argument('--data_dir', type=str, default=data_dir_path, help='Directory containing the training data')
parser.add_argument('--output_dir', type=str, default=output_dir_path, help='Directory to save output files')
args = parser.parse_args()

# # # Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path)


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
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train() 