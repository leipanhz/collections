import os
import argparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import json

def download_model_and_dataset(save_model_dir, save_dataset_dir):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dataset_dir):
        os.makedirs(save_dataset_dir)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)


    model_dir = os.path.join(save_model_dir, 'distilbert')
    dataset_dir = os.path.join(save_dataset_dir, 'imdb')

    # Download DistilBERT model and save it directly
    print("Downloading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Save the model and tokenizer
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    print(f"Model downloaded and saved to {model_dir}")

    # Step 2: Download IMDb dataset
    print("Downloading IMDb dataset...")
    dataset = load_dataset('imdb')
    
    # Save the dataset
    os.makedirs(dataset_dir, exist_ok=True)
    dataset.save_to_disk(dataset_dir)

    print(f"Dataset downloaded and saved to {dataset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download DistilBERT model and IMDb dataset to a specified directory.')
    parser.add_argument('--save_model_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--save_dataset_dir', type=str, required=True, help='Directory to save the dataset')
    args = parser.parse_args()

    download_model_and_dataset(args.save_model_dir, args.save_dataset_dir)
