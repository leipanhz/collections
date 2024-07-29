import os
import argparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
import datetime

def get_time_milis(t):
    return int(t.total_seconds() * 1000) # milliseconds

def start_timer():
    return datetime.datetime.now()

def load_model_and_dataset(model_dir, dataset_dir):
    # model_dir = os.path.join(save_model_dir, 'distilbert')
    # dataset_dir = os.path.join(save_data_dir, 'imdb')

    # Load tokenizer and model from local disk
    begin = start_timer()
    print(f'time now: {begin}')
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=2)  # 2 labels for sentiment analysis
    end = start_timer()
    print(f'time now: {end}, time diff: {end - begin}, totoal seconds: {(end - begin).total_seconds()}')
    print(f"************ Time to load tokenizer and models: {get_time_milis(end - begin)} miliseconds. ")

    # Load dataset from local disk
    begin = start_timer()
    dataset = load_from_disk(dataset_dir)
    end = start_timer()
    print(f"************ Time to load dataset from disks: {get_time_milis(end - begin)} miliseconds. ")

    return tokenizer, model, dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": (predictions == torch.tensor(labels)).float().mean().item()}

def main(args):
    # Load model and dataset
    tokenizer, model, dataset = load_model_and_dataset(args.model_dir, args.data_dir)

    # Tokenize the dataset
    begin = start_timer()
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # Prepare the dataset for training
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(args.train_samples))
    eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(args.eval_samples))
    end = start_timer()
    print(f"************ Time to prepare dataset for training: {get_time_milis(end - begin)} miliseconds. ")

    # Define the training arguments
    begin = start_timer()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        ddp_find_unused_parameters=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    end = start_timer() 
    print(f"************ Time to train the model: {get_time_milis(end - begin)} miliseconds. ")

    # Evaluate the model
    begin = start_timer()
    eval_results = trainer.evaluate()
    end = start_timer() 
    print(f"************ Time to evaluate the model: {get_time_milis(end - begin)} miliseconds. ")
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model
    begin = start_timer()
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    end = start_timer() 
    print(f"************ Time to save the model: {get_time_milis(end - begin)} miliseconds. ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DistilBERT model with IMDb dataset from local disk.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model is saved')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where the dataset are saved')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the training results')
    parser.add_argument('--train_samples', type=int, default=25000, help='Number of training samples to use')
    parser.add_argument('--eval_samples', type=int, default=5000, help='Number of evaluation samples to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')

    args = parser.parse_args()
    main(args)

