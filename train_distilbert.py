import os
import argparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Features, Value, ClassLabel
import torch
import time

def tokenize_function(data, tokenizer, textfield):
    return tokenizer(data[textfield], padding='max_length', truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": (predictions == torch.tensor(labels)).float().mean().item()}

def timing_tracing_wrapper(func):
    def wrapper(*args, **kwargs):
        # simulate filesystem activity tracing
        separator = kwargs.get("separator")
        print(f'================ {separator} ==============')
        os.path.isfile(f'/mnt/fs1/lroc/{separator}.txt')

        # run module and get execution time
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        running_time = end_time - start_time
        print(f"Running time for {func.__name__}: {running_time:.4f} seconds")

        return result

    return wrapper

@timing_tracing_wrapper
def load_model(model_dir):
    # model_dir = os.path.join(save_model_dir, 'distilbert')
    # dataset_dir = os.path.join(save_data_dir, 'imdb')

    # Load tokenizer and model from local disk
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=2)  # 2 labels for sentiment analysis

    return [tokenizer, model]

@timing_tracing_wrapper
def load_dataset_for_distilbert(dataset_paths):
    # https://huggingface.co/docs/datasets/v1.12.0/package_reference/loading_methods.html
    # dataset = load_from_disk(dataset_dir)

    features = Features({
    'overall': Value('int32'),
    'vote': Value('string'),
    'verified': Value('bool'),
    'reviewTime': Value('string'),
    'reviewerID': Value('string'),
    'asin': Value('string'),
    # 'style': {
    #     'Format:': Value('string')
    # },
    'reviewerName': Value('string'),
    'reviewText': Value('string'),
    'summary': Value('string'),
    'unixReviewTime': Value('int32'),
    })

    # dataset = load_dataset('json', data_files={'train': 'path/to/train.json', 'test': 'path/to/test.json'})
    dataset = load_dataset('json', data_files=dataset_paths, split='train', features=features)
    return [dataset]

@timing_tracing_wrapper
def tokenize_and_prepare_dataset(dataset, tokenizer):
    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda data: tokenize_function(data, tokenizer, args.textfield), batched=True)

    # Prepare the dataset for training
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(args.train_samples))
    eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(args.eval_samples))
    
    return[train_dataset, eval_dataset]

@timing_tracing_wrapper
def train_arguments(args):
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
    return [training_args]

@timing_tracing_wrapper
def train(model, training_args, train_dataset, eval_dataset):
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return [trainer]

@timing_tracing_wrapper
def eval_model(trainer):
    # Evaluate the model
    return [trainer.evaluate()]

# FIXME: check eval_trainer needed?
@timing_tracing_wrapper
def save_model(args, model, tokenizer):
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

def main(args):
    [tokenizer, model] = load_model(args.model_dir, separator="load_model")
    [dataset] = load_dataset_for_distilbert(args.data_dir, separator="load_dataset")
    [train_dataset, eval_dataset] = tokenize_and_prepare_dataset(dataset,tokenizer, separator='tokenize_dataset')
    [training_args] = train_arguments(args, separator='training_arguments')
    [trainer] = train(model, training_args, train_dataset, eval_dataset, 'start_training')
    [eval_trainer] = eval_model(trainer, separator="eval_model")
    save_model(separator='save_model_tokenizer')

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
    parser.add_argument('--textfield', type=str, default="reviewText", help="field name in the dataset that contains training data")

    args = parser.parse_args()
    main(args)

