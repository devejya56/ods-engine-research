import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_nlp_dataloader(dataset_name='imdb', model_name='distilbert-base-uncased', batch_size=16, subset_size=1000):
    """
    Loads an NLP dataset and tokenizes it for a specific model.
    Returns PyTorch DataLoader objects.
    """
    print(f"Loading {dataset_name} dataset for {model_name}...")
    
    # Load dataset
    if dataset_name == 'imdb':
        dataset = load_dataset("imdb")
        text_column = "text"
    elif dataset_name == 'ag_news':
        dataset = load_dataset("ag_news")
        text_column = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=[text_column])
    
    # Also need to handle other columns that might exist in imdb/ag_news
    # Usually we just need input_ids, attention_mask, and labels
    tokenized_datasets.set_format("torch")
    
    # Subsetting
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    if subset_size is not None and subset_size < len(train_dataset):
        # Huggingface dataset select
        train_dataset = train_dataset.select(range(subset_size))
        # Keep test set small for faster evaluation during research, e.g., 20% of subset
        test_size = min(len(test_dataset), int(subset_size * 0.2))
        test_dataset = test_dataset.select(range(test_size))
        print(f"[{dataset_name}] Using {subset_size} training samples, {test_size} test samples.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
