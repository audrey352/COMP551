import sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import unicodedata
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

import os
import logging # for logging training process
import time # to track training time
import pickle # to save models


ASSIGNMENT_DIR = sys.argv[1]
MODELDIR = ASSIGNMENT_DIR + '/models/'
DATA_DIR = ASSIGNMENT_DIR + '/datasets/'


def load_data(file_path, dtype=str)->np.ndarray:
    # read file 
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f]  # list of all lines

    if dtype == int:
        return [int(x) for x in lines]  # make sure target lists are int type
    
    return np.array(lines)  # return array


def clean_text(text):
    """
    Clean text by:
    - lowercasing
    - removing accents
    - removing punctuation
    - keeping only alphanumeric characters, hyphens and spaces
    - removing double spaces
    """
    # Lowercase
    text = text.lower()

    # Removing accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # Removing punctuation
    punctuation = ",.;:!?(){}[]\"'*/\\"
    for p in punctuation:
        text = text.replace(p, " ")

    # Keeping only alphanumeric characters, hyphens and spaces
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch == '-' or ch == ' ':
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    text = "".join(cleaned)

    # Removing double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    
    return text.strip()


class WOSDataset(Dataset):
    """
    Class to build a torch dataset from the WOS .txt files
    """

    def __init__(self, dataset_dir, num_children=5):
        self.X = load_data(dataset_dir + 'X.txt')
        self.Y = load_data(dataset_dir + 'Y.txt', dtype=int)
        self.YL1 = load_data(dataset_dir + 'YL1.txt', dtype=int)
        self.YL2 = load_data(dataset_dir + 'YL2.txt', dtype=int)

        # Compute flattened labels
        self.num_children = num_children
        self.YL_flat = np.array([yl1 * num_children + yl2 for yl1, yl2 in zip(self.YL1, self.YL2)])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        raw = self.X[idx]
        cleaned = clean_text(raw)

        return cleaned, self.Y[idx], self.YL1[idx], self.YL2[idx], self.YL_flat[idx]


# Loading WOS11967
dataset_dir = DATA_DIR + 'WOS11967/'
WOS11967_dataset = WOSDataset(dataset_dir)

# train/test split
TRAIN_RATIO = 0.8
num_train = int(len(WOS11967_dataset) * TRAIN_RATIO)
num_test = len(WOS11967_dataset) - num_train
train_dataset, test_dataset = torch.utils.data.random_split(WOS11967_dataset, [num_train, num_test])

# Get dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Note: X are strings so they currently get returned as a tuple when we get a batch
# eg. sample_X = (
#     "abstract1",
#     "abstract2",
#     "abstract3",
#     ...
# )

# Check data
num_instances = len(WOS11967_dataset)
print(f'Total number of samples: {len(WOS11967_dataset)}')
num_classes_YL1 = len(set(WOS11967_dataset.YL1))
print(f'Number of classes in YL1 (parent labels): {num_classes_YL1}')
num_classes_YL2 = len(set(WOS11967_dataset.YL2))
print(f'Number of classes in YL2 (child labels): {num_classes_YL2}')
num_classes_YL_flat = len(set(WOS11967_dataset.YL_flat))
print(f'Number of classes in flattened YL (parent-child combined labels): {num_classes_YL_flat}')

# Check a sample (one batch)
sample_X, sample_Y, sample_YL1, sample_YL2, sample_YL_flat = next(iter(train_loader))
print(len(sample_X))  # (batch_size,)  THIS IS A TUPLE OF STRINGS
print(sample_Y.shape)  # (batch_size,)


# --- Load tokenizer & model ---
num_classes = 35  # flattened parent-child labels
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def bert_collate(batch):
    """
    Function to collate a batch of data for BERT model
    1. tokenize texts
    2. pad sequences to max length in batch
    3. return input_ids, attention_mask, labels
    """

    texts = [item[0] for item in batch]  # get texts
    labels = [item[4] for item in batch]  # use flattened labels

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

# Get tokenized loaders
train_loader_bert = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=bert_collate)
test_loader_bert = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=bert_collate)


# --- Load tokenizer & Model---
num_classes = 35  # flattened parent-child labels
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, clean_up_tokenization_spaces=True)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=num_classes
).to(device)
original_state_dict = model.state_dict()  # save original state dict for resetting later

def bert_collate(batch):
    """
    Function to collate a batch of data for BERT model
    1. tokenize texts
    2. pad sequences to max length in batch
    3. return input_ids, attention_mask, labels
    """

    texts = [item[0] for item in batch]  # get texts
    labels = [item[4] for item in batch]  # use flattened labels

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

# Get tokenized loaders
train_loader_bert = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=bert_collate)
test_loader_bert = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=bert_collate)

# --- Fine-Tuning BERT Model ---
# ** Running this on mimi GPU with a python script **
learning_rates = [1e-5, 2e-5, 3e-5]

# Dict to store all results
results = {}

for lr in learning_rates:
    # Reset optimizer & model
    optimizer = AdamW(model.parameters(), lr=lr)
    model.load_state_dict(original_state_dict)

    # Store losses and accuracies
    train_losses, test_losses, test_accuracies = [], [], []

    # Training loop (only a few epochs)
    num_epochs = 3
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader_bert:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader_bert)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader_bert:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                total_test_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_test_loss = total_test_loss / len(test_loader_bert)
            test_losses.append(avg_test_loss)
            test_acc = correct / total
            test_accuracies.append(test_acc)

            print(f"Epoch {epoch+1}/{num_epochs} -- "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}, "
                f"Test Acc: {test_acc:.4f}")
        
    # Store results for this learning rate
    results[lr] = {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "test_accuracy": test_accuracies
    }

    # Save model for this learning rate
    torch.save(model.state_dict(), MODELDIR + f"bert_model_lr{lr}.pth")

# Save final results
with open(MODELDIR + "bert_tuning_results.pkl", "wb") as f:
    pickle.dump(results, f)