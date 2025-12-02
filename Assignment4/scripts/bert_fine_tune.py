import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertForSequenceClassification

import unidecode
import string

import os
import multiprocessing
import logging # for logging training process
import time # to track training time
import pickle # to save models
import unicodedata
import argparse


# FOR SCRIPT ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("assignment_dir", type=str,
                    help="Path to the assignment directory")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for BERT fine-tuning")
args = parser.parse_args()

ASSIGNMENT_DIR = args.assignment_dir
BATCH_SIZE_BERT = args.batch_size

MODELDIR = ASSIGNMENT_DIR + 'models/'
DATA_DIR = ASSIGNMENT_DIR + 'datasets/'


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

    punctuation = ",.;:!?\"'()[]{}<>@#$%^&*_+=/\\|`~•–—"  # list of punctuation to remove
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
        self.Xraw = load_data(os.path.join(dataset_dir, 'X.txt')) # not tensor because string elements! oh well :(
        self.X = self.Xraw.copy()
        self.label = 'yl1' # default label
        self.Y =  torch.LongTensor(load_data(os.path.join(dataset_dir, 'Y.txt'), dtype=int)) # using LongTensor for cross-entropy loss (required!)
        self.YL1 = torch.LongTensor(load_data(os.path.join(dataset_dir, 'YL1.txt'), dtype=int))
        self.YL2 = torch.LongTensor(load_data(os.path.join(dataset_dir, 'YL2.txt'), dtype=int))
        self.sentences = []
        self.embedded_sentences = None

        # Compute flattened labels
        self.num_children = num_children
        self.YL_flat =  torch.LongTensor(np.array([yl1 * num_children + yl2 for yl1, yl2 in zip(self.YL1, self.YL2)]))
        
    def set_label(self, label):
        label = label.lower() if isinstance(label, str) else label
        if label in ['y', 'yl1', 'yl2', 'flat']:
            self.label = label
        else:
            raise ValueError("label must be 'y', 'yl1', 'yl2', or 'flat'")
        
    def select_label(self, label=None):
        """
        State which label to use: 'y', 'yl1', 'yl2', or 'flat'
        """
        label = label.lower() if isinstance(label, str) else label
        if label == 'y':
            return self.Y
        elif label == 'yl1':
            return self.YL1
        elif label == 'yl2':
            return self.YL2
        elif label == 'flat':
            return self.YL_flat
        else:
            raise ValueError("label must be 'y', 'yl1', 'yl2', or 'flat'")
    
    def clean_data(self):
        self.X = np.array([clean_text(x) for x in self.Xraw])
        
    def get_sentences(self):
        self.sentences = [text.split() for text in self.X]
        
    def embed_words(self, embed_model):
        self.embedded_sentences = []
        nsentences = len(self.sentences)
        for i, sentence in enumerate(self.sentences):
            print(f'Embedding sentence {i}/{nsentences}', end='\r')
            embedded = [embed_model.wv[word] for word in sentence if word in embed_model.wv]
            embedded_tensor = torch.tensor(embedded, dtype=torch.float32)
            self.embedded_sentences.append(embedded_tensor)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # raw = self.X[idx]
        # cleaned = clean_text(raw)
        # return cleaned, self.Y[idx], self.YL1[idx], self.YL2[idx], self.YL_flat[idx]
        label = self.select_label(self.label)
        data = self.embedded_sentences if self.embedded_sentences is not None else self.X
        return data[idx], label[idx]
    

# Loading WOS11967
dataset_dir = './datasets/WOS11967/'
WOS11967_dataset = WOSDataset(dataset_dir)

# Pre-procesing data for LSTM
WOS11967_dataset.clean_data() 
WOS11967_dataset.get_sentences() # list of list of words

# train/test split
TRAIN_RATIO = 0.8
num_train = int(len(WOS11967_dataset) * TRAIN_RATIO)
num_test = len(WOS11967_dataset) - num_train
train_dataset, test_dataset = random_split(WOS11967_dataset, [num_train, num_test])


# --- Load tokenizer & Model---
num_classes = 35  # flattened parent-child labels
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, clean_up_tokenization_spaces=True)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=num_classes
).to(device)
original_state_dict = {k: v.clone() for k, v in model.state_dict().items()} # save original state dict for resetting later


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


# --- Fine-Tuning BERT Model ---
# ** Running this on mimi GPU with a python script **
learning_rates = [1e-5, 2e-5, 3e-5]

# Dict to store all results
results = {}

# Get tokenized loaders (with current batch size)
train_loader_bert = DataLoader(train_dataset, batch_size=BATCH_SIZE_BERT, shuffle=True, collate_fn=bert_collate)
test_loader_bert = DataLoader(test_dataset, batch_size=BATCH_SIZE_BERT, shuffle=False, collate_fn=bert_collate)

# Loop over learning rates
for lr in learning_rates:
    # Reset optimizer & model
    model.load_state_dict(original_state_dict)
    optimizer = AdamW(model.parameters(), lr=lr)
    torch.cuda.empty_cache()

    # Store losses and accuracies
    train_losses, test_losses, test_accuracies = [], [], []

    # Training loop (only a few epochs)
    num_epochs = 3
    for epoch in range(num_epochs):
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
    results[f"bs{BATCH_SIZE_BERT}_lr{lr}"] = {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "test_accuracy": test_accuracies
    }

    # Save model for this learning rate
    # torch.save(model.state_dict(), MODELDIR + f"bert_model_lr{lr}.pth")

# Save final results
with open(MODELDIR + f"bert_tuning_results_bs{BATCH_SIZE_BERT}.pkl", "wb") as f:
    pickle.dump(results, f)