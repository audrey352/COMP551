import numpy as np
import unicodedata
import torch
from torch.utils.data import Dataset
import os

REL_DATA_DIR = os.sep + os.path.join(*os.path.dirname(__file__).split(os.sep)[:-2], 'datasets') # robust relative path from script location to datasets/

def load_data(file_path, dtype=str)->np.ndarray:
    # read file 
    full_path = os.path.join(REL_DATA_DIR, file_path)
    with open(full_path, "r") as f:
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
    
def save_dataset(dataset, save_dir, filename):
    """
    Save dataset object to specified directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(dataset, os.path.join(save_dir, filename))
    
def load_dataset(load_dir, filename):
    """
    Load dataset object from specified directory
    """
    dataset = torch.load(os.path.join(load_dir, filename))
    return dataset