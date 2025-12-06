import torch
from torch.utils.data import random_split

import os
import multiprocessing

from utilities.plot_params import *  # import plotting parameters
from utilities.log_params import *  # import logging parameters
from utilities.data_loading import WOSDataset, load_dataset
from utilities.LSTM_model import LSTM


# set default save directory and parameters
curdir = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(curdir, 'LSTM_models/')
W2VDIR = os.path.join(SAVEDIR, 'Word2Vec_models/')
os.makedirs(W2VDIR, exist_ok=True)
os.makedirs(SAVEDIR, exist_ok=True)

# Dataset
DATASET = 'WOS11967'
# number of classes
N_CLASSES = 7 # number of classes in dataset
# train/test split ratio
TRAIN_RATIO = 0.8
# validation/train split ratio
VAL_TRAIN_SPLIT = 0.2
# batch size
BATCH_SIZE = 64
# number of epochs
NUM_EPOCHS = 50
# learning rate
LEARNING_RATE = 0.05
# optimizer
OPTIMIZER = 'Adam'
# hidden layer size for LSTM
HIDDEN_SIZE = 128
# dropout rate for LSTM
DROPOUT_RATE = 0
# vector size for Word2Vec embeddings
EMBEDDING_DIM = 100
# device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of processes for CPU training
NPROCESS = 16

modelname = f'LSTM_{DATASET}_e{NUM_EPOCHS}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_model.pth'

# loading dataset with embeddings
log.info("Loading dataset")
dataset = load_dataset(SAVEDIR, f'embedded_{DATASET}.pt')

# split dataset into training and test sets
num_train = int(len(dataset) * TRAIN_RATIO)
num_test = len(dataset) - num_train
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
log.info(f"Dataset loaded with {len(train_dataset)} training samples and {len(test_dataset)} test samples.")

lstm = LSTM(
        input_size=EMBEDDING_DIM,  # embedding size
        hidden_size=HIDDEN_SIZE,
        output_size=N_CLASSES,  # number of classes
        device=DEVICE,
        directory=SAVEDIR
        )

if modelname in os.listdir(SAVEDIR):
    state_dict = torch.load(os.path.join(SAVEDIR, modelname))
    lstm.load_state_dict(state_dict)
else:   
    # training LSTM

    # run on WOS11967 dataset
    lstm.fit(
        train_dataset, # subset of WOS11967_dataset
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        val_train_split=VAL_TRAIN_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=True,
        optimizer=OPTIMIZER,
        store_records=True
    )

    # save trained model
    model_save_path = os.path.join(SAVEDIR, modelname)
    torch.save(lstm.state_dict(), model_save_path)
    log.info(f"LSTM model saved to {model_save_path}")

# evaluate on test set
test_acc = lstm.evaluate_test_set(
    test_dataset,
    batch_size=BATCH_SIZE
)[0]

log.info(f"\nTest Accuracy: {test_acc*100:.2f}%")