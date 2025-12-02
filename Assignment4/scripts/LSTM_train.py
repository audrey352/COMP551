from gensim.models import Word2Vec

import torch
from torch.utils.data import random_split

import os
import multiprocessing

from utilities.plot_params import *  # import plotting parameters
from utilities.log_params import *  # import logging parameters
from utilities.data_loading import WOSDataset
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
NUM_EPOCHS = 10
# learning rate
LEARNING_RATE = 1e-2
# optimizer
OPTIMIZER = 'Adam'
# hidden layer size for LSTM
HIDDEN_SIZE = 256
# dropout rate for LSTM
DROPOUT_RATE = 0
# vector size for Word2Vec embeddings
EMBEDDING_DIM = 100
# device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# loading dataset
log.info("Loading dataset")
dataset = WOSDataset(DATASET)

# preprocessing data
log.info("Preprocessing data")
dataset.clean_data() 
dataset.get_sentences()

# training or loading Word2Vec model
w2v_modelname = f'w2v_{DATASET}.model'
if w2v_modelname in os.listdir(W2VDIR):
    log.info("\nLoading existing Word2Vec model")
    w2v_model = Word2Vec.load(os.path.join(W2VDIR, w2v_modelname))
else:
    log.info("\nTraining Word2Vec model")
    w2v_model = Word2Vec(
        sentences = dataset.sentences,
        vector_size = EMBEDDING_DIM,
        window = 5,
        min_count = 1,
        workers = multiprocessing.cpu_count()-1,
        epochs = 50
    )
    w2v_model.save(os.path.join(SAVEDIR, 'Word2Vec_models', w2v_modelname))
log.info("Word2Vec model ready.\n")

# apply embeddings to sentences
log.info("Applying Word2Vec embeddings to dataset")
dataset.embed_words(w2v_model)


# split dataset into training and test sets
num_train = int(len(dataset) * TRAIN_RATIO)
num_test = len(dataset) - num_train
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
log.info(f"Dataset loaded with {len(train_dataset)} training samples and {len(test_dataset)} test samples.")

# training LSTM
lstm = LSTM(
    input_size=EMBEDDING_DIM,  # embedding size
    hidden_size=HIDDEN_SIZE,
    output_size=N_CLASSES,  # number of classes
    device=DEVICE,
    directory=SAVEDIR
    )

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

# evaluate on test set
test_loss, test_acc = lstm.evaluate(
    test_dataset,
    batch_size=BATCH_SIZE
)

log.info(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")