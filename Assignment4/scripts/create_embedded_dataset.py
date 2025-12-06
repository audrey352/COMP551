from gensim.models import Word2Vec

import os
import multiprocessing

from utilities.plot_params import *  # import plotting parameters
from utilities.log_params import *  # import logging parameters
from utilities.data_loading import WOSDataset, save_dataset


# set default save directory and parameters
curdir = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(curdir, 'LSTM_models/')
W2VDIR = os.path.join(SAVEDIR, 'Word2Vec_models/')
os.makedirs(W2VDIR, exist_ok=True)
os.makedirs(SAVEDIR, exist_ok=True)

# Dataset
DATASET = 'WOS11967'
# vector size for Word2Vec embeddings
EMBEDDING_DIM = 100


# loading dataset
log.info("Loading dataset")
dataset = WOSDataset(DATASET)

# preprocessing data
log.info("Preprocessing data")
dataset.clean_data() 
dataset.get_sentences()

# training or loading Word2Vec model
w2v_modelname = f'w2v_{DATASET}.pkl'
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
    w2v_model.save(os.path.join(W2VDIR, w2v_modelname))
log.info("Word2Vec model ready.\n")

# apply embeddings to sentences
log.info("Applying Word2Vec embeddings to dataset")
dataset.embed_words(w2v_model)

# saving embedded dataset
save_dataset(dataset, SAVEDIR, f'embedded_{DATASET}.pt')
log.info(f"Embedded dataset saved to {os.path.join(SAVEDIR, f'embedded_{DATASET}.pt')}")