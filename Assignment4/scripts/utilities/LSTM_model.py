import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import time
import numpy as np
from utilities.log_params import log
import multiprocessing
import os

from utilities.log_params import log
from utilities.plot_params import *

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cuda', nprocess=None, directory=None):
        super().__init__()
        
        self.directory = directory
        
        # device setup
        device = device.lower()
        assert device in ('cuda', 'cpu'), "Device must be 'cuda' or 'cpu'"
        use_cuda = (device == 'cuda') and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if not use_cuda:
            nprocess = multiprocessing.cpu_count() // 2 if nprocess is None else nprocess # defaults to half available cores
            msg = "CUDA not available, using CPU instead" if device == 'cuda' else "Using CPU"
            torch.set_num_threads(nprocess)
            log.info(f"{msg} with nprocess={nprocess}")
        else:
            log.info("Using CUDA device for LSTM model!")
        self.to(self.device)
        self.nprocess = nprocess
        
        # based heavily on pytorch implementation (but changed notation)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # training records (to add...)
        self.training_time = []
        self.training_loss = []
        self.validation_loss = []
        
        
        # Input gate
        self.W_xi = nn.Linear(input_size, hidden_size, bias=True) # input
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True) # hidden
        
        # Forget gate
        self.W_xf = nn.Linear(input_size, hidden_size, bias=True) # input
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True) # hidden
        
        # Candidate state
        self.W_xc = nn.Linear(input_size, hidden_size, bias=True) # input
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=True) # hidden

        # Output gate
        self.W_xo = nn.Linear(input_size, hidden_size, bias=True) # input
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=True) # hidden
        
        # Output layer
        self.W_hy = nn.Linear(hidden_size, output_size, bias=True)
        
        
    def forward(self, x, hidden=None): 
        """ 
        Args:
            x: (batch_size, seq_len, input_size)
            hidden: tuple of (h, c) each (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            cs_t = torch.zeros(batch_size, self.hidden_size, device=x.device) # cell state
        else:
            h_t, cs_t = hidden
            
        for t in range(seq_len):
            x_t = x[:, t, :]
        
            i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h_t))
            f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h_t))
            c_t = torch.tanh(self.W_xc(x_t) + self.W_hc(h_t))
            o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_t))
        
            cs_t = f_t * cs_t + i_t * c_t # update cell state
            h_t = o_t * torch.tanh(cs_t) # update hidden state
             
        y = self.W_hy(h_t) # bias included in Linear layer
        return y, (h_t, cs_t)
    
    
    def fit(self, dataset, epochs=10, lr=1e-3, batch_size=32, shuffle=True, optimizer='Adam', val_train_split=0, store_records=False):
        """
        Can chose between optimizers: Adam, RMSprop, Adagrad
        """        
        log.info(" == LSTM Training Initialization ==")
        if val_train_split > 0:
            val_size = int(len(dataset) * val_train_split)
            train_size = len(dataset) - val_size
            train_set, val_set = random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=LSTM.collate_fn)
            dataset = train_set  # for training loader
            log.info(f"Splitting dataset into {100*(1-val_train_split):.1f}% train and {100*val_train_split:.1f}% validation")
        else:
            val_loader = None
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=LSTM.collate_fn)
        log.info(f"Training on {len(train_loader.dataset)} samples in {len(train_loader)} batches of size {batch_size}")
        
        # getting optimizer
        if  optimizer == 'Adam':
            opt = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'RMSprop':
            opt = torch.optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'Adagrad':
            opt = torch.optim.Adagrad(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        log.info(f"Using {optimizer} optimizer with learning rate {lr}")
        
        # defining loss function to be cross entropy
        loss_fn = nn.CrossEntropyLoss()

        # training loop!
        log.info(" == Training LSTM Model ==")
        for epoch in range(epochs):
            t = time.time()
            total_loss = 0
            for X, Y in train_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                opt.zero_grad()
                logits, _ = self.forward(X)
                loss = loss_fn(logits, Y)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                
            if val_loader is not None:
                val_loss = 0
                for val_X, val_Y in val_loader:
                    val_X, val_Y = val_X.to(self.device), val_Y.to(self.device)
                    val_logits, _ = self.forward(val_X)
                    v_loss = loss_fn(val_logits, val_Y)
                    val_loss += v_loss.item()
                avg_val_loss = val_loss / len(val_loader)
                    
            avg_loss = total_loss / len(train_loader)
            t_epoch = time.time() - t
            
            time_to_completion = t_epoch * (epochs - epoch - 1) / 60
            if store_records:
                self.training_time.append(t_epoch)
                self.training_loss.append(avg_loss)
                    
                if self.directory is not None:
                    plt.plot(self.training_loss)
                    if val_loader is not None:
                        self.validation_loss.append(avg_val_loss)
                        plt.plot(self.validation_loss)
                        plt.legend(['Training Loss', 'Validation Loss'])
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Loss over Epochs')
                    
                    plt.savefig (os.path.join(self.directory, f'training_loss_epoch_{epoch+1}.png'))
                    plt.close()

            # if log.getEffectiveLevel() <= logging.INFO: # print only if log level is INFO. Doing this to avoid issues with logging output
            if val_loader is not None:
                log.info(f"Epoch {epoch+1}/{epochs}, Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f} -- Time: {t_epoch:.2f}s == ~{time_to_completion:.2f}min remaining {' '*10}") 
            else:
                log.info(f"Epoch {epoch+1}/{epochs}, Train Loss = {avg_loss:.4f} -- Time: {t_epoch:.2f}s == ~{time_to_completion:.2f}min remaining {' '*10}") 
    
    
    @torch.no_grad() # disable gradient calculation for inference
    def predict(self, X):
        X = X.to(self.device)
        logits, _ = self.forward(X)
        return torch.argmax(logits, dim=1).cpu()

    def evaluate_acc(self, Y, Yhat):
        return np.mean(Y == Yhat)
    
    def evaluate_test_set(self, test_subset, batch_size=32):
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=LSTM.collate_fn)
        y_true = []
        y_pred = []
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            Yhat_batch = self.predict(X_batch)
            y_true.extend(Y_batch.numpy())
            y_pred.extend(Yhat_batch.numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = self.evaluate_acc(y_true, y_pred)
        return acc, y_true, y_pred
    
    def plot_training_loss(self):
        plt.plot(self.training_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss over Epochs')
        plt.show()
        
    @staticmethod
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        padded = pad_sequence(sequences, batch_first=True)
        labels = torch.stack(labels)
        return padded, labels