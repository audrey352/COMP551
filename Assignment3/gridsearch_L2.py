from torchvision import models, datasets, transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import tqdm

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import os
import pickle # to save models


# define plotting parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['legend.fancybox'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.autolimit_mode'] = 'data'  # default, ensures autoscale uses data
plt.rcParams["font.family"] = "serif"


# set default save directory and parameters
SAVEDIR = '/home/2024/aberni24/Courses/COMP551/Assignment3/figures/'
MODELDIR = '/home/2024/aberni24/Courses/COMP551/Assignment3/models/'
os.makedirs(SAVEDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# CE loss
def cross_entropy_loss(y_true, y_pred):
    N = y_true.shape[0]
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Compute cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred)) / N
    return loss


# Accuracy
def evaluate_acc(y_true, y_pred):
    correct_predictions = np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    return np.mean(correct_predictions)


def plot_gridsearch_results(
    tuning_results,
    L,
    batch_range,
    lr_range
):
    # plot heatmap of results
    heatmap_acc = pd.DataFrame(index=batch_range, columns=lr_range)
    for (batch_size, lr), acc in tuning_results.items():
        heatmap_acc.at[batch_size, lr] = acc[0]
        
    sns.heatmap(heatmap_acc.astype(float), annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, square=True)
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.grid(False)

    plt.title(f'Test Accuracy for MLP (L={L})')
    plt.savefig(os.path.join(SAVEDIR, f'gridsearch_mlp_L{L}.png'))
    plt.show()

    # make another one showing # epochs
    heatmap_epochs = pd.DataFrame(index=batch_range, columns=lr_range)
    for (batch_size, lr), acc in tuning_results.items():
        heatmap_epochs.at[batch_size, lr] = acc[1]
    sns.heatmap(heatmap_epochs.astype(int), annot=True, fmt="d", cmap="YlGnBu", cbar=False, square=True)
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.grid(False)

    plt.title(f'Number of Epochs for MLP (L={L})')
    plt.savefig(os.path.join(SAVEDIR, f'gridsearch_mlp_epochs_L{L}.png'))
    plt.show()


# --- Activation functions and their derivatives ---
def softmax(z):
    z = np.atleast_2d(z)  # make sure z is 2D
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

def relu(z, deriv=False):
    if deriv:
        return np.where(z > 0, 1, 0)
    return np.maximum(0, z)

def tanh(z, deriv=False):
    if deriv:
        return 1 - np.tanh(z)**2
    return np.tanh(z)

def leaky_relu(z, l=0.01, deriv=False):
    if deriv:
        return np.where(z > 0, 1, l)
    return np.maximum(0, z) + l * np.minimum(0, z)


# --- Function to Compute Gradient ---
def gradient(x, # N x D
             y, # N x C
             params, # Replaces explicit params below
            #  w, # M2 x C 
            #  v1 = None, # D x M1
            #  v2=None, # M1 x M2
             h = None,  # hidden layer activations (for L=2)
             L=0
             ):    
    # output layer (CE loss and softmax activation)
    N,D = x.shape

    # -- 0 hidden layers --
    # yh = softmax(WX)
    if L==0:
        w, b = params
        yh = softmax(np.dot(x, w) + b)  # N x C

        # compute dW = (yh - y)x /N
        dy = yh - y  # N x C
        dw = np.dot(x.T, dy)/N  # D x C
        db = np.mean(dy, axis=0) # C

        return dw, db

    # -- 1 hidden layer --
    # yh = softmax(W h(V1X)) with h = ReLU
    elif L==1:
        try:
            w, b, v1, c1 = params
        except ValueError:
            raise ValueError("Expected 4 parameters for L=1: w, b, v1, c1")
        # assert v1 is not None, "v1 must be provided for L=1"
        
        # forward pass
        q = np.dot(x, v1) + c1  # N x M1
        z = h(q)  # N x M1 (general activation!)
        yh = softmax(np.dot(z, w) + b)  # N x C

        # compute dW = (yh - y)z /N
        dy = yh - y  # N x C
        dw = np.dot(z.T, dy)/N  # M1 x C

        # compute dV1 = (yh-y)W dq x /N where dq = ReLU derivative
        dq = h(q, deriv=True)  # N x M1 (derivative of of activation function)
        dz = np.dot(dy, w.T)  # N x M1
        dv = np.dot(x.T, dz * dq)/N  # D x M1
        dc = np.mean(dz * dq, axis=0)
        db = np.mean(dy, axis=0)
        
        return dw, db, dv, dc

    # -- 2 hidden layers --
    # yh = softmax(W h(V2 g(V1 X))) with g = h = {ReLU, tanh, or leaky ReLU}
    elif L==2:
        try:
            w, b, v1, c1, v2, c2 = params
        except ValueError:
            raise ValueError("Expected 6 parameters for L=2: w, b, v1, c1, v2, c2")
        # assert v1 is not None and v2 is not None and h is not None, "v1, v2, and h must be provided for L=2"
        
        # forward pass
        q1 = np.dot(x, v1) + c1  # N x M1
        z1 = h(q1)  # N x M1
        q2 = np.dot(z1, v2) + c2  # N x M2
        z2 = h(q2)  # N x M2
        yh = softmax(np.dot(z2, w) + b)  # N x C

        # compute dW = (yh - y)z2 /N
        dy = yh - y  # N x C
        dw = np.dot(z2.T, dy)/N  # M2 x C

        # compute dV2 = (yh - y)W dq2 z1 /N where dq2 depends on h
        dq2 = h(q2, deriv=True) # dh/dq2, depends on activation function
        dz2 = np.dot(dy, w.T)  # N x M2
        dv2 = np.dot(z1.T, dz2 * dq2)/N  # M1 x M2

        # compute dV1 = (yh - y)W dq2 V2 dq1 x /N  where dq1 depends on h
        dq1 = h(q1, deriv=True)  # dh/dq1, depends on activation function
        dz1 = np.dot(dz2 * dq2, v2.T)  # N x M1
        dv1 = np.dot(x.T, dz1 * dq1)/N  # D x M1
        db = np.mean(dy, axis=0)
        dc2 = np.mean(dz2 * dq2, axis=0)
        dc1 = np.mean(dz1 * dq1, axis=0)
        
        return dw, db, dv1, dc1, dv2, dc2
    

class MLP:
    """
    Supports L = 0,1,2 hidden layers.

    Constructor takes:
    h = hidden activation function (same for all hidden layers)
    L = number of hidden layers
    M = number of hidden units, iterable (each element corresponds to a layer)
    
    -> Weights and biases are initialized randomly
    """

    def __init__(self, h=None, L=1, M=64, D=None, C=10, l1_reg=0, l2_reg=0): 
        self.h = h if h is not None else lambda x: x # default identity
        self.L = L
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        assert l1_reg == 0 or l2_reg == 0, "Only one of l1_reg or l2_reg can be non-zero."
        
        if isinstance(M, int):
            self.M = [M] * L  # same num of units in each layer
        else:
            assert len(M) == L, "Length of M must equal L"
            self.M = M
        
        # Initialize weights & biases depending on # of layers (L)
        assert D is not None, "Need number of features"
        if L == 0:  # no hidden layer
            self.w = np.random.randn(D, C) * 0.1
            self.b = np.zeros(C)
        elif L == 1:  # 1 hidden layer
            self.v = np.random.randn(D, self.M[0]) * 0.1  # 1st hidden layer, D x M
            self.c = np.zeros(self.M[0])
            self.w = np.random.randn(self.M[0], C) * 0.1  # output layer, M x C
            self.b = np.zeros(C)
        elif L == 2:  # 2 hidden layers
            self.v1 = np.random.randn(D, self.M[0]) * 0.1  # 1st hidden layer, D x M1
            self.c1 = np.zeros(self.M[0])
            self.v2 = np.random.randn(self.M[0], self.M[1]) * 0.1  # 2nd hidden layer, M1 x M2
            self.c2 = np.zeros(self.M[1])
            self.w = np.random.randn(self.M[1], C) * 0.1  # output layer, M2 x C
            self.b = np.zeros(C)


    def fit(self, x, y, optimizer):
        # Put current weights into a list (for gradient function)
        if self.L == 0:
            params0 = [self.w, self.b]
        elif self.L == 1:
            params0 = [self.w, self.b, self.v, self.c]
        elif self.L == 2:
            params0 = [self.w, self.b, self.v1, self.c1, self.v2, self.c2]
        
        # Define gradient function for optimizer
        gradient_fn = lambda x, y, p: gradient(x, y, p, h=self.h, L=self.L)  

        # Run optimizer
        self.params = optimizer.run(gradient_fn, x=x, y=y, params=params0, l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        return self
    

    def predict(self, x):
        if self.L == 0:
            w, b = self.params
            yh = softmax(np.dot(x, w) + b)  # N x C
        elif self.L == 1:
            w, b, v, c = self.params
            q = np.dot(x, v) + c  # N x M
            z = self.h(q)  # N x M
            yh = softmax(np.dot(z, w) + b)  # N x C
        elif self.L == 2:
            w, b, v1, c1, v2, c2 = self.params
            q1 = np.dot(x, v1) + c1  # N x M1
            z1 = self.h(q1)  # N x M1
            q2 = np.dot(z1, v2) + c2  # N x M2
            z2 = self.h(q2)  # N x M2
            yh = softmax(np.dot(z2, w) + b)  # N x C
        return yh


# Gradient Descent Optimizer
class GradientDescent:
    
    def __init__(self, learning_rate=.001, epsilon=1e-8, max_iters=1, record_grad=False):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.record_grad = record_grad
        self.history_grad = []
        
    def run(self, gradient_fn, x, y, params, l1_reg=0, l2_reg=0):
        """
        This does GD for x and y (which are already a SINGLE BATCH of the data)

        gradient_fn: function that computes gradients
        params: list of weight matrices, e.g., [w, b, v1, c1, v2, c2] for L=2
        """
        assert l1_reg == 0 or l2_reg == 0, "Only one of l1_reg or l2_reg can be non-zero."
        
        norms = np.array([np.inf])
        t=0

        while np.any(norms > self.epsilon) and t < self.max_iters:
            grad = gradient_fn(x, y, params)
            
            
            norms = np.array([np.linalg.norm(g) for g in grad])
            # record gradients if desired
            if self.record_grad:
                total_norm = np.sqrt(np.sum(norms**2))
                self.history_grad.append(total_norm)
                            
            for p in range(len(params)):
                if np.ndim(params[p]) > 1: # so not biases
                    params[p] -= self.learning_rate * (grad[p] + l1_reg * np.sign(params[p]) + l2_reg * params[p])
                else:
                    params[p] -= self.learning_rate * grad[p]
            t += 1
    
        return params
    

# --- Train on batches over epochs ---
def train_on_batches(model, optimizer, epochs, train_loader, val_loader=None, early_stopping_patience=3, num_classes=10, plot_train=False, verbose=True,
                     save_best_weights=True, save_fig=False, save_name=''):
    """
    Returns dictionary with:
    'grad': grad, # keeping all batches for now... can change later for space optimization
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc,
    'best_weights': best_weights,
    'fig': fig if plot_train else None,
    'ax': ax if plot_train else None   
    """
    
    grad, train_loss, val_loss = {}, {}, {}
    val_acc = {}
    best_val_loss = np.inf
    best_weights = None
    
    for epoch in range(epochs):
        grad[epoch], train_loss[epoch], val_loss[epoch] = [], [], []
        batch_num = len(train_loader)
        
        # Go through batches
        for i, (b_x, b_y) in enumerate(train_loader):
            # flatten images
            b_x = b_x.view(b_x.size(0), -1).numpy()
            # one-hot encode labels
            y_batch = np.eye(num_classes)[b_y.numpy()]
            # fit model on this batch
            model.fit(b_x, y_batch, optimizer)
            try:
                grad[epoch].append(optimizer.history_grad[-1])
            except:
                grad[epoch] = 0
            
            # compute loss on this batch
            yhat_batch = model.predict(b_x)
            batch_loss = cross_entropy_loss(y_batch, yhat_batch)
            if np.isnan(batch_loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {i+1}. Stopping this run.")
                return None 
            train_loss[epoch].append(batch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{batch_num} === Loss: {batch_loss:.3} {' '*10}", end='\r')
            # end of batch loop for training set
        
        avg_loss = np.mean(train_loss[epoch])
        avg_grad = np.mean(grad[epoch])
        if verbose:    
            print(f"Epoch {epoch+1}/{epochs} === Train Loss: {avg_loss:.3} | Avg Grad Norm: {avg_grad:.3} {' '*10}")
        
        # loop over validation batches to compute val loss
        if val_loader is not None:
            val_losses, val_accs = [], []
            for j, (val_x, val_y) in enumerate(val_loader):
                val_x = val_x.view(val_x.size(0), -1).numpy()
                y_val = np.eye(num_classes)[val_y.numpy()]
                yhat_val = model.predict(val_x)
                val_batch_loss = cross_entropy_loss(y_val, yhat_val)
                val_losses.append(val_batch_loss)
                # getting validation accuracy at current epoch
                yh = model.predict(val_x)
                val_accs.append(evaluate_acc(y_val, yh))
                if verbose:
                    print(f"Validation batch {j+1}/{len(val_loader)} === Val Loss: {val_batch_loss:.3} {' '*10}", end='\r')
                # end of batch loop for validation set
            val_loss[epoch] = np.mean(val_losses)
            val_acc[epoch] = np.mean(val_accs)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} === Train Loss: {avg_loss:.3} | Val Loss: {val_loss[epoch]:.3} | Avg Grad Norm: {avg_grad:.3} {' '*10}")

            if val_loss[epoch] < best_val_loss:
                best_val_loss = val_loss[epoch]
                best_weights = model.params.copy()  # save best weights
                if verbose:
                    print('Yay! New best validation loss!')
            
            # early stopping condition: Checking if validation loss increased compared to the previous n epochs
            if epoch >= early_stopping_patience:
                if all(val_loss[epoch] >= val_loss[epoch - p - 1] for p in range(early_stopping_patience)):
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss over the last {early_stopping_patience} epochs.")
                    break
        # end of validation part
        
        if verbose:
            print('-'*20, end='\n\n')
    
    # Plot training and validation loss and accuracy
    if plot_train:
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # plot loss
        ax.plot([np.mean(train_loss[e]) for e in train_loss.keys()], label='Train Loss')
        if val_loader is not None:
            ax.plot([val_loss[e] for e in val_loss.keys()], label='Val Loss')
        
        # Labels
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_title('Training and Validation Loss over Epochs')
        
        # second axis for validation accuracy
        ax1 = ax.twinx()
        ax1.plot(val_acc.keys(), val_acc.values(), color='green', label='Val Accuracy', linestyle='--')
        ax1.set_ylabel('Validation Accuracy', color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_ylim(0, 1)

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()

        # Place legend below the plot as horizontal
        ax1.legend(lines + lines2, labels + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), ncol=3)
        # ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        if save_fig:
            plt.savefig(os.path.join(SAVEDIR, f'epoch_training_evolution_{save_name}.png'), bbox_inches='tight')
        plt.show()
        
    if save_best_weights and best_weights is not None:
        if verbose:
            print('Setting model to best weights found (lowest validation loss)')
        model.params = best_weights  # set model to best weights found!

    return {
        'grad': grad, # keeping all batches for now... can change later for space optimization
        'train_loss': train_loss, 
        'val_loss': val_loss,
        'val_acc': val_acc,
        'best_weights': best_weights,
        'fig': fig if plot_train else None,
        'ax': ax if plot_train else None
        }


# Compute mean and std of train dataset (for normalization)
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
images, labels = dataset.data, dataset.targets
images_scaled = dataset.data.float() / 255.0  # convert to float and scale to [0,1]

# Compute mean and std over all training set (since greyscale images)
MEAN = images_scaled.mean()
STD = images_scaled.std()
print(MEAN, STD)


# Function to load the FashionMNIST dataset with specific transforms
def get_datasets(use_validation=True, transform=None):
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
        )
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
        )

    # validation if wanted!
    val_dataset = None
    if use_validation:
        val_frac = 0.1
        num_train = len(train_dataset)
        val_size = int(val_frac * num_train)
        train_size = num_train - val_size
        val_dataset, train_dataset = torch.utils.data.random_split(
            train_dataset,
            [val_size, train_size]
        ) 
    return train_dataset, val_dataset, test_dataset
    # Note that train_dataset and val_dataset are Subset objects now!!!


# Get a normalized dataset
USE_VALIDATION = True
transform = transforms.Compose([transforms.ToTensor(), # scales to [0,1]
                            transforms.Normalize((MEAN,), (STD,))  # mean 0, std 1
                            ])
train_dataset, val_dataset, test_dataset = get_datasets(USE_VALIDATION, transform)

# Data info
print('\nDataset information:')
print('Training samples:', len(train_dataset))
if USE_VALIDATION:
    print('Validation samples:', len(val_dataset))
print('Test samples:', len(test_dataset))

print('\nUnderlying dataset info:')
print(train_dataset.dataset.classes)
print(train_dataset.dataset.data.shape)
print(train_dataset.dataset.targets.shape)

# Function to get loaders
def get_data_loaders(train_dataset, test_dataset, val_dataset=None, batch_size=512, batch_tests=False, use_validation=True):
    # Get loaders (makes batches for later)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
        )
    test_dataset_size = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size if batch_tests else test_dataset_size, 
        shuffle=False
    )

    if use_validation:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    return (train_loader, val_loader, test_loader) if use_validation else (train_loader, test_loader)


# Get loaders for normalized dataset, with batches
BATCH_SIZE = 512
train_loader, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, val_dataset, BATCH_SIZE, USE_VALIDATION)

# looking at data in 1 batch
images0, labels0 = next(iter(train_loader))  # get first batch``
print(images0.shape)  # (batch_size, 1, 28, 28)
print(labels0.shape)  # (batch_size,)
# same for test loader

num_classes = len(train_dataset.dataset.classes)


# Define hyperparameter ranges to search
num_hyper = 5
lowest_batch_log = 4  # 2^4 = 16
highest_batch_log = lowest_batch_log + num_hyper -1

batch_range, lr_range = np.logspace(lowest_batch_log, highest_batch_log, num=num_hyper, base=2, dtype=int), np.linspace(0.01, 0.5, num_hyper)

print("Batch sizes to try:", batch_range)
print("Learning rates to try:", lr_range)
# could add: number of epochs, number of iterations within GD


# LOADING GRID SEARCH RESULTS (run to avoid re-running grid search)
with open(os.path.join(MODELDIR, 'grid_search_results.pkl'), 'rb') as f:
    grid_search_results = pickle.load(f)
    
tuning_results0 = grid_search_results['L0']
tuning_results1 = grid_search_results['L1']
tuning_results2 = grid_search_results['L2']

best_hyper0 = grid_search_results['L0_best']
best_hyper1 = grid_search_results['L1_best']
best_hyper2 = grid_search_results['L2_best']

# Getting optimal hyperparameters -- these will be used for all subsequent experiments
# 0 hidden layers
BATCH_SIZE0 = best_hyper0[0][0]
LEARNING_RATE0 = best_hyper0[0][1]
# 1 hidden layer
BATCH_SIZE1 = best_hyper1[0][0]
LEARNING_RATE1 = best_hyper1[0][1]
# 2 hidden layers
BATCH_SIZE2 = best_hyper2[0][0]
LEARNING_RATE2 = best_hyper2[0][1]

print(f"Optimal hyperparameters:\nL=0: Batch Size = {BATCH_SIZE0}, Learning Rate = {LEARNING_RATE0}\nL=1: Batch Size = {BATCH_SIZE1}, Learning Rate = {LEARNING_RATE1}\nL=2: Batch Size = {BATCH_SIZE2}, Learning Rate = {LEARNING_RATE2}")


# --- 2 Hidden Layers ---
# Reload data loaders with optimal batch size
train_loader, val_loader, test_loader = get_data_loaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    val_dataset=val_dataset,
    batch_size=BATCH_SIZE2,
    use_validation=USE_VALIDATION
    )

# Define MLP and optimizer
model2 = MLP(
    h=relu, 
    L=2, 
    M=256,
    D=28*28
    )

optimizer2 = GradientDescent(learning_rate=LEARNING_RATE2, record_grad=True)  # doing one GD iteration by batch in one epoch
num_epochs = 100

results_l2 = train_on_batches(
    model=model2, 
    optimizer=optimizer2,
    epochs=num_epochs, 
    train_loader=train_loader, 
    val_loader=val_loader if USE_VALIDATION else None, 
    early_stopping_patience=3,
    plot_train=True,
    verbose=False, 
    save_fig=True, save_name='L2'
    )


# Checking test predictions for first batch
test_images, test_labels = next(iter(test_loader))
test_images = test_images.view(test_images.size(0), -1).numpy()
test_labels_onehot = np.eye(num_classes)[test_labels.numpy()]

yhat = model2.predict(test_images) # N x C


# print("Truth:\n", np.array(test_labels), "\nPrediction:\n", np.argmax(yhat, axis=1))
print("Accuracy on test set:", evaluate_acc(test_labels_onehot, yhat))

# create confusion matrix
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true_label, pred_label in zip(test_labels.numpy(), np.argmax(yhat, axis=1)):
    confusion_matrix[true_label, pred_label] += 1
# visualize confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for MLP with 2 Hidden Layers')
plt.show()


# REGULARIZATION
lr_range = np.linspace(0.01, 0.05, num_hyper)
reg_range = np.linspace(0.001, 0.02, num_hyper)    
 
# --- Finding best L2 regularization for model with L=2 hidden layers ---
tuning_results2_l2reg = {} # dict to store test accuracies
epochs = 50

# get loaders for this batch size (using normalized data)
train_loader, val_loader, test_loader = get_data_loaders(train_dataset=train_dataset, 
                                                        test_dataset=test_dataset, 
                                                        val_dataset=val_dataset, 
                                                        batch_size=BATCH_SIZE2, 
                                                        use_validation=USE_VALIDATION)

for reg in reg_range:
    # loop over learning rates
    for lr in lr_range:
        print(f'doing lr={lr:.3} and L1 reg={reg:.3}')

        # define model and optimizer (use same model params as for part 1)
        model = MLP(h=relu, L=2, M=256, D=28*28, l2_reg=reg)
        optimizer = GradientDescent(learning_rate=lr)
        
        results = train_on_batches(model, 
                                   optimizer, 
                                   epochs=epochs, 
                                   train_loader=train_loader, 
                                   val_loader=val_loader, 
                                   early_stopping_patience=5,
                                   num_classes=num_classes, 
                                   plot_train=False, 
                                   verbose=False, 
                                   save_best_weights=True)
        
        # Evaluate on test set
        test_acc = []
        for test_x, test_y in test_loader:
            test_x = test_x.view(test_x.size(0), -1).numpy()
            y_test = np.eye(num_classes)[test_y.numpy()]
            yh_test = model.predict(test_x)
            test_acc.append(evaluate_acc(y_test, yh_test))
        avg_test_acc = np.mean(test_acc)
        
        epoch = len(results["train_loss"])
        
        # print(f"L2 reg: {reg:.3} Learning Rate: {lr:.3} === Test Accuracy: {avg_test_acc:.3}")
        tuning_results2_l2reg[(reg, lr)] = [avg_test_acc, epoch, model]

print('Completed L2 regularization grid search')

# finding best hyperparameters
best_hyper2_l2reg = max(tuning_results2_l2reg.items(), key=lambda x: x[1][0])
best_model2_l2reg = best_hyper2_l2reg[1][2]
print(f"Best hyperparameters for L=2 with L2 regularization: L2 Reg = {best_hyper2_l2reg[0][0]:.3}, Learning Rate = {best_hyper2_l2reg[0][1]:.3} with Test Accuracy = {best_hyper2_l2reg[1][0]:.3} over {best_hyper2_l2reg[1][1]} epochs")

# save all model results
with open(os.path.join(MODELDIR, 'L2reg_grid_search_results.pkl'), 'wb') as f:
    pickle.dump({
        'help': 'Each tuning_results dict has keys as (regularization coefficient, learning_rate) and values as [test_accuracy, num_epochs, model].',
        'L2_reg': tuning_results2_l2reg,
        'L2_reg_best': best_hyper2_l2reg,
    }, f)

print('results saved successfully')