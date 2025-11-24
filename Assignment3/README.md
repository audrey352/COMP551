PHYS551 - Applied Machine Learning
==================================

Assignment 3: Classification of Image Data with Multilayer Perceptrons and Convolutional Neural Networks
-----------------------------------------------------------------

This project explores the impact of various architectural and training decisions on the performance of neural networks. We investigate the effects of non-linearity, network depth, activation functions, regularization, data preprocessing, and pre-trained models on classification accuracy and training efficiency using the FashionMNIST dataset. The experiments are designed to demonstrate how these factors influence model performance and training behavior.

Tasks
-----

The following experiments were conducted in order:

1. MLP Architecture Comparison
   - Implemented three MLP models:
     1. No hidden layers (linear model).
     2. Single hidden layer with 256 units and ReLU activations.
     3. Two hidden layers with 256 units each and ReLU activations.
   - All models include a final softmax layer for classification.
   - Compared test accuracy of the three models.

2. Activation Function Comparison
   - Starting from the two-hidden-layer MLP, created additional models using tanh and Leaky-ReLU activations.
   - Trained each model and compared their test accuracies with the ReLU-based MLP.
   - Evaluated which activation functions perform better.

3. Regularization Effects
   - Hyperparameter search for optimal regularization strength and batch size.
   - Trained two-hidden-layer MLPs with L1 and L2 regularization applied independently.
   - Assessed how regularization affects test accuracy.

4. Effect of Input Normalization
   - Trained the same two-hidden-layer ReLU MLP as in Task 1 on unnormalized images.
   - Compared the resulting accuracy to the normalized case.

5. Data Augmentation
   - Retrained the regularized MLPs form Task 3 using data augmentation.
   - Evaluated and compared the test accuracy.

6. Convolutional Neural Network (CNN)
   - Implemented a CNN with:
     - 2 convolutional layers
     - 1 fully connected hidden layer (256 units)
     - 1 fully connected output layer (10 units, softmax)
   - All layers used ReLU activations.
   - Trained the CNN on FashionMNIST and compared accuracy with the MLPs.

7. CNN with Data Augmentation
   - Trained the same CNN using the augmented dataset from Task 5.
   - Compared the performance in terms of accuracy and training time to the non-augmented case.

8. Pre-Trained Model Fine-Tuning
   - Loaded a pre-trained model (e.g., ResNet) using PyTorch.
   - Froze all convolutional layers and removed the original fully connected layers.
   - Added new fully connected layers and trained only these layers on FashionMNIST with data augmentation.
   - Ran experiments to find the optimal number of fully connected layers to add.
   - Compared performance (accuracy and training time) to the best MLP and CNN.