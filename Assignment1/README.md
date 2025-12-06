# COMP551 - Applied Machine Learning

## Assignment 1: Exploring Linear and Logistic Regression 

This project explores the performance of **linear regression** and **logistic regression** using analytical solutions and gradient descent methods.  
The experiments use datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and compare metrics such as training/test loss, epochs to convergence, and learning rate effects.

---

## Features / Experiments

The following experiments were conducted in order:

1. **Baseline Performance**  
   - Train linear regression and fully batched logistic regression models.  
   - Use an 80/20 train-test split.  
   - Report performance on both the training and test sets.

2. **Feature Weights**  
   - Report the learned weights of each feature.  

3. **Varying Training Set Size**  
   - Train on subsets of the training data (20%, 30%, …, 80%).  
   - Plot performance curves (training vs. test sets) as a function of training size.  
   - Include confidence intervals/error bars over multiple runs to capture variability.

4. **Mini-Batch Size Experiments**  
   - Compare convergence speed and performance with different batch sizes (8, 16, 32, 64, 128).  
   - Report train and test loss.

5. **Learning Rate Variation**  
   - Train both linear and logistic regression with different learning rates.  
   - Compare their effects on convergence and final performance.
