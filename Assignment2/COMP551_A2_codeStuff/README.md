# PHYS551 - Applied Machine Learning

## Assignment 2: Regularization and Model Evaluation

This project explores the details of linear regression using non-linear bases, bias-variance trade-off, regularization with cross-validation and the effects of L1 and L2 regularization on loss.
The presented tasks use synthetic datasets and compare relationships between training/validation error and the choice of parameters (such as number of basis functions or regularization strength). We also look at metrics such as bias and variance for each explored parameter.

---

## Tasks

The following tasks were conducted in order:

1. **Linear Regression with Non-Linear Basis Functions**  
   - Generate synthetic noisy data by sampling from a non-linear function. 
   - Fit a linear regression model using non-linear basis functions (Gaussian and Sigmoid).
   - Vary the number of basis functions used between 0 and 45 to see the effect it has on the model.
   - Report the sum of squared errors (SSE) and the number of Gaussian bases that minimizes the validation SSE.

2. **Bias-Variance Tradeoff with Multiple Fits**  
   - Repeat the previous task for 10 different datasets.
   - Report the mean squared error (MSE) on the training and test sets.
   - Report the bias and variance for each choice of number of Gaussian bases.

3. **Regularization with Cross-Validation**  
   - Use L1 and L2 regularization for various regularization strengths.
   - Perform 10-fold cross-validation on 50 different datasets using regularization.
   - Report the mean squared error (MSE) on the training and validation sets.
   - Report the bias and variance for each choice of regularization strength. 
   - Find the regularization strength that minimizes the validation MSE.

4. **Effect of L1 and L2 Regularization on Loss**  
   - Generate synthetic data using a linear function.
   - Make contour plots and optimization paths for L1 and L2 regularization, for a range of strength values.
