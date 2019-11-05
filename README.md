# Nonlinear-Regression

1. polynomial regression
  Using the first 100 points as training data, and the remainder as testing data, fit a polynomial
basis function regression for degree 1 to degree 10 polynomials. Did not use any regularization.
Plot training error and test error (i.e. mean squared error) versus polynomial degree.

2. polynomial regression with one feature only
  It is difficult to visualize the results of high-dimensional regression. Instead, only use one of
the features (use X n(:,3)) and again perform polynomial regression. Produce plots of the
training data points, learned polynomial, and test data points. The code visualize 1d.m
may be useful.

3. polynomial regression with regularization
  Implement L2-regularized regression. Again, use the first 100 points, and only use the 3rd
feature. Fit a degree 8 polynomial using lambda = f0; 0:01; 0:1; 1; 10; 100; 1000g. Use 10-fold cross-
validation to decide on the best value for lambda. Produce a plot of average validation set error
versus regularizer value.
