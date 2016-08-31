function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  n = size(X,1);
  m = size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
  %%% Compute the objective/cost function %%%
  % Iterate over every column vector in train.X to 
  for i = 1:m
      x_i = X(:,i);
      z = theta'*x_i; 
      sigmoid = (1 / (1 + exp(-z)));
      estimate = y(i)*(log(sigmoid)) + (1-y(i))*log(1-sigmoid);
      f = f + estimate;
      for j = 1:n
        g_j = x_i(j) * (sigmoid - y(i));
        g(j) = g(j) + g_j;
      end
  end
  f = -f;
  