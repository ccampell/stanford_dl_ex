function [f,g] = logistic_regression_vec(theta, X,y)
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
  
  z = theta'*X;
  sigmoid = (1 ./ (1 + exp(-z)));
  f = (y*log(sigmoid)') + ((1-y)*log(1-sigmoid)');
  g = X*(sigmoid - y)';
  f = -f;
%   z = theta'*X;
%   sigmoid = (1 ./ (1 + exp(-z)));
%   f = sum(y.*log(sigmoid) + (1-y).*log(1-sigmoid));
%   g = X*(sigmoid - y)';
  
%   for i = 1:m
%       g = g + X(:,i) * (sigmoid(i) - y(i));
%   end
  
%   for i = 1:m
%       estimate = y(i)*log(sigmoid(i)) + (1-y(i))*log(1-sigmoid(i));
%       f = f + estimate;
%       g = g + X(:,i) * (sigmoid(i) - y(i));
%   end
  