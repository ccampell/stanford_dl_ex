function [f,g] = linear_regression(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));
  
  % Initialize Epsilon = 10^-4 for gradient descent checking. 
  epsilon = 10^-4;
  
  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  

%%% Compute the Linear Regression Objective/Cost %%%
% Dot product Theta Transpose with every feature column of matrix X
for i = 1:m
    x_i = X(:,i);
    % Take the dot product of column x^{i} from 'train.x' with weight vector
    % \theta^{T} to produce the scalar estimate. 
    % estimate = dot(x_i,theta.');
    estimate = theta' * x_i;
    % Sum the cost/objective over every column of x^{i} and store in 'f'.
    f = f + (estimate - y(i))^2;
    % Iterate over every parameter theta_{j} to compute the gradient.
    for j = 1:n
        % Update the gradient vector 'g' by mupltiplying the j'th element of
        % column vector x^{i} by predicted - actual. 
        g(j) = g(j) + (estimate - y(i)) * x_i(j);
        % Perform gradient analysis:
        epsilon_i = zeros(n);
        epsilon_i(i) = 1;
        theta_epsilon_plus = theta + epsilon*epsilon_i;
        theta_epsilon_minus = theta - epsilon*epsilon_i;
        comp_threshold = ?;
        cost_epsilon = 
        if abs(g(j)-cost_epsilon) <= comp_threshold
            
        end
            
    end
end
% Multiply the cost/objective 'f' by 1/2.
f = times(f,.5);
