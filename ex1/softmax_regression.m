function [f,g] = softmax_regression(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  K=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
  g = [g, zeros(n,1)];
  theta = [theta, zeros(n,1)];
  for i = 1:m
      for k = 1:K
          numerator = exp(theta(:,k)'*X(:,i));
          denominator = 0;
          for j = 1:K
              denominator = denominator + exp(theta(:,j)'*X(:,i));
          end
          if y(i) == k
              indicator = 1;
              f = f + indicator*log(numerator/denominator);
          else
              indicator = 0;
              % Multiply by zero (e.g. do nothing)
          end
          g(:,k) = g(:,k) + X(:,i)*(indicator-(numerator/denominator));
      end
  end
  f = -f;
  g = -g;
  % Remove extra column of zeros that was appended to g. 
  g = g(:,1:end-1);
  g=g(:); % make gradient a vector for minFunc
end