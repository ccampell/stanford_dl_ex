function [f,g] = softmax_regression_vec(theta, X,y)
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
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
  g = [g, zeros(n,1)];
  % Append a column of zeros to theta to garantee a unique solution. 
  theta = [theta, zeros(n,1)];
  % numerator = exp(theta(:,k)'*X(:,i));
  % Vectorized Denominator is a (K by M) matrix.
  inner_product = exp(theta'*X);
  % Take the sum of every column of inner_product; resulting in
  %     a (1 by M) summation vector. 
  sum_vec = sum(inner_product,1);
  % Divide the numerator by the denominator (K by M) / (1 x M). 
  h = bsxfun(@rdivide, inner_product, sum_vec);
  % h is a (K by M) matrix. 
  log_h = log(h);
  % Sum along the rows of log_h where the row_index == y.
  % We don't care about how good the predictor is for the wrong class; we
  % only care about when k == y.
  % Vectorization 1 of f:
  %   for i = 1:m
  %       f = f + log_h(y(i),i);
  %   end
  %   f = -f;
  % Vectorization 2 of f:
  I = sub2Ind(size(log_h),1:size(log_h,1),1:size(M));
  
  % Compute the gradient:
  for i = 1:m
      indicator = zeros(num_classes,1);
      indicator(y(i)) = 1;
      g = g + X(:,i)*(indicator - h(:,i))';
      %temp = temp(:,1:end-1);
      %g = g + temp;
  end
  g = -g;
  g = g(:,1:end-1);
  g=g(:); % make gradient a vector for minFunc

