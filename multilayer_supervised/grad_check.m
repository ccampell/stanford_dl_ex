function average_error = grad_check(fun, theta0, num_checks, varargin)

  delta=1e-3; 
  sum_error=0;

  fprintf(' Iter       i             err');
  fprintf('           g_est               g               f\n')

  for i=1:num_checks
    T = theta0;
    % j is a scalar (1 by 1) comprised of a random sample of
    % theta from 1 to num_elements(Theta)
    j = randsample(numel(T),1);
    % Epsilon-
    T0=T; T0(j) = T0(j)-delta;
    % Epsilon+
    T1=T; T1(j) = T1(j)+delta;
    % Get the results of the cost function for normal theta. 
    [f,g] = fun(T, varargin{:});
    % Get the results of J(epsilon+)
    f0 = fun(T0, varargin{:});
    % Get the results of J(epsilon-)
    f1 = fun(T1, varargin{:});
    % compute the gradient estimation as defined by the formula.
    g_est = (f1-f0) / (2*delta);
    % compute the error for the particular choice of delta. 
    % g_est is the gradient estimated from the function (correct grad).
    error = abs(g(j) - g_est);

    fprintf('% 5d  % 6d % 15g % 15f % 15f % 15f\n', ...
            i,j,error,g(j),g_est,f);

    sum_error = sum_error + error;
  end

  average_error=sum_error/num_checks;


