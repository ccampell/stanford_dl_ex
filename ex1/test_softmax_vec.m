m = 6000;
n = 76;
x = randn(n,m);
k = 10;
theta = randn(n,k-1);
theta = [theta, zeros(n,1)];

h = exp(theta'*x);
h = bsxfun(@rdivide, h, sum(h,1));
% Create a m by k matrix with one 1 per row. 
y = mnrnd(1,h');
[~, i] = max(y,[],2);
y = i;

% average_error = grad_check(@softmax_regression,theta(:,1:end-1),20,x,y);
% fprintf('avg_error=%f\n',average_error);
tic;
for i = 1:2
    [f_0,g_0] = softmax_regression(theta, x, y);
end
toc;

tic
for i = 1:2
    [f_1,g_1] = softmax_regression_vec(theta, x, y);
end
toc;

normf = norm(f_0-f_1);
normg = norm(g_0-g_1);
fprintf('normf=%f normg=%f\n',normf,normg)