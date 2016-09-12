m = 100;
n = 10;
x = randn(n,m);
theta = randn(n,1);
h = 1./(1+exp(-theta'*x));
r = rand(1,m);
% Element wise comparison, returns a boolean vector y (with same dimensions
% as h).
y = r < h;
tic;
for i = 1:10000
    [f_0,g_0] = logistic_regression(theta, x,y);
end
toc;
tic;
for i = 1:10000
    [f_1,g_1] = logistic_regression_vec(theta, x,y);
end
toc;
normf = norm(f_0-f_1);
normg = norm(g_0-g_1);
fprintf('normf=%f normg=%f\n',normf,normg)