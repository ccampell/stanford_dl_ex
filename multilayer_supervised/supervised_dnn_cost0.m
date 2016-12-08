function [ cost, grad, pred_prob] = supervised_dnn_cost0(theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;
%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% Perform Initialization of Container Variables (chris)
num_layers = numHidden + 2;
num_samples = size(data,2);
% Create a container for each layer's weighted sums (z): 
z = cell(num_layers,1);
% Create a container for each layer's activations (a):
a = cell(num_layers,1);
% Create a container for each layer's error-term (delta):
delta = cell(num_layers,1);
% Create a container for each layer's (excluding nl) vector-differential (nabla): 
nabla = cell(num_layers-1,1);
% Create container for each layer's sigmoid-derivative (f'(z)):
f_prime_z = cell(num_layers,1);

%% forward prop (chris)
% Perform feedforward pass computing activations for the network. 
z{1} = NaN;
a{1} = data;
z{2} = (stack{1}.W*data)+stack{1}.b;
a{2} = 1./(1 + exp(-z{2}));
z{3} = (stack{2}.W*a{2})+stack{2}.b;
a{3} = 1./(1 + exp(-z{3}));

% Assign activations to the expected variables:
hAct{1} = a{2};
hAct{2} = a{3};
pred_prob = a{3};
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;
%%% Create Y from the label matrix: %%%
% Create Y (a 60,000 x 10 matrix) from the class-label matrix (label): 
Y = zeros(size(stack{2}.b,1),size(labels,1));
% Convert the subscripts to a linear index:
linearIndex = sub2ind([size(Y,1), size(Y,2)], labels', 1:size(Y,2));
% Propegate Y with 1s wherever the original labels were: 
Y(linearIndex) = 1;
%% compute cost (chris)
% Compute the sum of the squared error between hypothesis - actual.
class_labels = 1:10;
for sample = 1:num_samples
    for class_label = 1:class_labels
        % If the column with a 1 in it is the same as the current class
        % label; then trigger the activation function. 
        if Y(:,sample)
            
        end
        print(class_label)
    end
    h_wxi = a{3}(:,sample)
end
sum_sqr_err = 0;
for sample = 1:num_samples
    h_wxi = a{3}(:,sample);
    % Compute the euclidean '2' normalization (magnitude) of the vectors:
    sum_sqr_err = sum_sqr_err + .5*(norm(h_wxi-Y(:,sample))^2);
end
avg_sum_sqr_err = sum_sqr_err*(1/num_samples);
%J_Wb = .5*norm(a{3}-Y);
%J_Wb = (1/num_samples)*sum(J_Wb,2);
%weight_decay = cell(num_layers - 1,1);
% Compute the sum of the squared values of W across column then row for
% every layer. 
weight_decay = 0;
for layer = num_layers - 1:-1:1
    weight_decay = weight_decay + sum(sum(stack{layer}.W.^2,2),1);
end
weight_decay = weight_decay*(ei.lambda/2);
cost = avg_sum_sqr_err + weight_decay;
% TODO: How to use bsxfun(@addition, W_1*data, b_1)?
%% compute gradients using backpropagation (chris)
% Compute f'(z) for layer nl.
% Element wise multiplication because of scalar formula. We are not doing
% a summation otherwise it would be a matrix multiply. 
f_prime_z{num_layers} = a{3}.*(1-a{3});
% Compute delta for layer nl.
delta{num_layers} = -(Y - a{3}).*f_prime_z{num_layers};
% Compute f'(z) and delta for every preceeding layer. 
for i = num_layers - 1:-1:1
    f_prime_z{i} = a{i}.*(1-a{i});
    % FOR DELTA 2 we need it to be 10x256 for W. and a 10x1 for b. Sum up
    % over the rows of b. 
    delta{i} = (stack{i}.W'*delta{i+1}).*f_prime_z{i};
    % Compute the partial derivatives for use in Batch-Gradient Descent:
    nabla{i} = struct('W', delta{i+1}*a{i}', 'b', sum(delta{i+1},2));
    % TODO: Change this to be the summation over rows of b, so that b is a
    % (10 x 1).
    %nabla{i} = struct('b', sum(delta{i+1},2));
end
% Normalize by 1/m and add the learning rate (lambda*w^(l)):
nabla{1}.W = (1/num_samples)*nabla{1}.W + ei.lambda*stack{1}.W;
nabla{1}.b = (1/num_samples)*nabla{1}.b;
nabla{2}.W = (1/num_samples)*nabla{2}.W + ei.lambda*stack{2}.W;
nabla{2}.b = (1/num_samples)*nabla{2}.b;
gradStack{1} = nabla{1};
gradStack{2} = nabla{2};
% Compute the gradient 
% for layer = num_layers - 1:-1:1
    % gradient{layer,1} = gradient{layer,1} + nabla{layer,1};
    % gradient{layer,2} = gradient{layer,2} + nabla{layer,2};
% end
%% compute weight penalty cost and gradient for non-bias terms

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end