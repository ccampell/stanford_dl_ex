function [ sig_z ] = sigmoid(z)
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
%% Calculate sigmoid of input vector: f(z)
sig_z = 1./(1 + exp(-z));
end

