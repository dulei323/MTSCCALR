
close all; clear; clc;
% ----------------------------------------
% Author: Lei Du, dulei@nwpu.edu.cn
% Date: 09-Dec-2020
% ----------------------------------------

% libsvm for classification
addpath(genpath('libsvm-3.24'));

%% Load data
load('synData.mat');

X = Data.X; [n, p] = size(X);
Y = Data.Y; [~, q] = size(Y);
label = Data.label;
n_class = length(unique(label));

Z = zeros(n, n_class);
for c = 1 : n_class
    Z(label == c, c) = 1;
end

%% Set tuned parameters
% for weight U
opts.lambda_u1 = 0.01;  % L2,1-norm
opts.lambda_u2 = 1;     % L1,1-norm
opts.lambda_u3 = 0.2;   % FGL-norm
% for weight V
opts.lambda_v1 = 0.01;  % L2,1-norm
opts.lambda_v2 = 1;     % L1,1-norm
opts.lambda_v3 = 0.001; % GGL-norm

trainData.n_class = n_class;
testData.n_class = n_class;

%% Kfold cross validation
k_fold = 5;
indices = crossvalind('Kfold', n, k_fold);

for k = 1 : k_fold
    fprintf('[conduct fold %d ', k);
    
    %% Split training data and test data
    idx_test = (indices == k);
    idx_train = ~idx_test;
    % training sets
    trainData.X = X(idx_train, :);
    trainData.Y = Y(idx_train, :);
    trainData.Z = Z(idx_train, :);
    % testing sets
    testData.X = X(idx_test, :);
    testData.Y = Y(idx_test, :);
    testData.Z = Z(idx_test, :);
    
    %% Train model
    tic;
    [U(:, :, k), V(:, :, k)] = MTSCCALR(trainData, opts);
    time(k, 1) = toc;
    
    %% Calculate canonical correlation coefficients (CCCs)
    CCCs_train(k, :) = calcCCC(trainData, U(:, :, k), V(:, :, k));
    CCCs_test(k, :) = calcCCC(testData, U(:, :, k), V(:, :, k));
    
    fprintf('(%.2fs)]\n', time(k));
end

%% Canonical weights
U_mean = mean(U, 3);
V_mean = mean(V, 3);

%% Correlation
% Row 1: training
CCCs_mean(1, :) = mean(CCCs_train);
CCCs_std(1, :) = std(CCCs_train, 1);
% Row 2: testing
CCCs_mean(2, :) = mean(CCCs_test);
CCCs_std(2, :) = std(CCCs_test, 1);

%% Classification
top_K_u = 10;
top_K_v = 10;
for k = 1 : k_fold
    idx_test = (indices == k);
    idx_train = ~idx_test;
    % training sets
    trainData.X = X(idx_train, :);
    trainData.Y = Y(idx_train, :);
    trainData.Z = Z(idx_train, :);
    % conduct oversample
    [trainData.X, trainData.Y, trainData.Z] = do_oversample(trainData);
    % testing sets
    testData.X = X(idx_test, :);
    testData.Y = Y(idx_test, :);
    testData.Z = Z(idx_test, :);
    
    % SVM: LIBSVM tool
    for c = 1 : n_class
        [~, idx_feature_u] = sort(abs(U(:, c, k)), 'descend');
        [~, idx_feature_v] = sort(abs(V(:, c, k)), 'descend');
        % train SVM model
        label_train = trainData.Z{c};
        label_train(label_train == 0) = -1;
        SVMModel = svmtrain(label_train, ...
            [trainData.X{c}(:, idx_feature_u(1 : top_K_u)), ...
            trainData.Y{c}(:, idx_feature_v(1 : top_K_v))]);
        % testing sets
        label_test = testData.Z(:, c);
        label_test(label_test == 0) = -1;
        label_pred = svmpredict(label_test, ...
            [testData.X(:, idx_feature_u(1 : top_K_u)), ...
            testData.Y(:, idx_feature_v(1 : top_K_v))], SVMModel);
        Class_test(k, c) = sum(label_pred == label_test) / length(label_pred);
        % training sets
        label_pred = svmpredict(label_train, ...
            [trainData.X{c}(:, idx_feature_u(1 : top_K_u)), ...
            trainData.Y{c}(:, idx_feature_v(1 : top_K_v))], SVMModel);
        Class_train(k, c) = sum(label_pred == label_train) / length(label_pred);
    end
end

% Row 1: training
Class_mean(1, :) = mean(Class_train);
Class_std(1, :) = std(Class_train, 1);
% Row 2: testing
Class_mean(2, :) = mean(Class_test);
Class_std(2, :) = std(Class_test, 1);

%% Draw figures
figure; colormap('Jet');
% Ground Truth
caxis_range = 2;
subplot(2, 2, 1); imagesc(Data.U', [-caxis_range caxis_range]); title('U');
subplot(2, 2, 2); imagesc(Data.V', [-caxis_range caxis_range]); title('V');
colorbar('Ticks', [-caxis_range 0 caxis_range]);
% MT-SCCALR
caxis_range = 0.2;
subplot(2, 2, 3); imagesc(U_mean', [-caxis_range caxis_range]);
subplot(2, 2, 4); imagesc(V_mean', [-caxis_range caxis_range]);
colorbar('Ticks', [-caxis_range 0 caxis_range]);

rmpath(genpath('libsvm-3.24'));
