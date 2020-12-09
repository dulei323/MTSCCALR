function [U, V] = MTSCCALR(data, opts)

% Multi-task sparse canonical correlation analysis and logistic regression

[~, p] = size(data.X); % SNP data
[~, q] = size(data.Y); % Img data
n_class = data.n_class;

% SNP weights
U = ones(p, n_class);
% Img weights
V = ones(q, n_class);

% class balancing
[X, Y, Z] = do_oversample(data); % oversample

for c = 1 : n_class
    % normalization
    X{c} = normalize(X{c}, 'norm');
    Y{c} = normalize(Y{c}, 'norm');
    
    % number of samples for each task
    [n_sample(c), ~] = size(X{c});
    
    % Pre-calculate covariance
    XX{c} = X{c}' * X{c};
    XY{c} = X{c}' * Y{c};
    YY{c} = Y{c}' * Y{c};
    YX{c} = XY{c}';
    
    % Scale U and V
    U(:, c) = U(:, c) / norm(X{c} * U(:, c));
    V(:, c) = V(:, c) / norm(Y{c} * V(:, c));
end

% Set tuned parameters
lambda_u1 = opts.lambda_u1; % L2,1
lambda_u2 = opts.lambda_u2; % L1,1
lambda_u3 = opts.lambda_u3; % FGL

lambda_v1 = opts.lambda_v1; % L2,1
lambda_v2 = opts.lambda_v2; % L1,1
lambda_v3 = opts.lambda_v3; % GGL

gamma_u = 0;
gamma_v = 0;

% Set iterative termination criterion
max_Iter = 100;
t = 0;

tol = 1e-5;
tu = inf;
tv = inf;

% Iteration
while (t < max_Iter && (tu > tol || tv > tol))
    t = t + 1;
    
    U_old = U;
    V_old = V;
    
    % Update V
    D1 = updateD(V); % L2,1-norm
    for c = 1 : n_class
        D1c = updateD(V(:, c)); % L1-norm
        Dg = updateD(V(:, c), 'GGL'); % GGL-norm
        
        % solve each v_c
        Yv = Y{c} * V(:, c);
        api = 1 ./ (1 + exp(-Yv));
        
        g = Y{c}' * (api - Z{c}) / n_sample(c) - 2 * YX{c} * U(:, c) + 2 * lambda_v1 * D1 * V(:, c) + 2 * lambda_v2 * D1c * V(:, c) ...
            + 2 * lambda_v3 * Dg * V(:, c) + 2 * (gamma_v + 1) * YY{c} * V(:, c); % Gradient
        
        w = (1 - api) .* api;
        W = diag(w);
        H = Y{c}' * W * Y{c} / n_sample(c) + 2 * lambda_v1 * D1 + 2 * lambda_v2 * D1c + 2 * lambda_v3 * Dg + 2 * (gamma_v + 1) * YY{c}; % Hessian
        
        V(:, c) = V(:, c) - H \ g;
        
        % scale each v_c
        V(:, c) = V(:, c) / norm(Y{c} * V(:, c));
    end
    
    % Update U
    D2 = updateD(U); % L2,1-norm
    for c = 1 : n_class
        D2c = updateD(U(:, c)); % L1-norm
        Df = updateD(U(:, c), 'FGL'); % FGL-norm
        
        % solve each u_c
        F1 = lambda_u1 * D2 + lambda_u2 * D2c + lambda_u3 * Df + (gamma_u + 1) * XX{c};
        b1 = XY{c} * V(:, c);
        U(:, c) = F1 \ b1;
        
        % scale each u_c
        U(:, c) = U(:, c) / norm(X{c} * U(:, c));
    end
    
    % Iterative termination
    tu = max(max(abs(U - U_old)));
    tv = max(max(abs(V - V_old)));
end

function D = updateD(W, type)

if nargin == 1
    % for L2,1-norm & L1,1-norm
    d = 1 ./ sqrt(sum(W .^ 2, 2) + eps);
elseif strcmpi(type, 'FGL')
    % for FGL-norm
    [n_features, ~] = size(W);
    structure = updateGraph(n_features, 'FGL');
    Gp = 1 ./ sqrt(structure * (W .^ 2) + eps);
    d = [Gp(1), sum(reshape(Gp(2 : end - 1), 2, [])), Gp(end)];
elseif strcmpi(type, 'GGL')
    % for GGL-norm
    [n_features, ~] = size(W);
    structure = updateGraph(n_features, 'GGL');
    Gp = 1 ./ sqrt(structure * (W .^ 2) + eps);
    d = sum(reshape(Gp, n_features - 1, []));
else
    error('Error type.');
end

D = diag(0.5 * d);

function E = updateGraph(n, type)

if strcmpi(type, 'FGL')
    E = zeros(2 * (n - 1), n);
    for i = 1 : n - 1
        j = i + 1;
        E(2 * i - 1, i) = 1;
        E(2 * i - 1, j) = 1;
        E(2 * i, i) = 1;
        E(2 * i, j) = 1;
    end
elseif strcmpi(type, 'GGL')
    num = 0;
    E = zeros(n * (n - 1), n);
    for i = 1 : n
        for j = 1 : n
            if i ~= j
                num = num + 1;
                E(num, i) = 1;
                E(num, j) = 1;
            end
        end
    end
else
    error('Error type.');
end
