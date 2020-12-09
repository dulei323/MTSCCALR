function CCCs = calcCCC(Data, U, V)

% calculate Canonical Correlation Coefficient

X = Data.X;
Y = Data.Y;

for c = 1 : Data.n_class
    CCCs(c) = corr(X * U(:, c), Y * V(:, c));
end
