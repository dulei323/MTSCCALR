function [X, Y, Z] = do_oversample(rawData)

% Oversample for class balancing

X = rawData.X; Y = rawData.Y; Z = rawData.Z;

for c = 1 : size(Z, 2)
    idx_p = find(Z(:, c) == 1);
    num_p = length(idx_p);
    idx_n = find(Z(:, c) == 0);
    num_n = length(idx_n);
    
    if num_p > num_n
        idx_oversample = idx_n(randi([1, num_n], num_p - num_n, 1));
    elseif num_n > num_p
        idx_oversample = idx_p(randi([1, num_p], num_n - num_p, 1));
    else
        idx_oversample = [];
    end
    
    tempX{c} = [X; X(idx_oversample, :)];
    tempY{c} = [Y; Y(idx_oversample, :)];
    tempZ{c} = [Z(:, c); Z(idx_oversample, c)];
end

X = tempX; Y = tempY; Z = tempZ;
