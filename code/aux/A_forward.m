function [y] = A_forward(x, scaling_factor,M,N)

% This function computes Ax and returns the LR image b. In this case, the LR image is obtained 
% using the Matlab function imresize to downscale the image using the bicubic kernel.

x = reshape(x,M,N);
y = imresize(x,1/scaling_factor);
y= y(:);

end

