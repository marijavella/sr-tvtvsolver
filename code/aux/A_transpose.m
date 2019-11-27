function [x] = A_transpose(y, scaling_factor,M,N)

% This function computes AT*x. Here, the input image is upscaled using the Matlab function 
% imresize and using the bicubic kernel.

y = reshape(y,M,N);
x= imresize(y,scaling_factor);
x= (x(:));

end

