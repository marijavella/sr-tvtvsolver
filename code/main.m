function [] = main(path_side_info, path_groundtruth, ...
        scaling_factor, filename_results,  ...
        varargin)

% function [] = main(path_side_info, path_groundtruth, ...
%                    scaling_factor, filename_results, varargin)
%
% Loops through the images in the folders 'path_side_info' and
% 'path_groundtruth', which should exist in equal numbers and 
% have the same order in the corresponding folders, and applies 
% the post-processing step proposed in 
%
%     M. Vella, J. F. C. Mota
%     Single Image Super-Resolution via CNN Architectures and 
%     TV-TV Minimization
%     BMVC, 2019
%     https://arxiv.org/abs/1907.05380
% 
% Note: To ensure that the same image in the respective folders correspond 
% to eachother, all the image names consist of the image number followed by the
% type of image (e.g., GT, HR) which are then accessed in ascending order.
%
% That post-processing step consists of solving a TV-TV minimization
% problem
%
%     minimize    TV(x) + β TV(x - w)
%        x
%     subject to  A*x = b,
%
% where β is a trade-off parameter which we set to 1, w is the output
% of a given super-resolution algorithm, e.g., a CNN, assumed stored
% as an image file in the folder 'path_side_info', and A is a
% downsampling operator (default: Matlab's imresize) which is used
% to generate a low-resolution image b from the ground truth images 
% asumed stored as an image file in the folder 'path_groundtruth'.
%
% Inputs:
%   - 'path_side_info': string containing the path to a directory
%      with images corresponding to w, i.e., the output of a
%      super-resolution algorithm. To use the available outputs for
%      example SRCNN with a scaling factor of 4 for Set5, the path
%      needs to be set to images/Test_Sets_Side/SRCNNNSet5x4. If 
%      considering the outputs from DRCN for a sclaing factor of 2
%      for BSD100, the path is images/Test_Sets_Side/DRCNBSD100x2 etc.    
%
%   - 'path_groundtruth': string containing the path to a directory
%      with the ground truth (high-resolution) images which are used to 
%      obtain the low-resolution image b. For instance, to use the 
%      available ground truth images of Set5 for a scaling factor of 2, 
%      the path should read images/Test_Sets_GT/Set5_x2_GT. Note that, 
%      the ground truth images are the same for each algorithm, it just 
%      differs for different scaling factors. 
%
%   - 'scaling_factor': whole numbers between 1 and 8 corresponding
%      to the upscaling factor. In our paper, we only consider
%      scaling factors x2 and x4 thus we only provide the outputs
%      for these factors. To use other scaling factor store the
%      respective images. Note that, for example, a scaling_factor 
%      of 2 means 4 times more pixels, as both the width and height
%      are augmented by 2.
%
%   - 'filename_results': string with the name of the file where the
%      results (PSNR and SSIM) will be stored. 
%  
% Optional inputs: pair 'string', value, where 'string' can be
%   - 'GPU': 0 or 1. If 1, the function uses code optimized for GPUs
%     (default: 0)
%
%   - 'SHOW_IMAGES': 0 or 1: If 1, processed images are displayed.
%     (default: 0)
%
%   - 'SHOW_RESULTS': 0 or 1: If 1, display PSNR and SSIM values.
%     (default: 0)
%
%   - 'A_forward': function handle that implements A*x, where x is a
%     vectorized version of a high-resolution image.
%     (default: the function in aux/A_forward.m, which uses Matlab's
%     imresize)
%
%   - 'A_transpose': function handle that implements A'*y, where y is
%     a vectorized version of a low-resolution image. 
%     (default: the function in aux/A_transpose.m, which uses Matlab's
%     imresize)
%


% Path to auxiliary functions
addpath('aux/')

% ======================================================
% Check inputs

% ------------------------------------------------------
% Check that 'path_side_info' and 'path_groundtruth' have the same
% number of images
files_path_side_info = [...
    dir([path_side_info, '/*.png']), ...
    dir([path_side_info, '/*.jpg'])  ...
    ];

files_path_groundtruth = [...
    dir([path_groundtruth, '/*.png']), ...
    dir([path_groundtruth, '/*.jpg'])  ...
    ];

if length(files_path_side_info) ~= length(files_path_groundtruth)
    error('Specified folders should have the same number of images')
end

if scaling_factor <= 1 || scaling_factor >= 9
    error('scaling_factor has to be between 1 and 8')
end

% *********
% Defaults
% *********
GPU = 0;
SHOW_IMAGES = 0;
SHOW_RESULTS = 0;
A_h  = @A_forward;
AT_h = @A_transpose;

% Read optional input
if (rem(length(varargin),2) == 1)
    error('Optional parameters should always go in pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})            
            case 'GPU'
                GPU = varargin{i+1};
            case 'SHOW_IMAGES'
                SHOW_IMAGES = varargin{i+1};
            case 'SHOW_RESULTS'
                SHOW_RESULTS = varargin{i+1};
            case 'A_forward'
                A_h = varargin{i+1};
            case 'A_transpose'
                AT_h = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end
% ======================================================

% ======================================================
% Main loop over all images

try
    % Images to be used as side information (w)
    DNNs = ReadImages(path_side_info);

    % Ground truth images (x)
    HR = ReadImages(path_groundtruth);
catch
    error('Something wrong with the folders used as input to main.m')
end

num_images = numel(HR);   % Number of images in the folders

% check if the images have the same size in the correspondig folders
for n = 1:1:num_images
   check(n) =  isequal(size(DNNs(n).data),size(HR(n).data)); 
end

if all(check) == 0
    fprintf('The images in the different folders do not correspond to eachother.')
end

psnr_tvtv = zeros(1, num_images);
ssim_tvtv = zeros(1, num_images);
psnr_cnn  = zeros(1, num_images);
ssim_cnn  = zeros(1, num_images);

for j = 1 : num_images

    % Image w
    im_w.out = DNNs(j).data;

    % If it has more than one channel (assumed RGB),
    % convert image to YCbCr color space and select Y channel 
    if size(im_w.out,3) > 1
        im1 = rgb2ycbcr(im_w.out);
        im_w.out = im1(:, :, 1);
    end

    % Highres image (keep all channels)
    im_HR.out = HR(j).data;

    % Create ground truth image by keeping only Y channel
    if size(im_HR.out,3) >1
        im1 = rgb2ycbcr(im_HR.out);
        im_gt.out = im1(:, :, 1);
    end

    % Dimensions of image
    [M, N] = size(im_gt.out); 
    n = M*N;

    % ----------------------------------------------
    % Vectorize images and rescale entries to [0, 1]

    % w
    w = reshape(im_w.out, [], 1); 
    w = double(w)/255; % normalizing the intensity values

    % x
    x = reshape(im_gt.out, [], 1); 
    x = double(x)/255; % normalizing the intensity values
    % ----------------------------------------------

    % Obtain the LR image b by sampling x
    b = A_h(x,scaling_factor,M,N); 

    % ----------------------------------------------
    %% Post-processing step using TVTVSolver

    beta = 1; 
    
    if GPU
        [x_ADMM, k_ADMM] = TVTV_Solver_GPU(M, N, b, w, beta, A_h, AT_h, scaling_factor);
    else
        [x_ADMM, k_ADMM] = TVTV_Solver_CPU(M, N, b, w, beta, A_h, AT_h, scaling_factor);
    end

    fprintf('Image %i processed \n',j)
    % ----------------------------------------------
    
    % ----------------------------------------------
    %% Reshape from vector to matrix and rescale entries to [1,255]
    x_hat = (reshape(x_ADMM,M,N)); 
    x_hat = uint8(x_hat*255); 
    % ----------------------------------------------

    if SHOW_IMAGES
        plotimages(j,im_HR.out, im_gt.out,im_w.out, x_hat, scaling_factor)
    end
    
    %% Compute the PSNR and SSIM values 
    [psnr_tvtv(j), ssim_tvtv(j)] = compute_diff(x_hat,         im_gt.out, scaling_factor);
    [psnr_cnn(j), ssim_cnn(j)]   = compute_diff(im_w.out, im_gt.out, scaling_factor);

    % Print results
    if SHOW_RESULTS
        fprintf('Image %2d: TVTV - PSNR: %2.4f dB - SSIM: %2.4f \n', j, psnr_tvtv(j), ssim_tvtv(j))
        fprintf('        : CNN  - PSNR: %2.4f dB - SSIM: %2.4f \n', psnr_cnn(j) , ssim_cnn(j))
    end
    
end

% ======================================================
% Compute the mean over all images of PSNR and SSIM
psnr_tvtv_mean = mean(psnr_tvtv);
psnr_cnn_mean  = mean(psnr_cnn);
ssim_tvtv_mean = mean(ssim_tvtv);
ssim_cnn_mean  = mean(ssim_cnn);

%% Storing the results for each image and saving the results in a matrix 
save(filename_results, 'psnr_tvtv', 'psnr_cnn', 'ssim_tvtv', 'ssim_cnn',  ...
'psnr_tvtv_mean', 'psnr_cnn_mean', 'ssim_tvtv_mean', 'ssim_cnn_mean') 
% ======================================================

