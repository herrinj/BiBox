function [blurred_image_frames, true_image] = GenerateSpeckleData(speckle_frames, image_file, pad_flag)
%
%   Provided by Jim Nagy
%
%   Inputs: 
%       speckle_frames - 3 dimensional array of speckle star data to blur 
%                        the true image
%           image_file - file path of image you want blurred by each of the speckle frames
%             pad_flag - 0 says no to padding, 1 says yes to padding
%
%   Outputs: 
%  blurred_image_frames - 3 dimensional array where the kth frame 
%                         corresponds to true_image blurred by the kth speckle frame
%            true_image - as in the input. If unspecified in input, uses
%                         generic satellite
%

if nargin<3
    pad_flag = 0;
end

if isempty(image_file)
    image_file = 'HST.jpg';
end

% Import, resize, and scale image from .jpg file, accounting for color or
% grayscale images
true_image = imread(image_file);
if size(true_image,3) < 2
    true_image = double(true_image); 
else 
    true_image = double(rgb2gray(true_image)); 
end

if pad_flag
    true_image = imresize(true_image, [192,192]);
    true_image = padarray(true_image, [32 32]);
else
    true_image = imresize(true_image, [256,256]);
end
true_image = true_image/sum(true_image(:));

% Blur each image frame using the speckle frames
[m,n,n_frames] = size(speckle_frames);
blurred_image_frames = zeros(256,256,size(speckle_frames,3));
true_image_fft = fft2(true_image);
for k = 1:n_frames
    tt = real(ifft2(fft2(fftshift(speckle_frames(:,:,k))) .* true_image_fft));
    tt = padarray(tt,[32 32]);
    x_shift = randi([-16 16],1); % Added for shifts in the data to simulate satellite tracking
    y_shift = randi([-16 16],1); 
    tt = tt(33+y_shift:m+32+y_shift, 33+x_shift:n+32+x_shift);  
    blurred_image_frames(:,:,k) = reshape(tt,256,256);
end
