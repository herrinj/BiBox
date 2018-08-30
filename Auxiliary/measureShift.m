function [s]= measureShift(im1,im2)
%
%   Determines shift between two 2D images such that:
%           
%        im1(x) = im2(x - s)
%
%   Input:  im1 - image 1
%           im2 - shifted image 2
%
%   Output:   s - measured shift between the two images
%

% Get phase of two images
phase1  = exp(i*angle(fft2(im1)));
phase2  = exp(i*angle(fft2(im2)));

% Subtract phases and ifft2 back to images space
deltaShift  = real(fftshift(ifft2(conj(phase1).*phase2)));

% Maximum entry at peak of shift
dc = size(im1)./2 + 1;
[~,sInd]= max(deltaShift(:));
s       = zeros(1,2);
[s(1),s(2)] = ind2sub(size(im1),sInd);
s = s-dc; % Correct by one

end

