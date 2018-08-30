function [im_shift] = shiftImage(im, s)
%
%   Takes and input image im and shifts it by s with reflexive BCs
%           
%        imshift(x) = im(x - s)
%
%   Input:   im - input image
%             s - 1x2 shift
%
%   Output:  im_shift - shifted output image
%

% Pad image reflexively
[m,n]   = size(im);
im_pad  = padarray(im,[abs(s(1)) abs(s(2))], 'circular','both');

% Extract shifted image
im_s11  = zeros(1,2);
im_s11(1) = abs(s(1));
im_s11(2) = abs(s(2));
im_s11    = im_s11 + s + 1;

im_shift = im_pad(im_s11(1):im_s11(1)+m-1 ,im_s11(2):im_s11(2) + n-1);

end
