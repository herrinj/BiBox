%%
% Test methods for correcting shift
clear all; close all; 

x = double(imread('Satellite.jpg'));
y = padarray(x,[5 3],0,'both');
x = x + randn(size(x));
y = y + randn(size(y));
p = [2 1];
y = y(1+p(1):256+p(1), 1+p(2):256+p(2));

% Use Fourier phase to determine shift and shift image
s = measureShift(x,y);
z = imageShift(y,s);

% Show results
figure(1);
subplot(2,3,1); imagesc(x); axis image; axis off;
subplot(2,3,2); imagesc(y); axis image; axis off;
subplot(2,3,3); imagesc(x-y); axis image; axis off;
subplot(2,3,4); imagesc(z); axis image; axis off;
subplot(2,3,5); imagesc(x-z); axis image; axis off;