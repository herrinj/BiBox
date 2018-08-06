%  graphing
clear, clc

%
% Initialize
%
N = 800; % the number of pixels in one dimension
D = 0.5; % the diameter of the telescope [m]
Cn2 = 1e-16; % the structure parameter
nlayers = 2; % the number of turbulent layers.

[phi, rat] = GenerateWavefront(N, D, Cn2, nlayers);
[mf, nf, nFrames] = size(phi);
pupil_mask = MakeMask(nf,0.5);

%
% compute the PSFs
%
for iter = 1:nFrames
  PSFt = abs( ifft2(pupil_mask.*exp(sqrt(-1)*phi(:,:,iter)))/sqrt(mf*nf) ).^2;
  center = round([mf/2, nf/2]);
  PSFt = circshift(PSFt, 1-center);
  PSF(:,:,iter) = PSFt/sum(PSFt(:));
end

%
% show the PSFs and the wavefront phase
%
[X, Y] = meshgrid(1:mf, 1:nf);
for iter = 1:nFrames
    figure, imshow(PSF(:,:,iter),[]), colormap(jet), colorbar, title(['The PSF of ', num2str(iter), ' layer'])
    figure, mesh(X,Y,PSF(:,:,iter)), colormap(jet),  title(['The PSF of ', num2str(iter), ' layer'])
    figure, imshow(phi(:,:,iter),[]), colormap(jet), colorbar,  title(['The wavefront phase of ', num2str(iter), ' layer'])
end
