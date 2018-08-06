function [image, his] = gs_error_reduce(pospec, phase, maxIter, tol, visualizer)
%
%   [image, his] = gerchberg_saxton(pospec, phase, maxIter, visualizer)
%
%   This implements a Gerchberg-Saxton type error reduction algorithm which
%   projects between the space of images with a known power spectrum 
%   (Fourier Domain) and non-negative, real entries. 
%
%   Source: Negrete-Regagnon (1996) and Fienup
%
%   Inputs: pospec - object power spectrum, expected NxN
%            phase - object phase, expected NxN
%          maxIter - maximum number of iterations
%       visualizer - tolerance for stopping criteria
%
%   Output: image - resulting image
%             his - 
%

if nargin < 2
    runMinimalExample;
    return;
end

if isempty(maxIter)
    maxIter = 10;
end

if isempty(tol)
    tol = 1e-4;
end

if nargin <5
    visualizer = 0;
end

iter = 0;
his = zeros(iter,1);
image = real(fftshift(ifft2(fftshift(pospec.*exp(i*phase)))));
vol = sum(image(:));

if visualizer
    figure; subplot(1,2,1); imagesc(image); colorbar; axis image; title('Input Image');
end
    
while iter < maxIter
   iter = iter + 1;
   
   % First we project into the Fourier domain and replace the Fourier
   % modulus with our known Fourier modulus (power spectrum)
   IMAGE = fftshift(fft2(fftshift(image)));
   phasor = exp(i*angle(IMAGE));
   
   % Next, we project back to the space of non-negative, real images
   image = real(fftshift(ifft2(fftshift(pospec.*phasor))));
   image(image < 0) = 0;
   %image = gdnnf_projection(image, vol);
   his(iter,1) = norm(pospec- abs(fftshift(fft2(fftshift(image)))))/norm(pospec);
   
   if visualizer
        str = sprintf('Iteration %d Tolerance %f', iter, his(iter,1));
        subplot(1,2,2); imagesc(image); colorbar; axis image; title(str); pause();
   end
        
   % Check for convergence
   if his(iter,1) < tol
       his = his(1:iter,:);
       return;
   end
end


end

function runMinimalExample
    setupBispectrumData;
    [image_gs, his] = gs_error_reduce(reshape(pospec,[256 256]), reshape(phase_recur,[256 256]), [], [], 1);
end