function [power_spec] = powerspec_accum(DATA, STAR, K_n, alpha_rn)
%
%   This 
%
%   Input: DATA - fft2's of image frames from which to accumulate the ensemble average
%                 of the modulus squared. Given in a 3D array
%          STAR - fft2's of star frames from which to accumulate the ensemble average
%                 of the modulus squared PSF. Given in a 3D array
%           K_n - number of photoevents per frame
%      alpha_rn - standard deviation of zero-mean white Gaussian read noise from CCD device
%
%
%   Output: powerspec - object's unbiased computed power spectrum  
%

if nargin < 2
    runMinimalExample;
    return;
end

if nargin < 3
    K_n = 0;
    alpha_rn = 0;
end


[m,n,nfr] = size(DATA);
nfr_st = size(STAR,3);

% Unbias the energy spectra of the object DATA and the STAR
DATA_avg = abs(DATA).^2;
STAR_avg = abs(STAR).^2;

% Accumulate the ensemble average of the modulus squared of the data frames
% Also, correct the data bias for read noise
DATA_avg = sum(DATA_avg,3)/nfr;

% Accumulate the ensemble average of the modulus squared of the star/PSF frames
STAR_avg = sum(STAR_avg,3)/nfr_st;

eps = 1e-16; % prevents division by 0
power_spec = sqrt(DATA_avg./(STAR_avg + eps));

function runMinimalExample

% Data parameters
nfr = 50; % number of data frames
D_r0 = 30; % D/r0 value (10,20,30,40,50...)
K_n = 3e6; % number of photoevents per frame (estimate for simulated data)
alpha_rn = 5; % standard deviation of zero-mean white Gaussian read noise from CCD device

% Generate star speckle data and blurred object data, courtesy of Jim Nagy
[ ~ , star, pupil_mask, ~] = GenerateSpecklePSFs(256,nfr,D_r0,[]);
[data, ~] = GenerateSpeckleData(star, 'Satellite.jpg');
data = scale_and_noise(data, K_n, alpha_rn);
star = scale_and_noise(star, K_n, alpha_rn); % How many photons per star?

% Move all the data to the Fourier Domain
DATA = fftshift(fft2(fftshift(data)));
STAR = fftshift(fft2(fftshift(star)));

% Computer power spectrum 
pospec = powerspec_accum(DATA, STAR, K_n, alpha_rn);
pospec_pupil = pospec.*pupil_mask;

figure; subplot(1,3,1); imagesc(log(pospec)); axis image; colorbar; title('Power Spectrum');
subplot(1,3,2); imagesc(log(pospec_pupil)); axis image; colorbar; title('Power Spectrum + Pupil Mask');
subplot(1,3,3); semilogy(diag(pospec)); hold on; semilogy(diag(pospec_pupil)); 
title('Cross section'); legend('Power Spectrum', 'Power Spectrum + Pupil Mask','Location','southoutside'); axis square; axis tight;


