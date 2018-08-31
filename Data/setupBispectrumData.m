% 
%   This script creates simulated image and speckle star data for a basic
%   problem in bispectral imaging, and then accumulates the power spectrum,
%   bispectrum, recursive phase recovery, and recursive image recovery
%   necessary to run objective function gradient tests and optimization
%   schemes in the 'Optimization' folder.
%         
%   Author: James Herring, jlherri@emory.edu
%   Modified: 9/7/17
%

% Check to see if the data has been generated previously and saved. If so
file_name = strcat(image_name,sprintf('_%d_%d.mat', nfr, D_r0));
data_path = fileparts(which(mfilename('fullpath')));
full_name = fullfile(data_path,file_name);
file_OK = exist(full_name,'file');
check_save = 0;
if file_OK && check_save
    load(full_name);
    return;
end


% Generate star speckle data and blurred object data, courtesy of Jim Nagy
[star_phase, star, pupil_mask, scaled_pupil_mask] = GenerateSpecklePSFs(256,nfr,D_r0,randi(1e8));
[data, obj] = GenerateSpeckleData(star, strcat(image_name,'.jpg'),0);
[data, avg_noise_norm] = scale_and_noise(data, K_n, sigma_rn);
obj = obj/max(obj(:));
[star_phase, star, pupil_mask, scaled_pupil_mask] = GenerateSpecklePSFs(256,nfr,D_r0,randi(1e8));
star = scale_and_noise(star, 5000.0, 1); % How many photons per star?

OBJ = fftshift(fft2(fftshift(obj)));
%ShowSpeckleData(star_phase, star, data);


% Move all the data to the Fourier Domain
DATA = fftshift(fft2(fftshift(data)));
STAR = fftshift(fft2(fftshift(star)));

% Generate the index structure to vectorize the bispectrum operations
N = size(data,1); % expects square images
fourier_rad = 64; 
second_rad = 5;
pupil_mask0 = MakeMask(fourier_rad*2,1);
pupil_mask = padarray(pupil_mask0, [(256 - size(pupil_mask0,1))/2, (256 - size(pupil_mask0,2))/2], 'both');
[b,~] = bindex(N,fourier_rad, second_rad,0); % Change 4th argument to 1 for visualization
A = phi_matrix(b, 256^2);

% Accumulate the objects power/energy spectrum in the Fourier domain
pospec = powerspec_accum(DATA, STAR, K_n, sigma_rn);
pospec_true = abs(fftshift(fft2(fftshift(obj))));
pospec = pospec.*pupil_mask;
pospec_true = pospec_true.*pupil_mask;
phase_true = angle(OBJ).*pupil_mask;

% Accumulate bispectrum
[bispec, bispec_phase, weights] = bispec_accum(DATA, b);

[star_bispec, star_bispec_phase,~] = bispec_accum(STAR, b);
%bispec = bispec./star_bispec;
bispec_true = phase_true(b.u) + phase_true(b.v) - phase_true(b.u_v);

% Recursive reconstruction
phasor_recur = recur_phase(exp(1i*bispec_phase), b, N);
phase_recur = angle(phasor_recur);
phase_recur = phase_foldout(phase_recur, 0);
IMAGE_recur = pospec.*exp(1i*phase_recur);
image_recur = real(fftshift(ifft2(fftshift(IMAGE_recur))));


save(full_name,'data','star','DATA','STAR','fourier_rad','second_rad','b','A','obj','pupil_mask','pospec','bispec_phase','bispec_true','phase_recur','phase_true','weights','K_n', 'sigma_rn')
