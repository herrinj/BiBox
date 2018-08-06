% 
%   This script tests the various optimization methods for matching the
%   object phase to the bispectrum phase
%
%   Author: James Herring, jlherri@emory.edu
%   Modified: 7/24/16
%

% Setup data and objective function
path_SpeckleImagingCodes;
setupBispectrumData;
image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*phase_recur(:)),[256 256])))));
its = 25;
tol = 1e-6;
avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));

%=============%
% phase_rec.m %
%=============%
[~,~,hess_const] = phase_rec(phase_recur(:),A,bispec_phase,weights,0,2);
obj_func = @(phase) phase_rec(phase,A,bispec_phase,weights,0,3,hess_const);
figure; subplot(2,3,1); imagesc(reshape(obj,[256 256])); axis image; colorbar; title('phase\_rec');
subplot(2,3,2); imagesc(reshape(image_recur,[256 256])/max(image_recur(:))); axis image; colorbar; title('recur');
subplot(2,3,3); imagesc(avg_data_frame); axis image; colorbar; title('avg. blurred frame');
% Run gradient descent
[x,flag, his_phase_gd, iters_phase_gd] = gradient_descent(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phase_gd = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));
subplot(2,3,4); imagesc(image_phase_gd/max(image_phase_gd(:))); axis image; colorbar; title('gd');
% Run NLCG
[x,flag, his_phase_nlcg, iters_phase_nlcg] = nlcg(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phase_nlcg = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));
subplot(2,3,5); imagesc(image_phase_nlcg/max(image_phase_nlcg(:))); axis image; colorbar; title('nlcg');
% Run damped Newton
[x,flag, his_phase_newt, iters_phase_newt] = damped_newton(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phase_newt = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));
subplot(2,3,6); imagesc(image_phase_newt/max(image_phase_newt(:))); axis image; colorbar; title('newton');

%==============%
% phasor_rec.m %
%==============%
[~,~,hess_const] = phasor_rec(phase_recur(:),A,bispec_phase,weights,0,2);
obj_func = @(phase) phasor_rec(phase,A,bispec_phase,weights,0,3,hess_const);
figure; subplot(2,3,1); imagesc(reshape(obj,[256 256])); axis image; colorbar; title('phasor\_rec');
subplot(2,3,2); imagesc(reshape(image_recur,[256 256])/max(image_recur(:))); axis image; colorbar; title('recur');
subplot(2,3,3); imagesc(avg_data_frame); axis image; colorbar; title('avg. blurred frame');
% Run gradient descent
[x,flag, his_phasor_gd, iters_phasor_gd] = gradient_descent(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phasor_gd = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));
subplot(2,3,4); imagesc(image_phasor_gd/max(image_phasor_gd(:))); axis image; colorbar; title('gd');
% Run NLCG
[x,flag, his_phasor_nlcg, iters_phasor_nlcg] = nlcg(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phasor_nlcg = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));
subplot(2,3,5); imagesc(image_phasor_nlcg/max(image_phasor_nlcg(:))); axis image; colorbar; title('nlcg');
% Run damped Newton
[x,flag, his_phasor_newt, iters_phasor_newt] = damped_newton(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phasor_newt = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));
subplot(2,3,6); imagesc(image_phasor_newt/max(image_phasor_newt(:))); axis image; colorbar; title('newton');

%===============%
% imphase_rec.m %
%===============%
figure; subplot(2,3,1); imagesc(reshape(obj,[256 256])); axis image; colorbar; title('imphase\_rec');
subplot(2,3,2); imagesc(reshape(image_recur,[256 256])/max(image_recur(:))); axis image; colorbar; title('recur');
subplot(2,3,3); imagesc(avg_data_frame); axis image; colorbar; title('avg. blurred frame');
% Run gradient descent
obj_func = @(image) imphase_rec(image,A,bispec_phase,weights, pupil_mask,100.0,'pos',pospec);
[image_imphase_gd,flag, his_imphase_gd, iters_imphase_gd] = gradient_descent(obj_func, image_recur(:), its, tol, 1);
image_imphase_gd = reshape(image_imphase_gd,[256 256]);
subplot(2,3,4); imagesc(image_imphase_gd/max(image_imphase_gd(:))); axis image; colorbar; title('gd');
% Run NLCG
obj_func = @(image) imphase_rec(image,A,bispec_phase,weights, pupil_mask,100.0,'pos',pospec);
[image_imphase_nlcg,flag, his_imphase_nlcg, iters_imphase_nlcg] = nlcg(obj_func, image_recur(:), its, tol, 1);
image_imphase_nlcg = reshape(image_imphase_nlcg,[256 256]);
subplot(2,3,5); imagesc(image_imphase_nlcg/max(image_imphase_nlcg(:))); axis image; colorbar; title('nlcg');
% Run damped Newton
obj_func = @(image) imphase_rec(image,A,bispec_phase,weights, pupil_mask,1000.0,'pos',pospec);
[image_imphase_newt,flag, his_imphase_newt, iters_imphase_newt] = damped_newton(obj_func, image_recur(:), its, tol, 1);
image_imphase_newt = reshape(image_imphase_newt,[256 256]);
subplot(2,3,6); imagesc(image_imphase_newt/max(image_imphase_newt(:))); axis image; colorbar; title('newton');

%================%
% imphasor_rec.m %
%================%
figure; subplot(2,3,1); imagesc(reshape(obj,[256 256])); axis image; colorbar; title('imphasor\_rec');
subplot(2,3,2); imagesc(reshape(image_recur,[256 256])/max(image_recur(:))); axis image; colorbar; title('recur');
subplot(2,3,3); imagesc(avg_data_frame); axis image; colorbar; title('avg. blurred frame');
% Run gradient descent
obj_func = @(image) imphasor_rec(image,A,bispec_phase, weights , pupil_mask, 100.0,'pos',pospec);
[image_imphasor_gd,flag, his_imphasor_gd, iters_imphasor_gd] = gradient_descent(obj_func, image_recur(:), its, tol, 1);
image_imphasor_gd = reshape(image_imphasor_gd,[256 256]);
subplot(2,3,4); imagesc(image_imphasor_gd/max(image_imphasor_gd(:))); axis image; colorbar; 
% Run NLCG
obj_func = @(image) imphasor_rec(image,A,bispec_phase, weights , pupil_mask, 100.0,'pos',pospec);
[image_imphasor_nlcg,flag, his_imphasor_nlcg, iters_imphasor_nlcg] = nlcg(obj_func, image_recur(:), its, tol, 1);
image_imphasor_nlcg = reshape(image_imphasor_nlcg,[256 256]);
subplot(2,3,5); imagesc(image_imphasor_nlcg/max(image_imphasor_nlcg(:))); axis image; colorbar; title('nlcg');
% Run damped Newton
obj_func = @(image) imphasor_rec(image,A,bispec_phase, weights , pupil_mask, 1000.0,'pos',pospec);
[image_imphasor_newt,flag, his_imphasor_newt, iters_imphasor_newt] = damped_newton(obj_func, image_recur(:), its, tol, 1);
image_imphasor_newt = reshape(image_imphasor_newt, [256 256]);
subplot(2,3,6); imagesc(image_imphasor_newt/max(image_imphasor_newt(:))); axis image; colorbar; title('newton');

% data_path = '/home/jlherri/Documents/MATLAB/Bispectral_Imaging/Data';
% his_file = fullfile(data_path, 'hisInfo_50_30_reg.mat');
% iter_file = fullfile(data_path, 'iterInfo_50_30_reg.mat');
%  
% save( his_file, 'his_phase_gd', 'his_phasor_gd', 'his_imphase_gd', 'his_imphasor_gd', 'his_phase_nlcg','his_phasor_nlcg', 'his_imphase_nlcg', 'his_imphasor_nlcg',...
%     'his_phase_newt', 'his_phasor_newt', 'his_imphase_newt', 'his_imphasor_newt');
%     
% save( iter_file, 'obj', 'pospec', 'iters_phase_gd', 'iters_phasor_gd', 'iters_imphase_gd', 'iters_imphasor_gd', 'iters_phase_nlcg', 'iters_phasor_nlcg',...
%     'iters_imphase_nlcg', 'iters_imphasor_nlcg', 'iters_phase_newt', 'iters_phasor_newt', 'iters_imphase_newt', 'iters_imphasor_newt');
