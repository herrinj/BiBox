% 
%   This script tests the various optimization methods in order to
%   determine a regularization parameter
%
%   Author: James Herring, jlherri@emory.edu
%   Modified: 8/2/16
%

% Setup data and objective function
path_SpeckleImagingCodes;
setupBispectrumData;

image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*phase_recur(:)),[256 256])))));
its = 250;
tol = 1e-6;

alpha = logspace(-5,5,11);
variance = sigma_rn.^2;
image_imphasor_gd_alpha = zeros(prod(size(image_recur)),length(alpha));
image_imphasor_nlcg_alpha = zeros(prod(size(image_recur)),length(alpha));
image_imphasor_newt_alpha = zeros(prod(size(image_recur)),length(alpha));
res_alpha_gd = zeros(length(alpha),1);
res_alpha_nlcg = zeros(length(alpha),1);
res_alpha_newt = zeros(length(alpha),1);
image_norm_gd = zeros(length(alpha),1);
image_norm_nlcg = zeros(length(alpha),1);
image_norm_newt = zeros(length(alpha),1);


%================%
% imphasor_rec.m %
%================%
for k = 1:length(alpha)
    fprintf('alpha = %1.2e \n', alpha(k));
    obj_func = @(image) imphasor_rec(image, A, bispec_phase, weights , pupil_mask, alpha(k),'pos',pospec);
    [image_imphasor_gd,flag, his_imphasor_gd, iters_imphasor_gd] = gradient_descent(obj_func, image_recur(:), its, tol, 1);
    [image_imphasor_nlcg,flag, his_imphasor_nlcg, iters_imphasor_nlcg] = nlcg(obj_func, image_recur(:), its, tol, 1);
    [image_imphasor_newt,flag, his_imphasor_newt, iters_imphasor_newt] = damped_newton(obj_func, image_recur(:), its, tol, 1);
    image_imphasor_gd_alpha(:,k) = image_imphasor_gd(:);
    image_imphasor_nlcg_alpha(:,k) = image_imphasor_nlcg(:);
    image_imphasor_newt_alpha(:,k) = image_imphasor_newt(:);
    image_norm_gd(k) = norm(image_imphasor_gd(:));
    image_norm_nlcg(k) = norm(image_imphasor_nlcg(:));
    image_norm_newt(k) = norm(image_imphasor_newt(:));
end

for k = 1:length(alpha)
    [res_alpha_gd(k),~,~] = imphasor_rec(image_imphasor_gd_alpha(:,k),A,bispec_phase, weights , pupil_mask, 0.0,'pos',pospec);
    [res_alpha_nlcg(k),~,~] = imphasor_rec(image_imphasor_nlcg_alpha(:,k),A,bispec_phase, weights , pupil_mask, 0.0,'pos',pospec);
    [res_alpha_newt(k),~,~] = imphasor_rec(image_imphasor_newt_alpha(:,k),A,bispec_phase, weights , pupil_mask, 0.0,'pos',pospec);
end

data_path = '/home/jlherri/Documents/MATLAB/Bispectral_Imaging/Data';
reg_file = fullfile(data_path, 'regInfo_50_30_imphasor_c.mat');

save( reg_file, 'alpha', 'variance', 'image_recur', 'image_imphasor_gd_alpha', 'image_imphasor_nlcg_alpha', 'image_imphasor_newt_alpha', 'image_norm_gd', 'image_norm_nlcg',...
    'image_norm_newt','res_alpha_gd', 'res_alpha_nlcg', 'res_alpha_newt');

