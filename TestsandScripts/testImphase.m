% 
%   This script tests the various optimization methods for matching the
%   object phase to the bispectrum phase
%
%   Author: James Herring, jlherri@emory.edu
%   Modified: 9/19/16
%

% Setup data and objective function
path_SpeckleImagingCodes;
[nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50);
setupBispectrumData;
image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*phase_recur(:)),[256 256])))));
its = 50;
tol = 1e-5;
avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));

%%

%===============%
% imphase_rec.m %
%===============%

% Run gradient descent
obj_func = @(image) imphase_rec(image,A,bispec_phase,weights, pupil_mask, [], 100.0,'pos',pospec);
[image_imphase_gd,flag, his_imphase_gd, iters_imphase_gd] = gradient_descent(obj_func, image_recur(:), its, tol, 1);
image_imphase_gd = reshape(image_imphase_gd,[256 256]);

% Run NLCG
obj_func = @(image) imphase_rec(image,A,bispec_phase,weights, pupil_mask, [], 100.0,'pos',pospec);
[image_imphase_nlcg,flag, his_imphase_nlcg, iters_imphase_nlcg] = nlcg(obj_func, image_recur(:), its, tol, 1);
image_imphase_nlcg = reshape(image_imphase_nlcg,[256 256]);

% Run damped Newton
ADA = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
obj_func = @(image) imphase_rec(image,A,bispec_phase,weights, pupil_mask, ADA, 100.0,'pos',pospec);
[image_imphase_newt,flag, his_imphase_newt, iters_imphase_newt] = damped_newton(obj_func, image_recur(:), its, tol, 1);
image_imphase_newt = reshape(image_imphase_newt,[256 256]);

%%
% Plot stuff

figure; subplot(2,3,1); imagesc(reshape(obj,[256 256])); axis image; colorbar; title('imphase\_rec');
subplot(2,3,2); imagesc(reshape(image_recur,[256 256])/max(image_recur(:))); axis image; colorbar; C = colormap; title('recur');
subplot(2,3,3); imagesc(avg_data_frame); axis image; colorbar; title('avg. blurred frame');
subplot(2,3,4); imagesc(image_imphase_gd/max(image_imphase_gd(:))); axis image; colorbar; title('gd');
subplot(2,3,5); imagesc(image_imphase_nlcg/max(image_imphase_nlcg(:))); axis image; colorbar; title('nlcg');
subplot(2,3,6); imagesc(image_imphase_newt/max(image_imphase_newt(:))); axis image; colorbar; title('newton');



%%
figure;
plot(his_imphase_gd(:,1)/his_imphase_gd(1,1),'r-o'); hold on;
plot(his_imphase_nlcg(:,1)/his_imphase_nlcg(1,1),'b-*');
plot(his_imphase_newt(:,1)/his_imphase_newt(1,1),'k-d');
%legend('Grad. Descent','NLCG','Gauss-Newton');
title('E_1(x)'); xlabel('Iteration'); ylabel('Rel. Obj. Function');
