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
%==============%
% phasor_rec.m %
%==============%
[~,~,hess_const] = phasor_rec(phase_recur(:),A,bispec_phase,weights,0,2);
obj_func = @(phase) phasor_rec(phase,A,bispec_phase,weights,0,3,hess_const);

% Run gradient descent
[x,flag, his_phasor_gd, iters_phasor_gd] = gradient_descent(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phasor_gd = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));

% Run NLCG
[x,flag, his_phasor_nlcg, iters_phasor_nlcg] = nlcg(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phasor_nlcg = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));
% Run damped Newton
[x,flag, his_phasor_newt, iters_phasor_newt] = damped_newton(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phasor_newt = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));

%%
% Plot stuff

figure; 
subplot(2,3,1); imagesc(reshape(obj,[256 256])); axis image; colorbar; title('phasor\_rec');
subplot(2,3,2); imagesc(reshape(image_recur,[256 256])/max(image_recur(:))); axis image; colorbar; C = colormap; title('recur');
subplot(2,3,3); imagesc(avg_data_frame); axis image; colorbar; title('avg. blurred frame');
subplot(2,3,4); imagesc(image_phasor_gd/max(image_phasor_gd(:))); axis image; colorbar; title('gd');
subplot(2,3,5); imagesc(image_phasor_nlcg/max(image_phasor_nlcg(:))); axis image; colorbar; title('nlcg');
subplot(2,3,6); imagesc(image_phasor_newt/max(image_phasor_newt(:))); axis image; colorbar; title('newton');

%%
figure;
plot(his_phasor_gd(:,1)/his_phasor_gd(1,1),'ro'); hold on;
plot(his_phasor_nlcg(:,1)/his_phasor_nlcg(1,1),'b*');
plot(his_phasor_newt(:,1)/his_phasor_newt(1,1),'kd');
%legend('Grad. Descent','NLCG','Gauss-Newton');
title('E_2(\phi)'); xlabel('Iteration'); ylabel('Rel. Obj. Function');
