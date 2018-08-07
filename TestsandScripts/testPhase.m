% 
%   This script tests the phase_rec objective function method for matching 
%   the object phase to the bispectrum phase
%
%   Author: James Herring, jlherri@emory.edu
%   Modified: 9/19/16
%

% Setup data and objective function
path_SpeckleImagingCodes;
[nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50);
setupBispectrumData;
image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_recur(:)),[256 256])))));
its = 50;
tol = 1e-5;
avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));

%%
%=============%
% phase_rec.m %
%=============%
[~,~,hess_const] = phase_rec(phase_recur(:),A,bispec_phase,weights,0,2);
obj_func = @(phase) phase_rec(phase,A,bispec_phase,weights,0,3,hess_const);

% Run gradient descent
[x,flag, his_phase_gd, iters_phase_gd] = gradient_descent(obj_func, phase_recur(:), its, tol, 1);
%%
x = phase_foldout(reshape(x,[256 256]), 0);
image_phase_gd = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));

% Run NLCG
[x,flag, his_phase_nlcg, iters_phase_nlcg] = nlcg(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phase_nlcg = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));

% Run damped Newton
[x,flag, his_phase_newt, iters_phase_newt] = damped_newton(obj_func, phase_recur(:), its, tol, 1);
x = phase_foldout(reshape(x,[256 256]), 0);
image_phase_newt = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*x(:)),[256 256])))));


%%
% Plot stuff
str_recur   = sprintf('|| Xtrue(:) - Xrecur(:) ||_2 = %.3f' , norm(obj(:) - image_recur(:)))
str_data   = sprintf('|| Xtrue(:) - Xdata(:) ||_2 = %.3f'   , norm(obj(:) - avg_data_frame(:)))
str_gd   = sprintf('|| Xtrue(:) - Xgd(:) ||_2 = %.3f'       , norm(obj(:) - image_phase_gd(:)))
str_nlcg  = sprintf('|| Xtrue(:) - Xnlcg(:) ||_2 = %.3f'    , norm(obj(:) - image_phase_nlcg(:)))
str_newt   = sprintf('|| Xtrue(:) - Xnewt(:) ||_2 = %.3f'   , norm(obj(:) - image_phase_newt(:)))

%%
figure; subplot(2,3,1); imagesc(reshape(obj,[256 256])); axis image; C = colormap; title('truth'); axis off;
subplot(2,3,2); imagesc(reshape(image_recur,[256 256])/max(image_recur(:))); axis image; colormap(C); title('recur');
subplot(2,3,3); imagesc(avg_data_frame); axis image; colormap(C); title('avg. blurred frame');
subplot(2,3,4); imagesc(image_phase_gd/max(image_phase_gd(:))); axis image; colormap(C); title('gd');
subplot(2,3,5); imagesc(image_phase_nlcg/max(image_phase_nlcg(:))); axis image; colormap(C); title('nlcg');
subplot(2,3,6); imagesc(image_phase_newt/max(image_phase_newt(:))); axis image; colormap(C); title('newton');

%%
figure;
plot(his_phase_gd(:,1)/his_phase_gd(1,1),'ro'); hold on;
plot(his_phase_nlcg(:,1)/his_phase_nlcg(1,1),'b*');
plot(his_phase_newt(:,1)/his_phase_newt(1,1),'kd');
%legend('Grad. Descent','NLCG','Gauss-Newton');
title('E_1(\phi)'); xlabel('Iteration'); ylabel('Rel. Obj. Function');
