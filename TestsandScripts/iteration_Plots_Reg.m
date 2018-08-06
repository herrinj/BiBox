% 
%   This script tests creates plots using the results of the optimization
%   to match the object phase to the bispectrum phase
%
%   Author: James Herring, jlherri@emory.edu
%   Modified: 7/20/16
%

%clear; close all;

load hisInfo_50_30_reg.mat
load iterInfo_50_30_reg.mat

% First, plot objective function and gradient values
figure; clf; plot(his_phase_gd(:,1)/max(his_phase_gd(:,1)),'r+'); hold on; plot(his_phase_nlcg(:,1)/max(his_phase_nlcg(:,1)),'b*'); plot(his_phase_newt(:,1)/max(his_phase_newt(:,1)),'kd');
title('E_1(\phi)'); xlabel('Iteration'); ylabel('Relative Obj. Func.');
legend('GD','NLCG','Newton');

figure; clf; plot(his_phasor_gd(:,1)/max(his_phasor_gd(:,1)),'r+'); hold on; plot(his_phasor_nlcg(:,1)/max(his_phasor_nlcg(:,1)),'b*'); plot(his_phasor_newt(:,1)/max(his_phasor_newt(:,1)),'kd');
title('E_2(\phi)'); xlabel('Iteration'); ylabel('Relative Obj. Func.');
legend('GD','NLCG','Newton');

figure; clf; plot(his_imphase_gd(:,1)/max(his_imphase_gd(:,1)),'r+'); hold on; plot(his_imphase_nlcg(:,1)/max(his_imphase_nlcg(:,1)),'b*'); plot(his_imphase_newt(:,1)/max(his_imphase_newt(:,1)),'kd');
title('E_1(f)'); xlabel('Iteration'); ylabel('Relative Obj. Func.');
legend('GD','NLCG','Newton');

figure; clf; plot(his_imphasor_gd(:,1)/max(his_imphasor_gd(:,1)),'r+'); hold on; plot(his_imphasor_nlcg(:,1)/max(his_imphasor_nlcg(:,1)),'b*'); plot(his_imphasor_newt(:,1)/max(his_imphasor_newt(:,1)),'kd');
title('E_2(f)'); xlabel('Iteration'); ylabel('Relative Obj. Func.');
legend('GD','NLCG','Newton');




% Reshape iterations
iters_phase_gd = phase_foldout(reshape(iters_phase_gd, 256, 256, []));
iters_phase_nlcg = phase_foldout(reshape(iters_phase_nlcg, 256, 256, []));
iters_phase_newt = phase_foldout(reshape(iters_phase_newt, 256, 256, []));
iters_phasor_gd = phase_foldout(reshape(iters_phasor_gd, 256, 256, []));
iters_phasor_nlcg = phase_foldout(reshape(iters_phasor_nlcg, 256, 256, []));
iters_phasor_newt = phase_foldout(reshape(iters_phasor_newt, 256, 256, []));
images_imphase_gd = reshape(iters_imphase_gd, 256, 256, []);
images_imphase_nlcg = reshape(iters_imphase_nlcg, 256, 256, []);
images_imphase_newt = reshape(iters_imphase_newt, 256, 256, []);
images_imphasor_gd = reshape(iters_imphasor_gd, 256, 256, []);
images_imphasor_nlcg = reshape(iters_imphasor_nlcg, 256, 256, []);
images_imphasor_newt = reshape(iters_imphasor_newt, 256, 256, []);

% Extract phases out of the imphas*.m iters and make images by combining
% with the calculated power spectrum
phases_imphase_gd = angle(fftshift(fft2(fftshift(images_imphase_gd))));
phases_imphase_nlcg = angle(fftshift(fft2(fftshift(images_imphase_nlcg))));
phases_imphase_newt = angle(fftshift(fft2(fftshift(images_imphase_newt))));
phases_imphasor_gd = angle(fftshift(fft2(fftshift(images_imphasor_gd))));
phases_imphasor_nlcg = angle(fftshift(fft2(fftshift(images_imphasor_nlcg))));
phases_imphasor_newt = angle(fftshift(fft2(fftshift(images_imphasor_newt))));
images_imphase_gd_ex = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*phases_imphase_gd))))));
images_imphase_nlcg_ex = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*phases_imphase_nlcg))))));
images_imphase_newt_ex = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*phases_imphase_newt))))));
images_imphasor_gd_ex = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*phases_imphasor_gd))))));
images_imphasor_nlcg_ex = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*phases_imphasor_nlcg))))));
images_imphasor_newt_ex = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*phases_imphasor_newt))))));

% Create images out of phase iters
images_phase_gd = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*iters_phase_gd))))));
images_phase_nlcg = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*iters_phase_nlcg))))));
images_phase_newt = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*iters_phase_newt))))));
images_phasor_gd = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*iters_phasor_gd))))));
images_phasor_nlcg = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*iters_phasor_nlcg))))));
images_phasor_newt = real(fftshift(ifft2(fftshift(bsxfun(@times,pospec,exp(1i*iters_phasor_newt))))));

% Calculate the correlations
% iters = 50;
% corr_phase_gd = zeros(iters,1); corr_phase_nlcg = zeros(iters,1); corr_phase_newt = zeros(iters,1);
% corr_phasor_gd = zeros(iters,1); corr_phasor_nlcg = zeros(iters,1); corr_phasor_newt = zeros(iters,1);
% corr_imphase_gd = zeros(iters,1); corr_imphase_nlcg = zeros(iters,1); corr_imphase_newt = zeros(iters,1);
% corr_imphasor_gd = zeros(iters,1); corr_imphasor_nlcg = zeros(iters,1); corr_imphasor_newt = zeros(iters,1);
% corr_imphase_gd_ex = zeros(iters,1); corr_imphase_nlcg_ex = zeros(iters,1); corr_imphase_newt_ex = zeros(iters,1);
% corr_imphasor_gd_ex = zeros(iters,1); corr_imphasor_nlcg_ex = zeros(iters,1); corr_imphasor_newt_ex = zeros(iters,1);
% for k = 1:iters
%     corr_phase_gd(k) = corr2(obj, images_phase_gd(:,:,k));
%     corr_phase_nlcg(k) = corr2(obj, images_phase_nlcg(:,:,k));
%     corr_phase_newt(k) = corr2(obj, images_phase_newt(:,:,k));
%     corr_phasor_gd(k) = corr2(obj, images_phasor_gd(:,:,k));
%     corr_phasor_nlcg(k) = corr2(obj, images_phasor_nlcg(:,:,k));
%     corr_phasor_newt(k) = corr2(obj, images_phasor_newt(:,:,k));
%     corr_imphase_gd(k) = corr2(obj, images_imphase_gd(:,:,k));
%     corr_imphase_nlcg(k) = corr2(obj, images_imphase_nlcg(:,:,k));
%     corr_imphase_newt(k) = corr2(obj, images_imphase_newt(:,:,k));
%     corr_imphasor_gd(k) = corr2(obj, images_imphasor_gd(:,:,k));
%     corr_imphasor_nlcg(k) = corr2(obj, images_imphasor_nlcg(:,:,k));
%     corr_imphasor_newt(k) = corr2(obj, images_imphasor_newt(:,:,k));
%     corr_imphase_gd_ex(k) = corr2(obj, images_imphase_gd_ex(:,:,k));
%     corr_imphase_nlcg_ex(k) = corr2(obj, images_imphase_nlcg_ex(:,:,k));
%     corr_imphase_newt_ex(k) = corr2(obj, images_imphase_newt_ex(:,:,k));
%     corr_imphasor_gd_ex(k) = corr2(obj, images_imphasor_gd_ex(:,:,k));
%     corr_imphasor_nlcg_ex(k) = corr2(obj, images_imphasor_nlcg_ex(:,:,k));
%     corr_imphasor_newt_ex(k) = corr2(obj, images_imphasor_newt_ex(:,:,k));
% end

% figure; 
% plot(corr_phase_gd,'+'); hold on; plot(corr_phase_nlcg,'+'); plot(corr_phase_newt,'+');
% plot(corr_phasor_gd,'o'); plot(corr_phasor_nlcg,'o'); plot(corr_phasor_newt,'o');
% plot(corr_imphase_gd,'h'); plot(corr_imphase_nlcg,'h'); plot(corr_imphase_newt,'h');
% plot(corr_imphasor_gd,'s'); plot(corr_imphasor_nlcg,'s'); plot(corr_imphasor_newt,'s');
% plot(corr_imphase_gd_ex,'d'); plot(corr_imphase_nlcg_ex,'d'); plot(corr_imphase_newt_ex,'d');
% plot(corr_imphasor_gd_ex,'*'); plot(corr_imphasor_nlcg_ex,'*'); plot(corr_imphasor_newt_ex,'*');
% title('Correlation');
% legend('phase gd', 'phase nlcg', 'phase newt', 'phasor gd', 'phasor nlcg', 'phasor newt',...
%     'imphase gd', 'imphase nlcg', 'imphase newt', 'imphasor gd', 'imphasor nlcg', 'imphasor newt',...
%     'imphase\_ex gd', 'imphase\_ex nlcg', 'imphase\_ex newt', 'imphasor\_ex gd', 'imphasor\_ex nlcg', 'imphasor\_ex newt',...
%     'Location','eastoutside');


