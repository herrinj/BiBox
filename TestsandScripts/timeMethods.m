% 
%   This script tests the various optimization methods for matching the
%   object phase to the bispectrum phase
%
%   Author: James Herring, jlherri@emory.edu
%   Modified: 7/24/16
%

phase_time_gd      = [];
phase_time_nlcg    = [];
phase_time_newt    = [];
phasor_time_gd     = [];
phasor_time_nlcg   = [];
phasor_time_newt   = [];
imphase_time_gd    = [];
imphase_time_nlcg  = [];
imphase_time_newt  = [];
imphasor_time_gd   = [];
imphasor_time_nlcg = [];
imphasor_time_newt = [];


phase_LSits_gd       = [];
phasor_LSits_gd      = [];
imphase_LSits_gd     = [];
imphasor_LSits_gd    = [];
phase_LSits_nlcg     = [];
phasor_LSits_nlcg    = [];
imphase_LSits_nlcg   = [];
imphasor_LSits_nlcg  = [];
phase_LSits_newt     = [];
phasor_LSits_newt    = [];
imphase_LSits_newt   = [];
imphasor_LSits_newt  = [];

for k = 1:10
    % Setup data and objective function
    [nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50);
    path_SpeckleImagingCodes;
    setupBispectrumData;
    image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*phase_recur(:)),[256 256])))));
    its = 50;
    tol = 1e-6;
    avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));

    %=============%
    % phase_rec.m %
    %=============%
    [~,~,hess_const] = phase_rec(phase_recur(:),A,bispec_phase,weights,0,2);
    obj_func = @(phase) phase_rec(phase,A,bispec_phase,weights,0,3,hess_const);

    % Run gradient descent
    [x,flag, his_phase_gd, iters_phase_gd] = gradient_descent(obj_func, phase_recur(:), its, tol, 1);
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

    %================%
    % imphasor_rec.m %
    %================%

    % Run gradient descent
    obj_func = @(image) imphasor_rec(image,A,bispec_phase, weights, [], pupil_mask, 100.0,'pos',pospec);
    [image_imphasor_gd,flag, his_imphasor_gd, iters_imphasor_gd] = gradient_descent(obj_func, image_recur(:), its, tol, 1);
    image_imphasor_gd = reshape(image_imphasor_gd,[256 256]);
    % Run NLCG

    obj_func = @(image) imphasor_rec(image,A,bispec_phase, weights, [], pupil_mask, 100.0,'pos',pospec);
    [image_imphasor_nlcg,flag, his_imphasor_nlcg, iters_imphasor_nlcg] = nlcg(obj_func, image_recur(:), its, tol, 1);
    image_imphasor_nlcg = reshape(image_imphasor_nlcg,[256 256]);

    % Run damped Newton
    ADA = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
    obj_func = @(image) imphasor_rec(image,A,bispec_phase, weights, ADA, pupil_mask, 100.0,'pos',pospec);
    [image_imphasor_newt,flag, his_imphasor_newt, iters_imphasor_newt] = damped_newton(obj_func, image_recur(:), its, tol, 1);
    image_imphasor_newt = reshape(image_imphasor_newt, [256 256]);

    phase_time_gd      = [phase_time_gd;    his_phase_gd(:,6)];
    phase_time_nlcg    = [phase_time_nlcg;  his_phase_nlcg(:,6)];
    phase_time_newt    = [phase_time_newt;  his_phase_newt(:,6)];
    phasor_time_gd     = [phasor_time_gd;   his_phasor_gd(:,6)];
    phasor_time_nlcg   = [phasor_time_nlcg; his_phasor_nlcg(:,6)];
    phasor_time_newt   = [phasor_time_newt; his_phasor_newt(:,6)];
    imphase_time_gd    = [imphase_time_gd;  his_imphase_gd(:,6)];
    imphase_time_nlcg  = [imphase_time_nlcg;his_imphase_nlcg(:,6) ];
    imphase_time_newt  = [imphase_time_newt;his_imphase_newt(:,6) ];
    imphasor_time_gd   = [imphasor_time_gd; his_imphasor_gd(:,6)];
    imphasor_time_nlcg = [imphasor_time_nlcg; his_imphasor_nlcg(:,6)];
    imphasor_time_newt = [imphasor_time_newt; his_imphasor_newt(:,6)];


    phase_LSits_gd       = [phase_LSits_gd; his_phase_gd(:,5)];
    phasor_LSits_gd      = [phasor_LSits_gd; his_phasor_gd(:,5)];
    imphase_LSits_gd     = [imphase_LSits_gd; his_imphase_gd(:,5)];
    imphasor_LSits_gd    = [imphasor_LSits_gd; his_imphasor_gd(:,5)];
    phase_LSits_nlcg     = [phase_LSits_nlcg; his_phase_nlcg(:,5)];
    phasor_LSits_nlcg    = [phasor_LSits_nlcg; his_phasor_nlcg(:,5)];
    imphase_LSits_nlcg   = [imphase_LSits_nlcg; his_imphase_nlcg(:,5)];
    imphasor_LSits_nlcg  = [imphasor_LSits_nlcg; his_imphasor_nlcg(:,5)];
    phase_LSits_newt     = [phase_LSits_newt; his_phase_newt(:,5)];
    phasor_LSits_newt    = [phasor_LSits_newt; his_phasor_newt(:,5)];
    imphase_LSits_newt   = [imphase_LSits_newt; his_imphase_newt(:,5)];
    imphasor_LSits_newt  = [imphasor_LSits_newt; his_imphasor_newt(:,5)];
    
end


%%
phase_time_gd_final      = sum(phase_time_gd)/length(phase_time_gd);
phase_time_nlcg_final    = sum(phase_time_nlcg)/length(phase_time_nlcg);
phase_time_newt_final    = sum(phase_time_newt)/length(phase_time_newt);
phasor_time_gd_final     = sum(phasor_time_gd)/length(phasor_time_gd);
phasor_time_nlcg_final   = sum(phasor_time_nlcg)/length(phasor_time_nlcg);
phasor_time_newt_final   = sum(phasor_time_newt)/length(phasor_time_newt);
imphase_time_gd_final    = sum(imphase_time_gd)/length(imphase_time_gd);
imphase_time_nlcg_final  = sum(imphase_time_nlcg)/length(imphase_time_nlcg);
imphase_time_newt_final  = sum(imphase_time_newt)/length(imphase_time_newt);
imphasor_time_gd_final   = sum(imphasor_time_gd)/length(imphasor_time_gd);
imphasor_time_nlcg_final = sum(imphasor_time_nlcg)/length(imphasor_time_nlcg);
imphasor_time_newt_final = sum(imphasor_time_newt)/length(imphasor_time_newt);

phase_LSits_gd_final       = sum(phase_LSits_gd)/length(phase_LSits_gd);
phase_LSits_nlcg_final     = sum(phase_LSits_nlcg)/length(phase_LSits_nlcg);
phase_LSits_newt_final     = sum(phase_LSits_newt)/length(phase_LSits_newt);
phasor_LSits_gd_final      = sum(phasor_LSits_gd)/length(phasor_LSits_gd);
phasor_LSits_nlcg_final    = sum(phasor_LSits_nlcg)/length(phasor_LSits_nlcg);
phasor_LSits_newt_final    = sum(phasor_LSits_newt)/length(phasor_LSits_newt);
imphase_LSits_gd_final     = sum(imphase_LSits_gd)/length(imphase_LSits_gd);
imphase_LSits_nlcg_final   = sum(imphase_LSits_nlcg)/length(imphase_LSits_nlcg);
imphase_LSits_newt_final   = sum(imphase_LSits_newt)/length(imphase_LSits_newt);
imphasor_LSits_gd_final    = sum(imphasor_LSits_gd)/length(imphasor_LSits_gd);;
imphasor_LSits_nlcg_final  = sum(imphasor_LSits_nlcg)/length(imphasor_LSits_nlcg);;
imphasor_LSits_newt_final  = sum(imphasor_LSits_newt)/length(imphasor_LSits_newt);;

fprintf('phase_gd \t: %f \t %f \n', phase_LSits_gd_final, phase_time_gd_final);
fprintf('phase_nlcg \t: %f \t %f \n',phase_LSits_nlcg_final, phase_time_nlcg_final);
fprintf('phase_newt \t: %f \t %f \n',phase_LSits_newt_final, phase_time_newt_final);

fprintf('phasor_gd \t: %f \t %f \n',phasor_LSits_gd_final, phasor_time_gd_final);
fprintf('phasor_nlcg \t: %f \t %f \n',phasor_LSits_nlcg_final, phasor_time_nlcg_final);
fprintf('phasor_newt \t: %f \t %f \n',phasor_LSits_newt_final, phasor_time_newt_final);

fprintf('imphase_gd \t: %f \t %f \n',imphase_LSits_gd_final, imphase_time_gd_final);
fprintf('imphase_nlcg \t: %f \t %f \n',imphase_LSits_nlcg_final, imphase_time_nlcg_final);
fprintf('imphase_newt \t: %f \t %f \n',imphase_LSits_newt_final, imphase_time_newt_final);

fprintf('imphasor_gd \t: %f \t %f \n',imphasor_LSits_gd_final, imphasor_time_gd_final);
fprintf('imphasor_nlcg \t: %f \t %f \n',imphasor_LSits_nlcg_final, imphasor_time_nlcg_final);
fprintf('imphasor_newt \t: %f \t %f \n',imphasor_LSits_newt_final, imphasor_time_newt_final);