%
%   This file chooses the regularization parameters for the imphase and 
%   immphasor objections using the average mean-squared error of the 
%   resulting solution. For optimization, we us (projected) Gauss-Newton.
%
clear all; close all;

% Setup data
path_SpeckleImagingCodes;
[nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50,'D_r0',30);
setupBispectrumData;
image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_recur(:)),[256 256])))));
dims = size(image_recur);
image_proj  = gdnnf_projection(image_recur, sum(image_recur(:))) + 1e-4;
avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));

% Setup Gauss-Newton parameters
upper_bound = ones(numel(image_proj),1);
lower_bound = zeros(numel(image_proj),1);
tolJ         = 1e-4;            
tolY         = 1e-4;           
tolG         = 1e1;
maxIter      = 50;
solverMaxIter= 250;              
solverTol    = 1e-1;

%%
% Run imphase and imphasor for a number of alpha parameters
alphaIts    = 11;
alpha       = logspace(5,-5,alphaIts); % logspace for alpha

imphase_GN_sols     = zeros(numel(image_recur(:)), alphaIts);
imphase_PGNR_sols   = zeros(numel(image_recur(:)), alphaIts);
imphasor_GN_sols    = zeros(numel(image_recur(:)), alphaIts);
imphasor_PGNR_sols  = zeros(numel(image_recur(:)), alphaIts);

imphase_GN_norm      = zeros(alphaIts,1);
imphasor_GN_norm     = zeros(alphaIts,1);
imphase_PGNR_norm    = zeros(alphaIts,1);
imphasor_PGNR_norm   = zeros(alphaIts,1);

% For positivity regularizer with Gauss-Newton
for k = 1:alphaIts
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alpha(k),'regularizer','pos');
    [imphase_GN, his_imphase_GN] = GaussNewtonProj(fctn, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                'iterSave',true);
    
    imphase_GN_norm(k) = norm(imphase_GN(:))^2;
    imphase_GN_sols(:,k) = imphase_GN./max(imphase_GN(:));
    clear GaussNewtonProj;
    clear imphaseObjFctn;   
    
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alpha(k),'regularizer','pos');
    [imphasor_GN, his_imphasor_GN] = GaussNewtonProj(fctn, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                'iterSave',true);
    
    imphasor_GN_norm(k) = norm(imphasor_GN(:))^2;
    imphasor_GN_sols(:,k) = imphasor_GN./max(imphasor_GN(:));
    clear GaussNewtonProj;
    clear imphasorObjFctn;
end

% For discrete gradient regularizer with projected Gauss-Newton
for k = 1:alphaIts
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alpha(k),'regularizer','grad');
    [imphase_PGNR, his_imphase_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    
    imphase_PGNR_norm(k) = norm(imphase_PGNR(:))^2;
    imphase_PGNR_sols(:,k) = imphase_PGNR./max(imphase_PGNR(:));
    clear GaussNewtonProj;
    clear imphaseObjFctn;

    fctn = @(x) imphasorObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alpha(k),'regularizer','grad');
    [imphasor_PGNR, his_imphasor_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                  'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                  'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                  'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    
    imphasor_PGNR_norm(k) = norm(imphasor_PGNR(:))^2;                                      
    imphasor_PGNR_sols(:,k) = imphasor_PGNR./max(imphasor_PGNR(:));
    clear GaussNewtonProj;
    clear imphasorObjFctn;
end

%%
% Calculate average mean-squared error for output images
obj                 = obj/max(obj(:));
imphase_GN_mse      = zeros(alphaIts,1);
imphasor_GN_mse     = zeros(alphaIts,1);
imphase_PGNR_mse    = zeros(alphaIts,1);
imphasor_PGNR_mse   = zeros(alphaIts,1);

for k = 1:alphaIts
    imphase_GN_mse(k)   = norm(obj(:) - imphase_GN_sols(:,k))^2;
    imphasor_GN_mse(k)  = norm(obj(:) - imphasor_GN_sols(:,k))^2;
    imphase_PGNR_mse(k) = norm(obj(:) - imphase_PGNR_sols(:,k))^2;
    imphasor_PGNR_mse(k)= norm(obj(:) - imphasor_PGNR_sols(:,k))^2;
end

%%
% Plot averaged mean-squared error over alpha

figure(); subplot(1,2,1);
loglog(alpha,imphase_GN_mse,'ro-');
hold on;
loglog(alpha,imphasor_GN_mse,'b*-');
leg = legend('imphase+pos','imphasor+pos');
leg.FontSize = 14;
tit = title('Mean-squared error vs. Alpha');
tit.FontSize = 16;
xlabel('\alpha'); ylabel('Mean-squared error');

subplot(1,2,2);
loglog(alpha,imphase_PGNR_mse,'kd-');
hold on;
loglog(alpha,imphasor_PGNR_mse,'mh-');
leg = legend('imphase+grad','imphasor+grad');
leg.FontSize = 14;
tit = title('Mean-squared error vs. Alpha');
tit.FontSize = 16;
xlabel('\alpha'); ylabel('Mean-squared error');

% %%
% % Calculate the residuals and logs of the solutions for L-curve
% imphase_GN_res      = zeros(alphaIts,1);
% imphasor_GN_res     = zeros(alphaIts,1);
% imphase_PGNR_res    = zeros(alphaIts,1);
% imphasor_PGNR_res   = zeros(alphaIts,1);
% 
% fctn_imphase    = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', 0,'regularizer','pos');
% fctn_imphasor   = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha', 0,'regularizer','pos');
% 
% for k = 1:alphaIts
%     imphase_GN_res(k)      = fctn_imphase(imphase_GN_sols(:,k));
%     imphasor_GN_res(k)     = fctn_imphasor(imphasor_GN_sols(:,k));
%     imphase_PGNR_res(k)    = fctn_imphase(imphase_PGNR_sols(:,k));
%     imphasor_PGNR_res(k)   = fctn_imphasor(imphase_PGNR_sols(:,k));
% end
% 
% %%
% %   Plot L-curve
% 
% figure(); subplot(1,2,1);
% loglog(imphase_GN_res, imphase_GN_norm,'ro-');
% hold on;
% loglog(imphasor_GN_res, imphasor_GN_norm,'b*-');
% leg = legend('imphase+pos','imphasor+pos');
% leg.FontSize = 14;
% tit = title('L-curve');
% tit.FontSize = 16;
% 
% subplot(1,2,2);
% loglog(imphase_PGNR_res, imphase_PGNR_norm,'kd-');
% hold on;
% loglog(imphasor_PGNR_res, imphasor_PGNR_norm,'mh-');
% leg = legend('imphase+grad','imphasor+grad');
% leg.FontSize = 14;
% tit = title('L-curve');
% tit.FontSize = 16;
% xlabel('$\| r_{\alpha} \|^2$','interpreter','latex'); ylabel('$\| x_{\alpha} \|^2$','interpreter','latex')
