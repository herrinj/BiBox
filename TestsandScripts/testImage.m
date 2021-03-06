%
%   This file tests optimization using projected Gauss-Newton with the
%   imphase and imphasor objective functions. 
%
clear all; close all;

% Setup data
path_SpeckleImagingCodes;
[nfr, D_r0, image_name, K_n, sigma_rn, fourier_rad, second_rad]  = setupBispectrumParams('nfr',100,'D_r0',50,'fourier_rad',96,'sigma_rn',5);
setupBispectrumData;
image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_recur(:)),[256 256])))));
dims = size(image_recur);
image_proj  = gdnnf_projection(image_recur, sum(image_recur(:))) + 1e-4;
avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));

% Setup Gauss-Newton parameters
upper_bound = inf*ones(numel(image_proj),1);
lower_bound = zeros(numel(image_proj),1);
tolJ         = 1e-4;            
tolY         = 1e-4;           
tolG         = 1e1;
tolN         = 1e-3;
maxIter      = 100;
solverMaxIter= 250;              
solverTol    = 1e-1;
alphaPos     = 1e3;
alphaGrad    = 1e-2;
alphaTV      = 1e4;

%%
% Run gradient descent for imphase
fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
tic();
[imphase_GD, his_imphase_GD] = GradientDescentProj(fctn, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'iterSave',true);
time_imphase_GD = toc();
imphase_GD = reshape(imphase_GD,[256 256]);
clear GradientDescentProj;
clear imphaseObjFctn;

% Run gradient descent for imphasor
fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
tic();
[imphasor_GD, his_imphasor_GD] = GradientDescentProj(fctn, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'iterSave',true);                        
time_imphasor_GD = toc();
imphasor_GD = reshape(imphasor_GD, [256 256]);
clear GradientDescentProj;
clear imphasorObjFctn;

%%
% Run projected gradient descent for imphase
fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaTV,'regularizer','tv','weights',weights);
tic();
[imphase_PGD, his_imphase_PGD] = GradientDescentProj(fctn, image_proj(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphase_PGD = toc();
imphase_PGD = reshape(imphase_PGD,[256 256]);
clear GradientDescentProj;
clear imphaseObjFctn;

% Run gradient descent for imphasor
fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaTV,'regularizer','tv','weights',weights);
tic();
[imphasor_PGD, his_imphasor_PGD] = GradientDescentProj(fctn, image_proj(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);                        
time_imphasor_PGD = toc();
imphasor_PGD = reshape(imphasor_PGD, [256 256]);
clear GradientDescentProj;
clear imphasorObjFctn;


%%
% Run NLCG for imphase
% fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
% tic();
% [imphase_NLCG, his_imphase_NLCG] = NonlinearCG(fctn, image_recur(:), 'maxIter',maxIter,...
%                                         'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
%                                         'iterSave',true);
% time_imphase_NLCG = toc();
% imphase_NLCG = reshape(imphase_NLCG,[256 256]);
% clear NonlinearCG;
% clear imphaseObjFctn;
% 
% % Run NLCG for imphasor
% fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
% tic();
% [imphasor_NLCG, his_imphasor_NLCG] = NonlinearCG(fctn, image_recur(:),'maxIter',maxIter,...
%                                           'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
%                                           'iterSave',true);
% time_imphasor_NLCG = toc();
% imphasor_NLCG = reshape(imphasor_NLCG, [256 256]);
% clear NonlinearCG;
% clear imphasorObjFctn;

%%
% Run LBFGS for imphase
fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
tic();
[imphase_BFGS, his_imphase_BFGS] = LBFGS(fctn, image_recur(:), 'maxIter',maxIter,...
                                        'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                        'iterSave',true);
time_imphase_BFGS = toc();
imphase_BFGS = reshape(imphase_BFGS,[256 256]);
clear LBFGS;
clear imphaseObjFctn;

% Run LBFGS for imphasor
fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
tic();
[imphasor_BFGS, his_imphasor_BFGS] = LBFGS(fctn, image_recur(:),'maxIter',maxIter,...
                                          'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                          'iterSave',true);
time_imphasor_BFGS = toc();
imphasor_BFGS = reshape(imphasor_BFGS, [256 256]);
clear LBFGS;
clear imphasorObjFctn;

%%
% Run Gauss-Newton for imphase
fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
tic();
[imphase_GN, his_imphase_GN] = GaussNewtonProj(fctn, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'tolN',tolN,'solver','bispIm','solverMaxIter',250,...
                                                'solverTol',1e-1,'iterSave',true);
time_imphase_GN = toc();
imphase_GN = reshape(imphase_GN,[256 256]);
clear GaussNewtonProj;
clear imphaseObjFctn;

% Run Gauss-Newton for imphasor
fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
tic();
[imphasor_GN, his_imphasor_GN] = GaussNewtonProj(fctn, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'tolN',tolN,'solver','bispIm','solverMaxIter',250,...
                                                'solverTol',1e-1,'iterSave',true);
time_imphasor_GN = toc();
imphasor_GN = reshape(imphasor_GN, [256 256]);
clear GaussNewtonProj;
clear imphasorObjFctn;

%%
% Run projected Gauss-Newton for imphase
fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaTV,'regularizer','tv','weights',weights);
tic();
[imphase_PGN, his_imphase_PGN] = GaussNewtonProj(fctn, image_proj(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'tolN',tolN,'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphase_PGN = toc();
imphase_PGN = reshape(imphase_PGN,[256 256]);
clear GaussNewtonProj;
clear imphaseObjFctn;

% Run projected Gauss-Newton for imphasor
fctn = @(x) imphasorObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaTV,'regularizer','tv','weights',weights);
tic();
[imphasor_PGN, his_imphasor_PGN] = GaussNewtonProj(fctn, image_proj(:),...
                                                  'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                  'tolN',tolN,'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                  'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphasor_PGN = toc();
imphasor_PGN = reshape(imphasor_PGN,[256 256]);
clear GaussNewtonProj;
clear imphasorObjFctn;

%%
% Run projected Gauss-Newton for imphase
fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaGrad,'regularizer','grad','weights',weights);
tic();
[imphase_PGNR, his_imphase_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'tolN',tolN,'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphase_PGNR = toc();
imphase_PGNR = reshape(imphase_PGNR,[256 256]);
clear GaussNewtonProj;
clear imphaseObjFctn;

% Run projected Gauss-Newton for imphasor
fctn = @(x) imphasorObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaGrad,'regularizer','grad','weights',weights);
tic();
[imphasor_PGNR, his_imphasor_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                  'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                  'tolN',tolN,'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                  'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphasor_PGNR = toc();
imphasor_PGNR = reshape(imphasor_PGNR,[256 256]);
clear GaussNewtonProj;
clear imphasorObjFctn;

%%
% Look at some results
obj             = obj/max(obj(:));
image_recur     = image_recur/max(image_recur(:));
image_proj      = image_proj/max(image_proj(:));
avg_data_frame  = avg_data_frame/max(avg_data_frame(:));
imphase_GD      = imphase_GD/max(imphase_GD(:));
imphase_PGD      = imphase_PGD/max(imphase_PGD(:));
% imphase_NLCG    = imphase_NLCG/max(imphase_NLCG(:));
imphase_BFGS    = imphase_BFGS/max(imphase_BFGS(:));
imphase_GN      = imphase_GN/max(imphase_GN(:));
imphase_PGN     = imphase_PGN/max(imphase_PGN(:));
imphase_PGNR    = imphase_PGNR/max(imphase_PGNR(:));
imphasor_GD     = imphasor_GD/max(imphasor_GD(:));
imphasor_PGD    = imphasor_PGD/max(imphasor_PGD(:));
% imphasor_NLCG   = imphasor_NLCG/max(imphasor_NLCG(:));
imphasor_BFGS   = imphasor_BFGS/max(imphasor_BFGS(:));
imphasor_GN     = imphasor_GN/max(imphasor_GN(:));
imphasor_PGN    = imphasor_PGN/max(imphasor_PGN(:));
imphasor_PGNR   = imphasor_PGNR/max(imphasor_PGNR(:));

figure(1); 
subplot(3,6,2);  imagesc(reshape(obj,[256 256])); axis image; axis off; colorbar; title('truth'); 
subplot(3,6,3);  imagesc(reshape(image_recur,[256 256])); axis image; axis off; colorbar; title('recur');
subplot(3,6,4);  imagesc(reshape(image_proj, [256 256])); axis image; axis off; colorbar;  title('proj. recur');
subplot(3,6,5);  imagesc(avg_data_frame); axis image; axis off; colorbar;  title('avg. blurred frame');

subplot(3,6,7);  imagesc(imphase_GD); axis image; axis off; colorbar;  title('Imphase - GD');
subplot(3,6,8);  imagesc(imphase_PGD); axis image; axis off; colorbar;  title('Imphase - PGD');
% subplot(3,6,8);  imagesc(imphase_NLCG); axis image; axis off; colorbar;  title('Imphase - NLCG');
subplot(3,6,9);  imagesc(imphase_BFGS); axis image; axis off; colorbar;  title('Imphase - LBFGS');
subplot(3,6,10); imagesc(imphase_GN); axis image; axis off; colorbar;  title('Imphase - GN');
subplot(3,6,11); imagesc(imphase_PGN); axis image; axis off; colorbar;  title('Imphase - PGN');
subplot(3,6,12); imagesc(imphase_PGNR); axis image; axis off; colorbar;  title('Imphase - PGNR');

subplot(3,6,13); imagesc(imphasor_GD); axis image; axis off; colorbar;  title('Imphasor - GD');
subplot(3,6,14); imagesc(imphasor_PGD); axis image; axis off; colorbar;  title('Imphasor - PGD');
% subplot(3,6,14); imagesc(imphasor_NLCG); axis image; axis off; colorbar;  title('Imphasor - NLCG');
subplot(3,6,15); imagesc(imphasor_BFGS); axis image; axis off; colorbar;  title('Imphasor - LBFGS');
subplot(3,6,16); imagesc(imphasor_GN); axis image; axis off; colorbar;  title('Imphasor - GN');
subplot(3,6,17); imagesc(imphasor_PGN); axis image; axis off; colorbar;  title('Imphasor - PGN');
subplot(3,6,18); imagesc(imphasor_PGNR); axis image; axis off; colorbar;  title('Imphasor - PGNR');

%%
% Relative objective function

figure();
plot((0:size(his_imphase_GD.array,1)-1)',his_imphase_GD.array(:,2)/his_imphase_GD.array(1,2)); 
hold on;
% plot((0:size(his_imphasor_GD.array,1)-1)',his_imphasor_GD.array(:,2)/his_imphasor_GD.array(1,2)); 
plot((0:size(his_imphase_PGD.array,1)-1)',his_imphase_PGD.array(:,2)/his_imphase_PGD.array(1,2)); 
% plot((0:size(his_imphasor_PGD.array,1)-1)',his_imphasor_PGD.array(:,2)/his_imphasor_PGD.array(1,2));
% plot((0:size(his_imphase_NLCG.array,1)-1)',his_imphase_NLCG.array(:,2)/his_imphase_NLCG.array(1,2)); 
% plot((0:size(his_imphasor_NLCG.array,1)-1)',his_imphasor_NLCG.array(:,2)/his_imphasor_NLCG.array(1,2));
plot((0:size(his_imphase_BFGS.array,1)-1)',his_imphase_BFGS.array(:,2)/his_imphase_BFGS.array(1,2)); 
% plot((0:size(his_imphasor_BFGS.array,1)-1)',his_imphasor_BFGS.array(:,2)/his_imphasor_BFGS.array(1,2));
plot((0:size(his_imphase_GN.array,1)-1)',his_imphase_GN.array(:,2)/his_imphase_GN.array(1,2)); 
% plot((0:size(his_imphasor_GN.array,1)-1)',his_imphasor_GN.array(:,2)/his_imphasor_GN.array(1,2)); 
plot((0:size(his_imphase_PGN.array,1)-1)',his_imphase_PGN.array(:,2)/his_imphase_PGN.array(1,2)); 
% plot((0:size(his_imphasor_PGN.array,1)-1)',his_imphasor_PGN.array(:,2)/his_imphasor_PGN.array(1,2));
plot((0:size(his_imphase_PGNR.array,1)-1)',his_imphase_PGNR.array(:,2)/his_imphase_PGNR.array(1,2)); 
% plot((0:size(his_imphasor_PGNR.array,1)-1)',his_imphasor_PGNR.array(:,2)/his_imphasor_PGNR.array(1,2));
% leg = legend('E1-GD-pen.reg','E2-GD-pen.reg','E1-PGD-R','E2-PGD-R',...
%              'E1-LBFGS-pen.reg','E2-LBFGS-pen.reg','E1-GN-pen.reg', 'E2-GN-pen.reg.',...
%              'E1-PGN','E2-PGN','E1-PGN-R','E2-PGN-R');
leg = legend('E1-GD-pen.reg','E1-PGD-R',...
             'E1-LBFGS-pen.reg','E1-GN-pen.reg',...
             'E1-PGN','E1-PGN-R');
leg.FontSize = 14;
tit = title('Rel. Obj. Func: $\frac{\|J\|}{\|J(0)\|}$','interpreter','latex');
tit.FontSize = 16;

%%
% Correct shifts

its_imphase_GD      = reshape(his_imphase_GD.iters, 256, 256, []);
its_imphasor_GD     = reshape(his_imphasor_GD.iters, 256, 256, []);
its_imphase_PGD     = reshape(his_imphase_PGD.iters, 256, 256, []);
its_imphasor_PGD    = reshape(his_imphasor_PGD.iters, 256, 256, []);
% its_imphase_NLCG    = reshape(his_imphase_NLCG.iters, 256, 256, []);
% its_imphasor_NLCG   = reshape(his_imphasor_NLCG.iters, 256, 256, []);
its_imphase_BFGS    = reshape(his_imphase_BFGS.iters, 256, 256, []);
its_imphasor_BFGS   = reshape(his_imphasor_BFGS.iters, 256, 256, []);
its_imphase_GN      = reshape(his_imphase_GN.iters, 256, 256, []);
its_imphasor_GN     = reshape(his_imphasor_GN.iters, 256, 256, []);
its_imphase_PGN     = reshape(his_imphase_PGN.iters, 256, 256, []);
its_imphasor_PGN    = reshape(his_imphasor_PGN.iters, 256, 256, []);
its_imphase_PGNR    = reshape(his_imphase_PGNR.iters, 256, 256, []);
its_imphasor_PGNR   = reshape(his_imphasor_PGNR.iters, 256, 256, []);

for j = 1:size(its_imphase_GD,3)
    s = measureShift(obj,its_imphase_GD(:,:,j));
    its_imphase_GD(:,:,j) = shiftImage(its_imphase_GD(:,:,j),s);
end

for j = 1:size(its_imphasor_GD,3)
    s = measureShift(obj,its_imphasor_GD(:,:,j));
    its_imphasor_GD(:,:,j) = shiftImage(its_imphasor_GD(:,:,j),s);
end

for j = 1:size(its_imphase_PGD,3)
    s = measureShift(obj,its_imphase_PGD(:,:,j));
    its_imphase_PGD(:,:,j) = shiftImage(its_imphase_PGD(:,:,j),s);
end

for j = 1:size(its_imphasor_PGD,3)
    s = measureShift(obj,its_imphasor_PGD(:,:,j));
    its_imphasor_PGD(:,:,j) = shiftImage(its_imphasor_PGD(:,:,j),s);
end

% for j = 1:size(its_imphase_NLCG,3)
%     s = measureShift(obj,its_imphase_NLCG(:,:,j));
%     its_imphase_NLCG(:,:,j) = shiftImage(its_imphase_NLCG(:,:,j),s);
% end
% 
% for j = 1:size(its_imphasor_NLCG,3)
%     s = measureShift(obj,its_imphasor_NLCG(:,:,j));
%     its_imphasor_NLCG(:,:,j) = shiftImage(its_imphasor_NLCG(:,:,j),s);
% end

for j = 1:size(its_imphase_BFGS,3)
    s = measureShift(obj,its_imphase_BFGS(:,:,j));
    its_imphase_BFGS(:,:,j) = shiftImage(its_imphase_BFGS(:,:,j),s);
end

for j = 1:size(its_imphasor_BFGS,3)
    s = measureShift(obj,its_imphasor_BFGS(:,:,j));
    its_imphasor_BFGS(:,:,j) = shiftImage(its_imphasor_BFGS(:,:,j),s);
end

for j = 1:size(its_imphase_GN,3)
    s = measureShift(obj,its_imphase_GN(:,:,j));
    its_imphase_GN(:,:,j) = shiftImage(its_imphase_GN(:,:,j),s);
end

for j = 1:size(its_imphasor_GN,3)
    s = measureShift(obj,its_imphasor_GN(:,:,j));
    its_imphasor_GN(:,:,j) = shiftImage(its_imphasor_GN(:,:,j),s);
end

for j = 1:size(its_imphase_PGN,3)
    s = measureShift(obj,its_imphase_PGN(:,:,j));
    its_imphase_PGN(:,:,j) = shiftImage(its_imphase_PGN(:,:,j),s);
end

for j = 1:size(its_imphasor_PGN,3)
    s = measureShift(obj,its_imphasor_PGN(:,:,j));
    its_imphasor_PGN(:,:,j) = shiftImage(its_imphasor_PGN(:,:,j),s);
end

for j = 1:size(its_imphase_PGNR,3)
    s = measureShift(obj,its_imphase_PGNR(:,:,j));
    its_imphase_PGNR(:,:,j) = shiftImage(its_imphase_PGNR(:,:,j),s);
end

for j = 1:size(its_imphasor_PGNR,3)
    s = measureShift(obj,its_imphasor_PGNR(:,:,j));
    its_imphasor_PGNR(:,:,j) = shiftImage(its_imphasor_PGNR(:,:,j),s);
end

its_imphase_GD      = reshape(its_imphase_GD,[], size(its_imphase_GD,3));
its_imphasor_GD     = reshape(its_imphasor_GD,[], size(its_imphasor_GD,3));
its_imphase_PGD     = reshape(its_imphase_PGD,[], size(its_imphase_PGD,3));
its_imphasor_PGD    = reshape(its_imphasor_PGD,[], size(its_imphasor_PGD,3));
% its_imphase_NLCG    = reshape(its_imphase_NLCG,[], size(its_imphase_NLCG,3));
% its_imphasor_NLCG   = reshape(its_imphasor_NLCG,[], size(its_imphasor_NLCG,3));
its_imphase_BFGS    = reshape(its_imphase_BFGS,[], size(its_imphase_BFGS,3));
its_imphasor_BFGS   = reshape(its_imphasor_BFGS,[], size(its_imphasor_BFGS,3));
its_imphase_GN      = reshape(its_imphase_GN,[], size(its_imphase_GN,3));
its_imphasor_GN     = reshape(its_imphasor_GN,[], size(its_imphasor_GN,3));
its_imphase_PGN     = reshape(its_imphase_PGN,[], size(its_imphase_PGN,3));
its_imphasor_PGN    = reshape(its_imphasor_PGN,[], size(its_imphasor_PGN,3));
its_imphase_PGNR    = reshape(its_imphase_PGNR,[], size(its_imphase_PGNR,3));
its_imphasor_PGNR   = reshape(its_imphasor_PGNR,[], size(its_imphasor_PGNR,3));
    
%%
% Relative error plots

RE_imphase_GD       = zeros(size(its_imphase_GD,2),1);
RE_imphasor_GD      = zeros(size(its_imphasor_GD,2),1);
RE_imphase_PGD      = zeros(size(its_imphase_PGD,2),1);
RE_imphasor_PGD     = zeros(size(its_imphasor_PGD,2),1);
% RE_imphase_NLCG     = zeros(size(its_imphase_NLCG,2),1);
% RE_imphasor_NLCG    = zeros(size(its_imphasor_NLCG,2),1);
RE_imphase_BFGS     = zeros(size(its_imphase_BFGS,2),1);
RE_imphasor_BFGS    = zeros(size(its_imphasor_BFGS,2),1);
RE_imphase_GN       = zeros(size(its_imphase_GN,2),1);
RE_imphasor_GN      = zeros(size(its_imphasor_GN,2),1);
RE_imphase_PGN      = zeros(size(its_imphase_PGN,2),1);
RE_imphasor_PGN     = zeros(size(its_imphasor_PGN,2),1);
RE_imphase_PGNR     = zeros(size(its_imphase_PGNR,2),1);
RE_imphasor_PGNR    = zeros(size(its_imphasor_PGNR,2),1);

for j = 1:length(RE_imphase_GD)
   RE_imphase_GD(j) = norm((its_imphase_GD(:,j)/max(its_imphase_GD(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphasor_GD)
   RE_imphasor_GD(j) = norm((its_imphasor_GD(:,j)/max(its_imphasor_GD(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphase_PGD)
   RE_imphase_PGD(j) = norm((its_imphase_PGD(:,j)/max(its_imphase_PGD(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphasor_PGD)
   RE_imphasor_PGD(j) = norm((its_imphasor_PGD(:,j)/max(its_imphasor_PGD(:,j))) - obj(:))/norm(obj(:));  
end

% for j = 1:length(RE_imphase_NLCG)
%    RE_imphase_NLCG(j) = norm((its_imphase_NLCG(:,j)/max(its_imphase_NLCG(:,j))) - obj(:))/norm(obj(:));  
% end
% 
% for j = 1:length(RE_imphasor_NLCG)
%    RE_imphasor_NLCG(j) = norm((its_imphasor_NLCG(:,j)/max(its_imphasor_NLCG(:,j))) - obj(:))/norm(obj(:));  
% end

for j = 1:length(RE_imphase_BFGS)
   RE_imphase_BFGS(j) = norm((its_imphase_BFGS(:,j)/max(its_imphase_BFGS(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphasor_BFGS)
   RE_imphasor_BFGS(j) = norm((its_imphasor_BFGS(:,j)/max(its_imphasor_BFGS(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphase_GN)
   RE_imphase_GN(j) = norm((its_imphase_GN(:,j)/max(its_imphase_GN(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphasor_GN)
   RE_imphasor_GN(j) = norm((its_imphasor_GN(:,j)/max(its_imphasor_GN(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphase_PGN)
   RE_imphase_PGN(j) = norm((its_imphase_PGN(:,j)/max(its_imphase_PGN(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphasor_PGN)
   RE_imphasor_PGN(j) = norm((its_imphasor_PGN(:,j)/max(its_imphasor_PGN(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphase_PGNR)
   RE_imphase_PGNR(j) = norm((its_imphase_PGNR(:,j)/max(its_imphase_PGNR(:,j))) - obj(:))/norm(obj(:));  
end

for j = 1:length(RE_imphasor_PGNR)
   RE_imphasor_PGNR(j) = norm((its_imphasor_PGNR(:,j)/max(its_imphasor_PGNR(:,j))) - obj(:))/norm(obj(:));  
end
figure();
plot((0:length(RE_imphase_GD)-1)',RE_imphase_GD); 
hold on;
plot((0:length(RE_imphasor_GD)-1)' ,RE_imphasor_GD); 
plot((0:length(RE_imphase_PGD)-1)',RE_imphase_PGD); 
plot((0:length(RE_imphasor_PGD)-1)' ,RE_imphasor_PGD);
% plot((0:length(RE_imphase_NLCG)-1)',RE_imphase_NLCG); 
% plot((0:length(RE_imphasor_NLCG)-1)' ,RE_imphasor_NLCG); 
plot((0:length(RE_imphase_BFGS)-1)',RE_imphase_BFGS); 
plot((0:length(RE_imphasor_BFGS)-1)' ,RE_imphasor_BFGS); 
plot((0:length(RE_imphase_GN)-1)',RE_imphase_GN); 
plot((0:length(RE_imphasor_GN)-1)' ,RE_imphasor_GN); 
plot((0:length(RE_imphase_PGN)-1)' ,RE_imphase_PGN); 
plot((0:length(RE_imphasor_PGN)-1)',RE_imphasor_PGN); 
plot((0:length(RE_imphase_PGNR)-1)' ,RE_imphase_PGNR); 
plot((0:length(RE_imphasor_PGNR)-1)',RE_imphasor_PGNR);
leg = legend('E1-GD-pen.reg','E2-GD-pen.reg','E1-PGD-R','E2-PGD_R',...
             'E1-LBFGS-pen.reg','E2-LBFGS-pen.reg','E1-GN-pen.reg', 'E2-GN-pen.reg.',...
             'E1-PGN','E2-PGN','E1-PGN-R','E2-PGN-R');
leg.FontSize = 14;
tit = title('RE: $\frac{\|x-x_{true}\|^2}{\|x_{true}\|^2}$','interpreter','latex');
tit.FontSize = 16;

%%
% Normalized cross-correlation
its_imphase_GD      = his_imphase_GD.iters;
its_imphasor_GD     = his_imphasor_GD.iters;
its_imphase_PGD     = his_imphase_PGD.iters;
its_imphasor_PGD    = his_imphasor_PGD.iters;
% its_imphase_NLCG    = his_imphase_NLCG.iters;
% its_imphasor_NLCG   = his_imphasor_NLCG.iters;
its_imphase_BFGS    = his_imphase_BFGS.iters;
its_imphasor_BFGS   = his_imphasor_BFGS.iters;
its_imphase_GN      = his_imphase_GN.iters;
its_imphasor_GN     = his_imphasor_GN.iters;
its_imphase_PGN     = his_imphase_PGN.iters;
its_imphasor_PGN    = his_imphasor_PGN.iters;
its_imphase_PGNR    = his_imphase_PGNR.iters;
its_imphasor_PGNR   = his_imphasor_PGNR.iters;

NCC_imphase_GD       = zeros(size(its_imphase_GD,2),1);
NCC_imphasor_GD      = zeros(size(its_imphasor_GD,2),1);
NCC_imphase_PGD      = zeros(size(its_imphase_PGD,2),1);
NCC_imphasor_PGD     = zeros(size(its_imphasor_PGD,2),1);
% NCC_imphase_NLCG     = zeros(size(its_imphase_NLCG,2),1);
% NCC_imphasor_NLCG    = zeros(size(its_imphasor_NLCG,2),1);
NCC_imphase_BFGS     = zeros(size(its_imphase_BFGS,2),1);
NCC_imphasor_BFGS    = zeros(size(its_imphasor_BFGS,2),1);
NCC_imphase_GN       = zeros(size(its_imphase_GN,2),1);
NCC_imphasor_GN      = zeros(size(its_imphasor_GN,2),1);
NCC_imphase_PGN      = zeros(size(its_imphase_PGN,2),1);
NCC_imphasor_PGN     = zeros(size(its_imphasor_PGN,2),1);
NCC_imphase_PGNR     = zeros(size(its_imphase_PGNR,2),1);
NCC_imphasor_PGNR    = zeros(size(its_imphasor_PGNR,2),1);

for j = 1:length(NCC_imphase_GD)
   im_act = its_imphase_GD(:,j)/max(its_imphase_GD(:,j));
   NCC_imphase_GD(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphasor_GD)
   im_act = its_imphasor_GD(:,j)/max(its_imphasor_GD(:,j));
   NCC_imphasor_GD(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphase_PGD)
   im_act = its_imphase_PGD(:,j)/max(its_imphase_PGD(:,j));
   NCC_imphase_PGD(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphasor_PGD)
   im_act = its_imphasor_PGD(:,j)/max(its_imphasor_PGD(:,j));
   NCC_imphasor_PGD(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

% for j = 1:length(NCC_imphase_NLCG)
%    im_act = its_imphase_NLCG(:,j)/max(its_imphase_NLCG(:,j));
%    NCC_imphase_NLCG(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
% end
% 
% for j = 1:length(NCC_imphasor_NLCG)
%    im_act = its_imphasor_NLCG(:,j)/max(its_imphasor_NLCG(:,j));
%    NCC_imphasor_NLCG(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
% end

for j = 1:length(NCC_imphase_BFGS)
   im_act = its_imphase_BFGS(:,j)/max(its_imphase_BFGS(:,j));
   NCC_imphase_BFGS(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphasor_BFGS)
   im_act = its_imphasor_BFGS(:,j)/max(its_imphasor_BFGS(:,j));
   NCC_imphasor_BFGS(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphase_GN)
   im_act = its_imphase_GN(:,j)/max(its_imphase_GN(:,j));
   NCC_imphase_GN(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphasor_GN)
   im_act = its_imphasor_GN(:,j)/max(its_imphasor_GN(:,j));
   NCC_imphasor_GN(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphase_PGN)
    im_act = its_imphase_PGN(:,j)/max(its_imphase_PGN(:,j));
    NCC_imphase_PGN(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphasor_PGN)
   im_act = its_imphasor_PGN(:,j)/max(its_imphasor_PGN(:,j));
   NCC_imphasor_PGN(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphase_PGNR)
    im_act = its_imphase_PGNR(:,j)/max(its_imphase_PGNR(:,j));
    NCC_imphase_PGNR(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_imphasor_PGNR)
   im_act = its_imphasor_PGNR(:,j)/max(its_imphasor_PGNR(:,j));
   NCC_imphasor_PGNR(j) = 0.5 - 0.5*((im_act(:)'*obj(:))^2/((im_act(:)'*im_act(:))*(obj(:)'*obj(:))));  
end

figure();
plot((0:length(NCC_imphase_GD)-1)',NCC_imphase_GD); 
hold on;
plot((0:length(NCC_imphasor_GD)-1)' ,NCC_imphasor_GD); 
plot((0:length(NCC_imphase_PGD)-1)',NCC_imphase_PGD); 
plot((0:length(NCC_imphasor_PGD)-1)' ,NCC_imphasor_PGD);
% plot((0:length(NCC_imphase_NLCG)-1)',NCC_imphase_NLCG); 
% plot((0:length(NCC_imphasor_NLCG)-1)' ,NCC_imphasor_NLCG);
plot((0:length(NCC_imphase_BFGS)-1)',NCC_imphase_BFGS); 
plot((0:length(NCC_imphasor_BFGS)-1)' ,NCC_imphasor_BFGS);
plot((0:length(NCC_imphase_GN)-1)',NCC_imphase_GN); 
plot((0:length(NCC_imphasor_GN)-1)' ,NCC_imphasor_GN); 
plot((0:length(NCC_imphase_PGN)-1)' ,NCC_imphase_PGN); 
plot((0:length(NCC_imphasor_PGN)-1)',NCC_imphasor_PGN); 
plot((0:length(NCC_imphase_PGNR)-1)' ,NCC_imphase_PGNR); 
plot((0:length(NCC_imphasor_PGNR)-1)',NCC_imphasor_PGNR);
leg = legend('E1-GD-pen.reg','E2-GD-pen.reg','E1-PGD-R','E2-PGD-R',...
             'E1-LBFGS-pen.reg','E2-LBFGS-pen.reg','E1-GN-pen.reg', 'E2-GN-pen.reg.',...
             'E1-PGN','E2-PGN','E1-PGN-R','E2-PGN-R');
leg.FontSize = 14;
tit = title('NCC:$\frac{1}{2} \left(1 - \frac{\langle x, x_{true}\rangle^2}{\|x\|^2 \|x_{true}\|^2} \right)$','interpreter','latex');
tit.FontSize = 16;
%%
% Print some results to the terminal window
fprintf('\n***** Relative Error Minima *****\n');
fprintf('min(RE_imphase_GD)     = %1.4e \n', min(RE_imphase_GD));
fprintf('min(RE_imphasor_GD)    = %1.4e \n', min(RE_imphasor_GD));
fprintf('min(RE_imphase_PGD)    = %1.4e \n', min(RE_imphase_PGD));
fprintf('min(RE_imphasor_PGD)   = %1.4e \n', min(RE_imphasor_PGD));
% fprintf('min(RE_imphase_NLCG)   = %1.4e \n', min(RE_imphase_NLCG));
% fprintf('min(RE_imphasor_NLCG)  = %1.4e \n', min(RE_imphasor_NLCG));
fprintf('min(RE_imphase_LBFGS)  = %1.4e \n', min(RE_imphase_BFGS));
fprintf('min(RE_imphasor_LBFGS) = %1.4e \n', min(RE_imphasor_BFGS));
fprintf('min(RE_imphase_GN)     = %1.4e \n', min(RE_imphase_GN));
fprintf('min(RE_imphasor_GN)    = %1.4e \n', min(RE_imphasor_GN));
fprintf('min(RE_imphase_PGN)    = %1.4e \n', min(RE_imphase_PGN));
fprintf('min(RE_imphasor_PGN)   = %1.4e \n', min(RE_imphasor_PGN));
fprintf('min(RE_imphase_PGNR)   = %1.4e \n', min(RE_imphase_PGNR));
fprintf('min(RE_imphasor_PGNR)  = %1.4e \n', min(RE_imphasor_PGNR));

fprintf('\n***** Normalized Cross-Correlation Minima *****\n');
fprintf('min(NCC_imphase_GD)    = %1.4e \n', min(NCC_imphase_GD));
fprintf('min(NCC_imphasor_GD)   = %1.4e \n', min(NCC_imphasor_GD));
fprintf('min(NCC_imphase_PGD)   = %1.4e \n', min(NCC_imphase_PGD));
fprintf('min(NCC_imphasor_PGD)  = %1.4e \n', min(NCC_imphasor_PGD));
% fprintf('min(NCC_imphase_NLCG)  = %1.4e \n', min(NCC_imphase_NLCG));
% fprintf('min(NCC_imphasor_NLCG) = %1.4e \n', min(NCC_imphasor_NLCG));
fprintf('min(NCC_imphase_LBFGS) = %1.4e \n', min(NCC_imphase_BFGS));
fprintf('min(NCC_imphasor_LBFGS)= %1.4e \n', min(NCC_imphasor_BFGS));
fprintf('min(NCC_imphase_GN)    = %1.4e \n', min(NCC_imphase_GN));
fprintf('min(NCC_imphasor_GN)   = %1.4e \n', min(NCC_imphasor_GN));
fprintf('min(NCC_imphase_PGN)   = %1.4e \n', min(NCC_imphase_PGN));
fprintf('min(NCC_imphasor_PGN)  = %1.4e \n', min(NCC_imphasor_PGN));
fprintf('min(NCC_imphase_PGNR)  = %1.4e \n', min(NCC_imphase_PGNR));
fprintf('min(NCC_imphasor_PGNR) = %1.4e \n', min(NCC_imphasor_PGNR));

fprintf('\n***** Total Time Elapsed *****\n');
fprintf('time(imphase_GD)       = %1.4e \n', time_imphase_GD);
fprintf('time(imphasor_GD)      = %1.4e \n', time_imphasor_GD);
fprintf('time(imphase_PGD)      = %1.4e \n', time_imphase_PGD);
fprintf('time(imphasor_PGD)     = %1.4e \n', time_imphasor_PGD);
% fprintf('time(imphase_NLCG)     = %1.4e \n', time_imphase_NLCG);
% fprintf('time(imphasor_NLCG)    = %1.4e \n', time_imphasor_NLCG);
fprintf('time(imphase_LBFGS)    = %1.4e \n', time_imphase_BFGS);
fprintf('time(imphasor_LBFGS)   = %1.4e \n', time_imphasor_BFGS);
fprintf('time(imphase_GN)       = %1.4e \n', time_imphase_GN);
fprintf('time(imphasor_GN)      = %1.4e \n', time_imphasor_GN);
fprintf('time(imphase_PGN)      = %1.4e \n', time_imphase_PGN);
fprintf('time(imphasor_PGN)     = %1.4e \n', time_imphasor_PGN);
fprintf('time(imphase_PGNR)     = %1.4e \n', time_imphase_PGNR);
fprintf('time(imphasor_PGNR)    = %1.4e \n', time_imphasor_PGNR);

fprintf('\n***** Time per Iteration *****\n');
fprintf('time(imphase_GD)/its    = %1.4e \n', time_imphase_GD/size(its_imphase_GD,2));
fprintf('time(imphasor_GD)/its   = %1.4e \n', time_imphasor_GD/size(its_imphasor_GD,2));
fprintf('time(imphase_PGD)/its   = %1.4e \n', time_imphase_PGD/size(its_imphase_PGD,2));
fprintf('time(imphasor_PGD)/its  = %1.4e \n', time_imphasor_PGD/size(its_imphasor_PGD,2));
% fprintf('time(imphase_NLCG)/its  = %1.4e \n', time_imphase_NLCG/size(its_imphase_NLCG,2));
% fprintf('time(imphasor_NLCG)/its = %1.4e \n', time_imphasor_NLCG/size(its_imphasor_NLCG,2));
fprintf('time(imphase_LBFGS)/its = %1.4e \n', time_imphase_BFGS/size(its_imphase_BFGS,2));
fprintf('time(imphasor_LBFGS)/its= %1.4e \n', time_imphasor_BFGS/size(its_imphasor_BFGS,2));
fprintf('time(imphase_GN)/its    = %1.4e \n', time_imphase_GN/size(its_imphase_GN,2));
fprintf('time(imphasor_GN)/its   = %1.4e \n', time_imphasor_GN/size(its_imphasor_GN,2));
fprintf('time(imphase_PGN)/its   = %1.4e \n', time_imphase_PGN/size(its_imphase_PGN,2));
fprintf('time(imphasor_PGN)/its  = %1.4e \n', time_imphasor_PGN/size(its_imphasor_PGN,2));
fprintf('time(imphase_PGNR)/its  = %1.4e \n', time_imphase_PGNR/size(its_imphase_PGNR,2));
fprintf('time(imphasor_PGNR)/its = %1.4e \n', time_imphasor_PGNR/size(its_imphasor_PGNR,2));

fprintf('\n***** Outer Iterations til Convergence *****\n');
fprintf('iters(imphase_GD)      = %d \n', size(its_imphase_GD,2)-1);
fprintf('iters(imphasor_GD)     = %d \n', size(its_imphasor_GD,2)-1);
fprintf('iters(imphase_PGD)     = %d \n', size(its_imphase_PGD,2)-1);
fprintf('iters(imphasor_PGD)    = %d \n', size(its_imphasor_PGD,2)-1);
% fprintf('iters(imphase_NLCG)    = %d \n', size(its_imphase_NLCG,2)-1);
% fprintf('iters(imphasor_NLCG)   = %d \n', size(its_imphasor_NLCG,2)-1);
fprintf('iters(imphase_LBFGS)   = %d \n', size(its_imphase_BFGS,2)-1);
fprintf('iters(imphasor_LBFGS)  = %d \n', size(its_imphasor_BFGS,2)-1);
fprintf('iters(imphase_GN)      = %d \n', size(its_imphase_GN,2)-1);
fprintf('iters(imphasor_GN)     = %d \n', size(its_imphasor_GN,2)-1);
fprintf('iters(imphase_PGN)     = %d \n', size(its_imphase_PGN,2)-1);
fprintf('iters(imphasor_PGN)    = %d \n', size(its_imphasor_PGN,2)-1);
fprintf('iters(imphase_PGNR)    = %d \n', size(its_imphase_PGNR,2)-1);
fprintf('iters(imphasor_PGNR)   = %d \n', size(its_imphasor_PGNR,2)-1);

fprintf('\n***** Avg. Line Search Iterations per Outer Iteration *****\n');
fprintf('LS(imphase_GD)/its     = %1.1f \n', sum(his_imphase_GD.array(:,6)/(size(its_imphase_GD,2)-1)));
fprintf('LS(imphasor_GD)/its    = %1.1f \n', sum(his_imphasor_GD.array(:,6)/(size(its_imphasor_GD,2)-1)));
fprintf('LS(imphase_PGD)/its    = %1.1f \n', sum(his_imphase_PGD.array(:,6)/(size(its_imphase_PGD,2)-1)));
fprintf('LS(imphasor_PGD)/its   = %1.1f \n', sum(his_imphasor_PGD.array(:,6)/(size(its_imphasor_PGD,2)-1)));
% fprintf('LS(imphase_NLCG)/its   = %1.1f \n', sum(his_imphase_NLCG.array(:,5)/(size(its_imphase_NLCG,2)-1)));
% fprintf('LS(imphasor_NLCG)/its  = %1.1f \n', sum(his_imphasor_NLCG.array(:,5)/(size(its_imphasor_NLCG,2)-1)));
fprintf('LS(imphase_LBFGS)/its  = %1.1f \n', sum(his_imphase_BFGS.array(:,5)/(size(its_imphase_BFGS,2)-1)));
fprintf('LS(imphasor_LBFGS)/its = %1.1f \n', sum(his_imphasor_BFGS.array(:,5)/(size(its_imphasor_BFGS,2)-1)));
fprintf('LS(imphase_GN)/its     = %1.1f \n', sum(his_imphase_GN.array(:,7)/(size(its_imphase_GN,2)-1)));
fprintf('LS(imphasor_GN)/its    = %1.1f \n', sum(his_imphasor_GN.array(:,7)/(size(its_imphasor_GN,2)-1)));
fprintf('LS(imphase_PGN)/its    = %1.1f \n', sum(his_imphase_PGN.array(:,7)/(size(its_imphase_PGN,2)-1)));
fprintf('LS(imphasor_PGN)/its   = %1.1f \n', sum(his_imphasor_PGN.array(:,7)/(size(its_imphasor_PGN,2)-1)));
fprintf('LS(imphase_PGNR)/its   = %1.1f \n', sum(his_imphase_PGNR.array(:,7)/(size(its_imphase_PGNR,2)-1)));
fprintf('LS(imphasor_PGNR)/its  = %1.1f \n', sum(his_imphasor_PGNR.array(:,7)/(size(its_imphasor_PGNR,2)-1)));
