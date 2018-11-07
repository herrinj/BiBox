%
%   This file tests optimization using projected Gauss-Newton with the
%   phase and phasor objective functions. 
%

clear all; close all;

% Setup data
path_SpeckleImagingCodes;
[nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',100,'D_r0',30);
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
tolN         = 5e-3;
maxIter      = 100;
solverMaxIter= 250;              
solverTol    = 1e-1;

%%
% Run gradient descent for phase
fctn = @(x) phaseObjFctn(x, A, bispec_phase,'weights',weights);
tic();
[phase_GD, his_phase_GD] = GradientDescentProj(fctn, phase_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'iterSave',true);
time_phase_GD = toc();
clear GradientDescentProj;
clear phaseObjFctn;
phase_GD  = phase_foldout(reshape(phase_GD,[256 256]), 0);
image_GD  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GD(:)),[256 256])))));

% Run gradient descent for phasor
fctn = @(x) phasorObjFctn(x, A, bispec_phase,'weights',weights);
tic();
[phasor_GD, his_phasor_GD] = GradientDescentProj(fctn, phase_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'iterSave',true);                        
time_phasor_GD = toc();
clear GradientDescentProj;
clear phasorObjFctn;
phasor_GD  = phase_foldout(reshape(phasor_GD,[256 256]), 0);
imagor_GD  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GD(:)),[256 256])))));

%%
% Run NLCG for phase
fctn = @(x) phaseObjFctn(x, A, bispec_phase,'weights',weights);
[phase_NLCG, his_phase_NLCG] = NonlinearCG(fctn, phase_recur(:), 'maxIter',maxIter,...
                                           'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                           'iterSave',true);
time_phase_NLCG = toc();
clear NonlinearCG;
clear phaseObjFctn;
phase_NLCG  = phase_foldout(reshape(phase_NLCG,[256 256]), 0);
image_NLCG  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_NLCG(:)),[256 256])))));


% Run NLCG for phasor
fctn = @(x) phasorObjFctn(x, A, bispec_phase,'weights',weights);
tic();
[phasor_NLCG, his_phasor_NLCG] = NonlinearCG(fctn, phase_recur(:),'maxIter',maxIter,...
                                             'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                             'iterSave',true);
time_phasor_NLCG = toc();
clear NonlinearCG;
clear phasorObjFctn;
phasor_NLCG  = phase_foldout(reshape(phasor_NLCG,[256 256]), 0);
imagor_NLCG  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_NLCG(:)),[256 256])))));

%%
% Run LBFGS for phase
fctn = @(x) phaseObjFctn(x, A, bispec_phase,'weights',weights);
[phase_BFGS, his_phase_BFGS] = LBFGS(fctn, phase_recur(:), 'maxIter',maxIter,...
                                     'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                     'iterSave',true);
time_phase_BFGS = toc();
clear LBFGS;
clear phaseObjFctn;
phase_BFGS  = phase_foldout(reshape(phase_BFGS,[256 256]), 0);
image_BFGS  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_BFGS(:)),[256 256])))));


% Run BFGS for phasor
fctn = @(x) phasorObjFctn(x, A, bispec_phase,'weights',weights);
tic();
[phasor_BFGS, his_phasor_BFGS] = LBFGS(fctn, phase_recur(:),'maxIter',maxIter,...
                                       'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                       'iterSave',true);
time_phasor_BFGS = toc();
clear LBFGS;
clear phasorObjFctn;
phasor_BFGS  = phase_foldout(reshape(phasor_BFGS,[256 256]), 0);
imagor_BFGS  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_BFGS(:)),[256 256])))));


%%
% Run Gauss-Newton for phase
fctn = @(x) phaseObjFctn(x, A, bispec_phase,'Hflag','full','weights',weights);
tic();
[phase_GNF, his_phase_GNF] = GaussNewtonProj(fctn, phase_recur(:),...
                                           'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                           'tolN',tolN,'solver','bispPhFull','solverMaxIter',250,...
                                           'solverTol',1e-1,'iterSave',true);
time_phase_GNF = toc();
clear GaussNewtonProj;
clear phaseObjFctn;
phase_GNF  = phase_foldout(reshape(phase_GNF,[256 256]), 0);
image_GNF  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GNF(:)),[256 256])))));


% Run Gauss-Newton for imphasor
fctn = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','full','weights',weights);
tic();
[phasor_GNF, his_phasor_GNF] = GaussNewtonProj(fctn, phase_recur(:),...
                                             'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                             'tolN',tolN,'solver','bispPhFull','solverMaxIter',250,...
                                             'solverTol',1e-1,'iterSave',true);
time_phasor_GNF = toc();
clear GaussNewtonProj;
clear phasorObjFctn;
phasor_GNF  = phase_foldout(reshape(phasor_GNF,[256 256]), 0);
imagor_GNF  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GNF(:)),[256 256])))));

%%
% Run Gauss-Newton for phase
fctn = @(x) phaseObjFctn(x, A, bispec_phase,'Hflag','trunc','weights',weights);
tic();
[phase_GNT, his_phase_GNT] = GaussNewtonProj(fctn, phase_recur(:),...
                                           'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                           'tolN',tolN,'solver','bispPhTrunc','solverMaxIter',250,...
                                           'solverTol',1e-1,'iterSave',true);
time_phase_GNT = toc();
clear GaussNewtonProj;
clear phaseObjFctn;
phase_GNT  = phase_foldout(reshape(phase_GNT,[256 256]), 0);
image_GNT  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GNT(:)),[256 256])))));

% Run Gauss-Newton for imphasor
fctn = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','trunc','weights',weights);
tic();
[phasor_GNT, his_phasor_GNT] = GaussNewtonProj(fctn, phase_recur(:),...
                                             'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                             'tolN',tolN,'solver','bispPhTrunc','solverMaxIter',250,...
                                             'solverTol',1e-1,'iterSave',true);
time_phasor_GNT = toc();
clear GaussNewtonProj;
clear phasorObjFctn;
phasor_GNT  = phase_foldout(reshape(phasor_GNT,[256 256]), 0);
imagor_GNT  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GNT(:)),[256 256])))));


%%
% Run Gauss-Newton for phase
fctn = @(x) phaseObjFctn(x, A, bispec_phase,'Hflag','ichol','weights',weights);
tic();
[phase_GNI, his_phase_GNI] = GaussNewtonProj(fctn, phase_recur(:),...
                                           'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                           'tolN',tolN,'solver','bispPhIchol','solverMaxIter',250,...
                                           'solverTol',1e-1,'iterSave',true);
time_phase_GNI = toc();
clear GaussNewtonProj;
clear phaseObjFctn;
phase_GNI  = phase_foldout(reshape(phase_GNI,[256 256]), 0);
image_GNI  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GNI(:)),[256 256])))));

% Run Gauss-Newton for imphasor
fctn = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','ichol','weights',weights);
tic();
[phasor_GNI, his_phasor_GNI] = GaussNewtonProj(fctn, phase_recur(:),...
                                             'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                             'tolN',tolN,'solver','bispPhIchol','solverMaxIter',250,...
                                             'solverTol',1e-1,'iterSave',true);
time_phasor_GNI = toc();
clear GaussNewtonProj;
clear phasorObjFctn;
phasor_GNI  = phase_foldout(reshape(phasor_GNI,[256 256]), 0);
imagor_GNI  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GNI(:)),[256 256])))));

%%
% Look at some results
obj             = obj/max(obj(:));
image_recur     = image_recur/max(image_recur(:));
avg_data_frame  = avg_data_frame/max(avg_data_frame(:));
image_GD        = image_GD/max(image_GD(:));
image_NLCG      = image_NLCG/max(image_NLCG(:));
image_BFGS      = image_BFGS/max(image_BFGS(:));
image_GNF       = image_GNF/max(image_GNF(:));
image_GNT       = image_GNT/max(image_GNT(:));
image_GNI       = image_GNI/max(image_GNI(:));
imagor_GD       = imagor_GD/max(imagor_GD(:));
imagor_NLCG     = imagor_NLCG/max(imagor_NLCG(:));
imagor_BFGS     = imagor_BFGS/max(imagor_BFGS(:));
imagor_GNF      = imagor_GNF/max(imagor_GNF(:));
imagor_GNT      = imagor_GNT/max(imagor_GNT(:));
imagor_GNI      = imagor_GNI/max(imagor_GNI(:));

figure; 
subplot(3,6,1);  imagesc(reshape(obj,[256 256])); axis image; axis off; colorbar; title('truth'); 
subplot(3,6,2);  imagesc(reshape(image_recur,[256 256])); axis image; axis off; colorbar; title('recur');
subplot(3,6,3);  imagesc(avg_data_frame); axis image; axis off; colorbar;  title('avg. blurred frame');

subplot(3,6,7);  imagesc(image_GD); axis image; axis off; colorbar;  title('Imphase - GD');
subplot(3,6,8);  imagesc(image_NLCG); axis image; axis off; colorbar;  title('Imphase - NLCG');
subplot(3,6,9);  imagesc(image_BFGS); axis image; axis off; colorbar;  title('Imphase - LBFGS');
subplot(3,6,10); imagesc(image_GNF); axis image; axis off; colorbar;  title('Imphase - GNF');
subplot(3,6,11); imagesc(image_GNT); axis image; axis off; colorbar;  title('Imphase - GNT');
subplot(3,6,12); imagesc(image_GNI); axis image; axis off; colorbar;  title('Imphase - GNI');

subplot(3,6,13); imagesc(imagor_GD); axis image; axis off; colorbar;  title('Imphasor - GD');
subplot(3,6,14); imagesc(imagor_NLCG); axis image; axis off; colorbar;  title('Imphasor - NLCG');
subplot(3,6,15); imagesc(imagor_BFGS); axis image; axis off; colorbar;  title('Imphasor - LBFGS');
subplot(3,6,16); imagesc(imagor_GNF); axis image; axis off; colorbar;  title('Imphasor - GNF');
subplot(3,6,17); imagesc(imagor_GNT); axis image; axis off; colorbar;  title('Imphasor - GNT');
subplot(3,6,18); imagesc(imagor_GNI); axis image; axis off; colorbar;  title('Imphasor - GNI');

%%
% Relative objective function

figure();
plot((0:size(his_phase_GD.array,1)-1)',his_phase_GD.array(:,2)/his_phase_GD.array(1,2)); 
hold on;
plot((0:size(his_phasor_GD.array,1)-1)',his_phasor_GD.array(:,2)/his_phasor_GD.array(1,2)); 
plot((0:size(his_phase_NLCG.array,1)-1)',his_phase_NLCG.array(:,2)/his_phase_NLCG.array(1,2)); 
plot((0:size(his_phasor_NLCG.array,1)-1)',his_phasor_NLCG.array(:,2)/his_phasor_NLCG.array(1,2)); 
plot((0:size(his_phase_BFGS.array,1)-1)',his_phase_BFGS.array(:,2)/his_phase_BFGS.array(1,2)); 
plot((0:size(his_phasor_BFGS.array,1)-1)',his_phasor_BFGS.array(:,2)/his_phasor_BFGS.array(1,2)); 
plot((0:size(his_phase_GNF.array,1)-1)',his_phase_GNF.array(:,2)/his_phase_GNF.array(1,2)); 
plot((0:size(his_phasor_GNF.array,1)-1)',his_phasor_GNF.array(:,2)/his_phasor_GNF.array(1,2)); 
plot((0:size(his_phase_GNT.array,1)-1)',his_phase_GNT.array(:,2)/his_phase_GNT.array(1,2)); 
plot((0:size(his_phasor_GNT.array,1)-1)',his_phasor_GNT.array(:,2)/his_phasor_GNT.array(1,2));
plot((0:size(his_phase_GNI.array,1)-1)',his_phase_GNI.array(:,2)/his_phase_GNI.array(1,2)); 
plot((0:size(his_phasor_GNI.array,1)-1)',his_phasor_GNI.array(:,2)/his_phasor_GNI.array(1,2));
leg = legend('E1-GD','E2-GD','E1-NLCG','E2-NLCG','E1-LBFGS','E2-LBFGS',...
             'E1-GNF', 'E2-GNF','E1-GNT','E2-GNT','E1-GNI','E2-GNI');
leg.FontSize = 14;
tit = title('Rel. Obj. Func: $\frac{\|J\|}{\|J(0)\|}$','interpreter','latex');
tit.FontSize = 16;

%%
% Relative error plots
its_phase_GD      = reshape(his_phase_GD.iters, 256, 256, []);
its_phasor_GD     = reshape(his_phasor_GD.iters, 256, 256, []);
its_phase_NLCG    = reshape(his_phase_NLCG.iters, 256, 256, []);
its_phasor_NLCG   = reshape(his_phasor_NLCG.iters, 256, 256, []);
its_phase_BFGS    = reshape(his_phase_BFGS.iters, 256, 256, []);
its_phasor_BFGS   = reshape(his_phasor_BFGS.iters, 256, 256, []);
its_phase_GNF     = reshape(his_phase_GNF.iters, 256, 256, []);
its_phasor_GNF    = reshape(his_phasor_GNF.iters, 256, 256, []);
its_phase_GNT     = reshape(his_phase_GNT.iters, 256, 256, []);
its_phasor_GNT    = reshape(his_phasor_GNT.iters, 256, 256, []);
its_phase_GNI     = reshape(his_phase_GNI.iters, 256, 256, []);
its_phasor_GNI    = reshape(his_phasor_GNI.iters, 256, 256, []);

RE_phase_GD       = zeros(size(its_phase_GD,3),1);
RE_phasor_GD      = zeros(size(its_phasor_GD,3),1);
RE_phase_NLCG     = zeros(size(its_phase_NLCG,3),1);
RE_phasor_NLCG    = zeros(size(its_phasor_NLCG,3),1);
RE_phase_BFGS     = zeros(size(its_phase_BFGS,3),1);
RE_phasor_BFGS    = zeros(size(its_phasor_BFGS,3),1);
RE_phase_GNF      = zeros(size(its_phase_GNF,3),1);
RE_phasor_GNF     = zeros(size(its_phasor_GNF,3),1);
RE_phase_GNT      = zeros(size(its_phase_GNT,3),1);
RE_phasor_GNT     = zeros(size(its_phasor_GNT,3),1);
RE_phase_GNI      = zeros(size(its_phase_GNI,3),1);
RE_phasor_GNI     = zeros(size(its_phasor_GNI,3),1);

for j = 1:length(RE_phase_GD)
   active_ph  = phase_foldout(its_phase_GD(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phase_GD(j) = norm(((active_im(:)/max(active_im(:))) - obj(:))/norm(obj(:)));  
end

for j = 1:length(RE_phasor_GD)
   active_ph  = phase_foldout(its_phasor_GD(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phasor_GD(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phase_NLCG)
   active_ph  = phase_foldout(its_phase_NLCG(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phase_NLCG(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phasor_NLCG)
   active_ph  = phase_foldout(its_phasor_NLCG(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phasor_NLCG(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phase_BFGS)
   active_ph  = phase_foldout(its_phase_BFGS(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phase_BFGS(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phasor_BFGS)
   active_ph  = phase_foldout(its_phasor_BFGS(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phasor_BFGS(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phase_GNF)
   active_ph  = phase_foldout(its_phase_GNF(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phase_GNF(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phasor_GNF)
   active_ph  = phase_foldout(its_phasor_GNF(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phasor_GNF(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phase_GNT)
   active_ph  = phase_foldout(its_phase_GNT(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phase_GNT(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phasor_GNT)
   active_ph  = phase_foldout(its_phasor_GNT(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phasor_GNT(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phase_GNI)
   active_ph  = phase_foldout(its_phase_GNI(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phase_GNI(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

for j = 1:length(RE_phasor_GNI)
   active_ph  = phase_foldout(its_phasor_GNI(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   active_im  = active_im/max(active_im(:));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   RE_phasor_GNI(j) = norm((active_im(:) - obj(:))/norm(obj(:)));
end

clear active_ph active_im;

figure();
plot((0:length(RE_phase_GD)-1)',    RE_phase_GD); 
hold on;
plot((0:length(RE_phasor_GD)-1)' ,  RE_phasor_GD); 
plot((0:length(RE_phase_NLCG)-1)',  RE_phase_NLCG); 
plot((0:length(RE_phasor_NLCG)-1)', RE_phasor_NLCG); 
plot((0:length(RE_phase_BFGS)-1)',  RE_phase_BFGS); 
plot((0:length(RE_phasor_BFGS)-1)', RE_phasor_BFGS);
plot((0:length(RE_phase_GNF)-1)',   RE_phase_GNF); 
plot((0:length(RE_phasor_GNF)-1)',  RE_phasor_GNF); 
plot((0:length(RE_phase_GNT)-1)',   RE_phase_GNT); 
plot((0:length(RE_phasor_GNT)-1)',  RE_phasor_GNT); 
plot((0:length(RE_phase_GNI)-1)',   RE_phase_GNI); 
plot((0:length(RE_phasor_GNI)-1)',  RE_phasor_GNI);
leg = legend('E1-GD','E2-GD','E1-NLCG','E2-NLCG','E1-LBFGS','E2-LBFGS',...
             'E1-GNF', 'E2-GNF','E1-GNT','E2-GNT','E1-GNI','E2-GNI');
leg.FontSize = 14;
tit = title('RE: $\frac{\|x-x_{true}\|^2}{\|x_{true}\|^2}$','interpreter','latex');
tit.FontSize = 16;

%%
% Relative error plots
its_phase_GD      = reshape(his_phase_GD.iters, 256, 256, []);
its_phasor_GD     = reshape(his_phasor_GD.iters, 256, 256, []);
its_phase_NLCG    = reshape(his_phase_NLCG.iters, 256, 256, []);
its_phasor_NLCG   = reshape(his_phasor_NLCG.iters, 256, 256, []);
its_phase_BFGS    = reshape(his_phase_BFGS.iters, 256, 256, []);
its_phasor_BFGS   = reshape(his_phasor_BFGS.iters, 256, 256, []);
its_phase_GNF     = reshape(his_phase_GNF.iters, 256, 256, []);
its_phasor_GNF    = reshape(his_phasor_GNF.iters, 256, 256, []);
its_phase_GNT     = reshape(his_phase_GNT.iters, 256, 256, []);
its_phasor_GNT    = reshape(his_phasor_GNT.iters, 256, 256, []);
its_phase_GNI     = reshape(his_phase_GNI.iters, 256, 256, []);
its_phasor_GNI    = reshape(his_phasor_GNI.iters, 256, 256, []);

NCC_phase_GD      = zeros(size(its_phase_GD,3),1);
NCC_phasor_GD     = zeros(size(its_phasor_GD,3),1);
NCC_phase_NLCG    = zeros(size(its_phase_NLCG,3),1);
NCC_phasor_NLCG   = zeros(size(its_phasor_NLCG,3),1);
NCC_phase_BFGS    = zeros(size(its_phase_BFGS,3),1);
NCC_phasor_BFGS   = zeros(size(its_phasor_BFGS,3),1);
NCC_phase_GNF     = zeros(size(its_phase_GNF,3),1);
NCC_phasor_GNF    = zeros(size(its_phasor_GNF,3),1);
NCC_phase_GNT     = zeros(size(its_phase_GNT,3),1);
NCC_phasor_GNT    = zeros(size(its_phasor_GNT,3),1);
NCC_phase_GNI     = zeros(size(its_phase_GNI,3),1);
NCC_phasor_GNI    = zeros(size(its_phasor_GNI,3),1);

for j = 1:length(NCC_phase_GD)
   active_ph  = phase_foldout(its_phase_GD(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phase_GD(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phasor_GD)
   active_ph  = phase_foldout(its_phasor_GD(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phasor_GD(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phase_NLCG)
   active_ph  = phase_foldout(its_phase_NLCG(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phase_NLCG(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phasor_NLCG)
   active_ph  = phase_foldout(its_phasor_NLCG(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phasor_NLCG(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phase_BFGS)
   active_ph  = phase_foldout(its_phase_BFGS(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phase_BFGS(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phasor_BFGS)
   active_ph  = phase_foldout(its_phasor_BFGS(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phasor_BFGS(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phase_GNF)
   active_ph  = phase_foldout(its_phase_GNF(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phase_GNF(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phasor_GNF)
   active_ph  = phase_foldout(its_phasor_GNF(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phasor_GNF(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phase_GNT)
   active_ph  = phase_foldout(its_phase_GNT(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phase_GNT(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phasor_GNT)
   active_ph  = phase_foldout(its_phasor_GNT(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phasor_GNT(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phase_GNI)
   active_ph  = phase_foldout(its_phase_GNI(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phase_GNI(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

for j = 1:length(NCC_phasor_GNI)
   active_ph  = phase_foldout(its_phasor_GNI(:,:,j), 0);
   active_im  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*active_ph(:)),[256 256])))));
   s = measureShift(obj,active_im);
   active_im  = shiftImage(active_im,s);
   active_im  = active_im(:)/max(active_im(:));
   NCC_phasor_GNI(j) = 0.5 - 0.5*((active_im(:)'*obj(:))^2/((active_im(:)'*active_im(:))*(obj(:)'*obj(:))));  
end

clear active_ph active_im;

figure();
plot((0:length(NCC_phase_GD)-1)',    NCC_phase_GD); 
hold on;
plot((0:length(NCC_phasor_GD)-1)' ,  NCC_phasor_GD); 
plot((0:length(NCC_phase_NLCG)-1)',  NCC_phase_NLCG); 
plot((0:length(NCC_phasor_NLCG)-1)', NCC_phasor_NLCG); 
plot((0:length(NCC_phase_BFGS)-1)',  NCC_phase_BFGS); 
plot((0:length(NCC_phasor_BFGS)-1)', NCC_phasor_BFGS);
plot((0:length(NCC_phase_GNF)-1)',   NCC_phase_GNF); 
plot((0:length(NCC_phasor_GNF)-1)',  NCC_phasor_GNF); 
plot((0:length(NCC_phase_GNT)-1)',   NCC_phase_GNT); 
plot((0:length(NCC_phasor_GNT)-1)',  NCC_phasor_GNT); 
plot((0:length(NCC_phase_GNI)-1)',   NCC_phase_GNI); 
plot((0:length(NCC_phasor_GNI)-1)',  NCC_phasor_GNI);
leg = legend('E1-GD','E2-GD','E1-NLCG','E2-NLCG','E1-LBFGS','E2-LBFGS',...
             'E1-GNF', 'E2-GNF','E1-GNT','E2-GNT','E1-GNI','E2-GNI');
leg.FontSize = 14;
tit = title('NCC:$\frac{1}{2} \left(1 - \frac{\langle x, x_{true}\rangle^2}{\|x\|^2 \|x_{true}\|^2} \right)$','interpreter','latex');
tit.FontSize = 16;

%%
% Print some results to the terminal window
fprintf('\n***** Relative Error Minima *****\n');
fprintf('min(RE_phase_GD)     = %1.4e \n', min(RE_phase_GD));
fprintf('min(RE_phasor_GD)    = %1.4e \n', min(RE_phasor_GD));
fprintf('min(RE_phase_NLCG)   = %1.4e \n', min(RE_phase_NLCG));
fprintf('min(RE_phasor_NLCG)  = %1.4e \n', min(RE_phasor_NLCG));
fprintf('min(RE_phase_LBFGS)  = %1.4e \n', min(RE_phase_BFGS));
fprintf('min(RE_phasor_LBFGS) = %1.4e \n', min(RE_phasor_BFGS));
fprintf('min(RE_phase_GNF)    = %1.4e \n', min(RE_phase_GNF));
fprintf('min(RE_phasor_GNF)   = %1.4e \n', min(RE_phasor_GNF));
fprintf('min(RE_phase_GNT)    = %1.4e \n', min(RE_phase_GNT));
fprintf('min(RE_phasor_GNT)   = %1.4e \n', min(RE_phasor_GNT));
fprintf('min(RE_phase_GNI)    = %1.4e \n', min(RE_phase_GNI));
fprintf('min(RE_phasor_GNI)   = %1.4e \n', min(RE_phasor_GNI));

fprintf('\n***** Normalized Cross-Correlation Minima *****\n');
fprintf('min(NCC_phase_GD)    = %1.4e \n', min(NCC_phase_GD));
fprintf('min(NCC_phasor_GD)   = %1.4e \n', min(NCC_phasor_GD));
fprintf('min(NCC_phase_NLCG)  = %1.4e \n', min(NCC_phase_NLCG));
fprintf('min(NCC_phasor_NLCG) = %1.4e \n', min(NCC_phasor_NLCG));
fprintf('min(NCC_phase_LBFGS) = %1.4e \n', min(NCC_phase_BFGS));
fprintf('min(NCC_phasor_LBFGS)= %1.4e \n', min(NCC_phasor_BFGS));
fprintf('min(NCC_phase_GNF)   = %1.4e \n', min(NCC_phase_GNF));
fprintf('min(NCC_phasor_GNF)  = %1.4e \n', min(NCC_phasor_GNF));
fprintf('min(NCC_phase_GNT)   = %1.4e \n', min(NCC_phase_GNT));
fprintf('min(NCC_phasor_GNT)  = %1.4e \n', min(NCC_phasor_GNT));
fprintf('min(NCC_phase_GNI)   = %1.4e \n', min(NCC_phase_GNI));
fprintf('min(NCC_phasor_GNI)  = %1.4e \n', min(NCC_phasor_GNI));

fprintf('\n***** Total Time Elapsed *****\n');
fprintf('time(phase_GD)       = %1.4e \n', time_phase_GD);
fprintf('time(phasor_GD)      = %1.4e \n', time_phasor_GD);
fprintf('time(phase_NLCG)     = %1.4e \n', time_phase_NLCG);
fprintf('time(phasor_NLCG)    = %1.4e \n', time_phasor_NLCG);
fprintf('time(phase_LBFGS)    = %1.4e \n', time_phase_BFGS);
fprintf('time(phasor_LBFGS)   = %1.4e \n', time_phasor_BFGS);
fprintf('time(phase_GNF)      = %1.4e \n', time_phase_GNF);
fprintf('time(phasor_GNF)     = %1.4e \n', time_phasor_GNF);
fprintf('time(phase_GNT)      = %1.4e \n', time_phase_GNT);
fprintf('time(phasor_GNT)     = %1.4e \n', time_phasor_GNT);
fprintf('time(phase_GNI)      = %1.4e \n', time_phase_GNI);
fprintf('time(phasor_GNI)     = %1.4e \n', time_phasor_GNI);

fprintf('\n***** Time per Iteration *****\n');
fprintf('time(phase_GD)/its    = %1.4e \n', time_phase_GD/size(its_phase_GD,3));
fprintf('time(phasor_GD)/its   = %1.4e \n', time_phasor_GD/size(its_phasor_GD,3));
fprintf('time(phase_NLCG)/its  = %1.4e \n', time_phase_NLCG/size(its_phase_NLCG,3));
fprintf('time(phasor_NLCG)/its = %1.4e \n', time_phasor_NLCG/size(its_phasor_NLCG,3));
fprintf('time(phase_LBFGS)/its = %1.4e \n', time_phase_BFGS/size(its_phase_BFGS,3));
fprintf('time(phasor_LBFGS)/its= %1.4e \n', time_phasor_BFGS/size(its_phasor_BFGS,3));
fprintf('time(phase_GNF)/its   = %1.4e \n', time_phase_GNF/size(its_phase_GNF,3));
fprintf('time(phasor_GNF)/its  = %1.4e \n', time_phasor_GNF/size(its_phasor_GNF,3));
fprintf('time(phase_GNT)/its   = %1.4e \n', time_phase_GNT/size(its_phase_GNT,3));
fprintf('time(phasor_GNT)/its  = %1.4e \n', time_phasor_GNT/size(its_phasor_GNT,3));
fprintf('time(phase_GNI)/its   = %1.4e \n', time_phase_GNI/size(its_phase_GNI,3));
fprintf('time(phasor_GNI)/its  = %1.4e \n', time_phasor_GNI/size(its_phasor_GNI,3));

fprintf('\n***** Outer Iterations til Convergence *****\n');
fprintf('iters(phase_GD)      = %d \n', size(its_phase_GD,3)-1);
fprintf('iters(phasor_GD)     = %d \n', size(its_phasor_GD,3)-1);
fprintf('iters(phase_NLCG)    = %d \n', size(its_phase_NLCG,3)-1);
fprintf('iters(phasor_NLCG)   = %d \n', size(its_phasor_NLCG,3)-1);
fprintf('iters(phase_LBFGS)   = %d \n', size(its_phase_BFGS,3)-1);
fprintf('iters(phasor_LBFGS)  = %d \n', size(its_phasor_BFGS,3)-1);
fprintf('iters(phase_GNF)     = %d \n', size(its_phase_GNF,3)-1);
fprintf('iters(phasor_GNF)    = %d \n', size(its_phasor_GNF,3)-1);
fprintf('iters(phase_GNT)     = %d \n', size(its_phase_GNT,3)-1);
fprintf('iters(phasor_GNT)    = %d \n', size(its_phasor_GNT,3)-1);
fprintf('iters(phase_GNI)     = %d \n', size(its_phase_GNI,3)-1);
fprintf('iters(phasor_GNI)    = %d \n', size(its_phasor_GNI,3)-1);

fprintf('\n***** Avg. Line Search Iterations per Outer Iteration *****\n');
fprintf('LS(phase_GD)/its     = %1.1f \n', sum(his_phase_GD.array(:,6)/(size(its_phase_GD,3)-1)));
fprintf('LS(phasor_GD)/its    = %1.1f \n', sum(his_phasor_GD.array(:,6)/(size(its_phasor_GD,3)-1)));
fprintf('LS(phase_NLCG)/its   = %1.1f \n', sum(his_phase_NLCG.array(:,5)/(size(its_phase_NLCG,3)-1)));
fprintf('LS(phasor_NLCG)/its  = %1.1f \n', sum(his_phasor_NLCG.array(:,5)/(size(its_phasor_NLCG,3)-1)));
fprintf('LS(phase_LBFGS)/its  = %1.1f \n', sum(his_phase_BFGS.array(:,5)/(size(its_phase_BFGS,3)-1)));
fprintf('LS(phasor_LBFGS)/its = %1.1f \n', sum(his_phasor_BFGS.array(:,5)/(size(its_phasor_BFGS,3)-1)));
fprintf('LS(phase_GNF)/its    = %1.1f \n', sum(his_phase_GNF.array(:,7)/(size(its_phase_GNF,3)-1)));
fprintf('LS(phasor_GNF)/its   = %1.1f \n', sum(his_phasor_GNF.array(:,7)/(size(its_phasor_GNF,3)-1)));
fprintf('LS(phase_GNT)/its    = %1.1f \n', sum(his_phase_GNT.array(:,7)/(size(its_phase_GNT,3)-1)));
fprintf('LS(phasor_GNT)/its   = %1.1f \n', sum(his_phasor_GNT.array(:,7)/(size(its_phasor_GNT,3)-1)));
fprintf('LS(phase_GNI)/its    = %1.1f \n', sum(his_phase_GNI.array(:,7)/(size(its_phase_GNI,3)-1)));
fprintf('LS(phasor_GNI)/its   = %1.1f \n', sum(his_phasor_GNI.array(:,7)/(size(its_phasor_GNI,3)-1)));
