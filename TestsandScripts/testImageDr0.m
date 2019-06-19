%
%   This file tests optimization using projected Gauss-Newton with the
%   imphase and imphasor objective functions. 
%
clear all; close all;

% Setup Gauss-Newton parameters
n = 256^2;
upper_bound  = inf*ones(n,1);
lower_bound  = zeros(n,1);
tolJ         = 1e-4;            
tolY         = 1e-4;           
tolG         = 1e1;
tolN         = 1e-3;
maxIter      = 100;
solverMaxIter= 250;              
solverTol    = 1e-1;
alphaTV      = 1e4;
Dr0          = [10 20 30 40 50]; 


%%
% Set up loop of runs

m    = numel(Dr0);
runs = 10;

RE_init          = zeros(m,runs);
RE_init_proj     = zeros(m,runs);
RE_phase_GNI     = zeros(m,runs);
RE_phasor_GNI    = zeros(m,runs);
RE_imphase_PGNR  = zeros(m,runs);
RE_imphasor_PGNR = zeros(m,runs);

for j = 1:m
    for k = 1:runs
    
    % Setup data at each run
    path_SpeckleImagingCodes;
    [nfr, D_r0, image_name, K_n, sigma_rn, fourier_rad, second_rad]  = setupBispectrumParams('nfr',100,'D_r0',Dr0(j));
    setupBispectrumData;
    image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_recur(:)),[256 256])))));
    dims = size(image_recur);
    image_proj  = gdnnf_projection(image_recur, sum(image_recur(:))) + 1e-4;
    avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));
    obj = obj/max(obj(:));

    
    RE_init(j,k)      = norm(image_recur(:)/max(image_recur(:)) - obj(:))/norm(obj(:)); 
    RE_init_proj(j,k) = norm(image_proj(:)/max(image_proj(:))  - obj(:))/norm(obj(:)); 

    % Run Gauss-Newton for phase
    fctn = @(x) phaseObjFctn(x, A, bispec_phase,'Hflag','ichol','weights',weights);
    tic();
    [phase_GNI, his_phase_GNI] = GaussNewtonProj(fctn, phase_recur(:),...
                                               'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                               'solver','bispPhIchol','solverMaxIter',250,'solverTol',1e-1,...
                                               'tolN',tolN,'iterSave',true);
    phase_GNI  = phase_foldout(reshape(phase_GNI,[256 256]), 0);
    image_GNI  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GNI(:)),[256 256])))));
    image_GNI  = image_GNI/max(image_GNI(:));
    s          = measureShift(obj,image_GNI);
    image_GNI  = shiftImage(image_GNI,s);
    RE_phase_GNI(j,k)  = norm(image_GNI(:) - obj(:))/norm(obj(:));  

    
    clear GaussNewtonProj;
    clear phaseObjFctn;
    
    % Run Gauss-Newton for phasor
    fctn = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','ichol','weights',weights);
    tic();
    [phasor_GNI, his_phasor_GNI] = GaussNewtonProj(fctn, phase_recur(:),...
                                                 'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                 'solver','bispPhIchol','solverMaxIter',250,'solverTol',1e-1,...
                                                 'tolN',tolN,'iterSave',true);
    phasor_GNI  = phase_foldout(reshape(phasor_GNI,[256 256]), 0);
    imagor_GNI  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GNI(:)),[256 256])))));
    imagor_GNI  = imagor_GNI/max(imagor_GNI(:));
    s           = measureShift(obj,imagor_GNI);
    imagor_GNI  = shiftImage(imagor_GNI,s);
    RE_phasor_GNI(j,k)  = norm(imagor_GNI(:) - obj(:))/norm(obj(:));  
    
    clear GaussNewtonProj;
    clear phasorObjFctn;
    
    
    % Run projected Gauss-Newton for imphase
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaTV,'regularizer','tv','weights',weights);
    tic();
    [imphase_PGNR, his_imphase_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                    'tolN',tolN,'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    imphase_PGNR   = reshape(imphase_PGNR,[256 256])/max(imphase_PGNR(:));
    s              = measureShift(obj,imphase_PGNR);
    imphase_PGNR   = shiftImage(imphase_PGNR,s);
    RE_imphase_PGNR(j,k)  = norm(imphase_PGNR(:) - obj(:))/norm(obj(:));  

    clear GaussNewtonProj;
    clear imphaseObjFctn;

    % Run projected Gauss-Newton for imphasor
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaTV,'regularizer','tv','weights',weights);
    [imphasor_PGNR, his_imphasor_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                      'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                      'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                      'tolN',tolN,'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    imphasor_PGNR  = reshape(imphasor_PGNR,[256 256])/max(imphasor_PGNR(:));
    s              = measureShift(obj,imphasor_PGNR);
    imphasor_PGNR  = shiftImage(imphasor_PGNR,s);
    RE_imphasor_PGNR(j,k) = norm(imphasor_PGNR(:) - obj(:))/norm(obj(:));  

    clear GaussNewtonProj;
    clear imphasorObjFctn;
  
    end
    
end

%%
% Print some results to the terminal window
clc;

for j = 1:m
    fprintf('\n***** Relative Error Minima D/r0 = %d *****\n',Dr0(j));
    fprintf('min(RE_init)          = %1.4e \n', sum(RE_init(j,:),2)/runs);
    fprintf('min(RE_init_proj)     = %1.4e \n', sum(RE_init_proj(j,:),2)/runs);
    fprintf('min(RE_phase_GNI)     = %1.4e \n', sum(RE_phase_GNI(j,:),2)/runs);
    fprintf('min(RE_phasor_GNI)    = %1.4e \n', sum(RE_phasor_GNI(j,:),2)/runs);
    fprintf('min(RE_imphase_PGNR)  = %1.4e \n', sum(RE_imphase_PGNR(j,:),2)/runs);
    fprintf('min(RE_imphasor_PGNR) = %1.4e \n', sum(RE_imphasor_PGNR(j,:),2)/runs);
end


