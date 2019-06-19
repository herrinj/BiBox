%
%   This file tests optimization using projected Gauss-Newton with the
%   phase and phasor objective functions. 
%
clear all; close all;

% Setup Gauss-Newton parameters
n = 256^2;
tolJ         = 1e-4;            
tolY         = 1e-4;           
tolG         = 1e1;
tolN         = 1e-3;
maxIter      = 100;
solverMaxIter= 250;              
solverTol    = 1e-1;

%%
% Set up loop of runs

runs = 50;

% Alot memory
sol_phase_GD       = zeros(n,runs);
sol_phasor_GD      = zeros(n,runs);
sol_phase_NLCG     = zeros(n,runs);
sol_phasor_NLCG    = zeros(n,runs);
sol_phase_BFGS     = zeros(n,runs);
sol_phasor_BFGS    = zeros(n,runs);
sol_phase_GNF      = zeros(n,runs);
sol_phasor_GNF     = zeros(n,runs);
sol_phase_GNT      = zeros(n,runs);
sol_phasor_GNT     = zeros(n,runs);
sol_phase_GNI      = zeros(n,runs);
sol_phasor_GNI     = zeros(n,runs);

ROF_phase_GD       = zeros(runs,1);
ROF_phasor_GD      = zeros(runs,1);
ROF_phase_NLCG     = zeros(runs,1);
ROF_phasor_NLCG    = zeros(runs,1);
ROF_phase_BFGS     = zeros(runs,1);
ROF_phasor_BFGS    = zeros(runs,1);
ROF_phase_GNF      = zeros(runs,1);
ROF_phasor_GNF     = zeros(runs,1);
ROF_phase_GNT      = zeros(runs,1);
ROF_phasor_GNT     = zeros(runs,1);
ROF_phase_GNI      = zeros(runs,1);
ROF_phasor_GNI     = zeros(runs,1);

RE_phase_GD        = zeros(runs,1);
RE_phasor_GD       = zeros(runs,1);
RE_phase_NLCG      = zeros(runs,1);
RE_phasor_NLCG     = zeros(runs,1);
RE_phase_BFGS      = zeros(runs,1);
RE_phasor_BFGS     = zeros(runs,1);
RE_phase_GNF       = zeros(runs,1);
RE_phasor_GNF      = zeros(runs,1);
RE_phase_GNT       = zeros(runs,1);
RE_phasor_GNT      = zeros(runs,1);
RE_phase_GNI       = zeros(runs,1);
RE_phasor_GNI      = zeros(runs,1);

NCC_phase_GD       = zeros(runs,1);
NCC_phasor_GD      = zeros(runs,1);
NCC_phase_NLCG     = zeros(runs,1);
NCC_phasor_NLCG    = zeros(runs,1);
NCC_phase_BFGS     = zeros(runs,1);
NCC_phasor_BFGS    = zeros(runs,1);
NCC_phase_GNF      = zeros(runs,1);
NCC_phasor_GNF     = zeros(runs,1);
NCC_phase_GNT      = zeros(runs,1);
NCC_phasor_GNT     = zeros(runs,1);
NCC_phase_GNI      = zeros(runs,1);
NCC_phasor_GNI     = zeros(runs,1);

time_phase_GD      = zeros(runs,1);
time_phasor_GD     = zeros(runs,1);
time_phase_NLCG    = zeros(runs,1);
time_phasor_NLCG   = zeros(runs,1);
time_phase_BFGS    = zeros(runs,1);
time_phasor_BFGS   = zeros(runs,1);
time_phase_GNF     = zeros(runs,1);
time_phasor_GNF    = zeros(runs,1);
time_phase_GNT     = zeros(runs,1);
time_phasor_GNT    = zeros(runs,1);
time_phase_GNI     = zeros(runs,1);
time_phasor_GNI    = zeros(runs,1);

its_phase_GD       = zeros(runs,1);
its_phasor_GD      = zeros(runs,1);
its_phase_NLCG     = zeros(runs,1);
its_phasor_NLCG    = zeros(runs,1);
its_phase_BFGS     = zeros(runs,1);
its_phasor_BFGS    = zeros(runs,1);
its_phase_GNF      = zeros(runs,1);
its_phasor_GNF     = zeros(runs,1);
its_phase_GNT      = zeros(runs,1);
its_phasor_GNT     = zeros(runs,1);
its_phase_GNI      = zeros(runs,1);
its_phasor_GNI     = zeros(runs,1);

LS_phase_GD        = zeros(runs,1);
LS_phasor_GD       = zeros(runs,1);
LS_phase_NLCG      = zeros(runs,1);
LS_phasor_NLCG     = zeros(runs,1);
LS_phase_BFGS      = zeros(runs,1);
LS_phasor_BFGS     = zeros(runs,1);
LS_phase_GNF       = zeros(runs,1);
LS_phasor_GNF      = zeros(runs,1);
LS_phase_GNT       = zeros(runs,1);
LS_phasor_GNT      = zeros(runs,1);
LS_phase_GNI       = zeros(runs,1);
LS_phasor_GNI      = zeros(runs,1);

for k = 1:runs
    
    % Setup data at each run
    path_SpeckleImagingCodes;
    [nfr, D_r0, image_name, K_n, sigma_rn, fourier_rad, second_rad]  = setupBispectrumParams('nfr',100,'D_r0',30);
    setupBispectrumData;
    image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_recur(:)),[256 256])))));
    dims = size(image_recur);
    avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));
    obj = obj/max(obj(:));
    

    % Run gradient descent for phase
    fctn = @(x) phaseObjFctn(x, A, bispec_phase,'weights',weights);
    tic();
    [phase_GD, his_phase_GD] = GradientDescentProj(fctn, phase_recur(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'iterSave',true);
    time_phase_GD(k) = toc();
    phase_GD  = phase_foldout(reshape(phase_GD,[256 256]), 0);
    image_GD  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GD(:)),[256 256])))));
    image_GD          = image_GD/max(image_GD(:));
    s                 = measureShift(obj,image_GD);
    image_GD          = shiftImage(image_GD,s);
    sol_phase_GD(:,k) = image_GD(:);
    its_phase_GD(k)   = size(his_phase_GD.iters,2)-1;
    LS_phase_GD(k)    = sum(his_phase_GD.array(:,6));
    ROF_phase_GD(k)      = his_phase_GD.array(end,2)/his_phase_GD.array(1,2);
    clear GradientDescentProj;
    clear phaseObjFctn;
    
    % Run gradient descent for phasor
    fctn = @(x) phasorObjFctn(x, A, bispec_phase,'weights',weights);
    tic();
    [phasor_GD, his_phasor_GD] = GradientDescentProj(fctn, phase_recur(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'iterSave',true);                        
    time_phasor_GD(k) = toc();
    phasor_GD  = phase_foldout(reshape(phasor_GD,[256 256]), 0);
    imagor_GD  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GD(:)),[256 256])))));
    imagor_GD           = imagor_GD/max(imagor_GD(:));
    s                   = measureShift(obj,imagor_GD);
    imagor_GD           = shiftImage(imagor_GD,s);
    sol_phasor_GD(:,k)  = imagor_GD(:);
    its_phasor_GD(k)    = size(his_phasor_GD.iters,2)-1;
    LS_phasor_GD(k)     = sum(his_phasor_GD.array(:,6));
    ROF_phasor_GD(k)       = his_phasor_GD.array(end,2)/his_phasor_GD.array(1,2);
    clear GradientDescentProj;
    clear phasorObjFctn;

    % Run NLCG for phase
    fctn = @(x) phaseObjFctn(x, A, bispec_phase,'weights',weights);
    [phase_NLCG, his_phase_NLCG] = NonlinearCG(fctn, phase_recur(:), 'maxIter',maxIter,...
                                               'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                               'iterSave',true);
    time_phase_NLCG(k) = toc();
    phase_NLCG  = phase_foldout(reshape(phase_NLCG,[256 256]), 0);
    image_NLCG  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_NLCG(:)),[256 256])))));
    image_NLCG          = image_NLCG/max(image_NLCG(:));
    s                   = measureShift(obj,image_NLCG);
    image_NLCG          = shiftImage(image_NLCG,s);
    sol_phase_NLCG(:,k) = image_NLCG(:);
    its_phase_NLCG(k)   = size(his_phase_NLCG.iters,2)-1;
    LS_phase_NLCG(k)    = sum(his_phase_NLCG.array(:,5));
    ROF_phase_NLCG(k)   = his_phase_NLCG.array(end,2)/his_phase_NLCG.array(1,2);
    clear NonlinearCG;
    clear phaseObjFctn;

    % Run NLCG for phasor
    fctn = @(x) phasorObjFctn(x, A, bispec_phase,'weights',weights);
    tic();
    [phasor_NLCG, his_phasor_NLCG] = NonlinearCG(fctn, phase_recur(:),'maxIter',maxIter,...
                                                 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                 'iterSave',true);
    time_phasor_NLCG(k) = toc();
    phasor_NLCG  = phase_foldout(reshape(phasor_NLCG,[256 256]), 0);
    imagor_NLCG  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_NLCG(:)),[256 256])))));
    imagor_NLCG           = imagor_NLCG/max(imagor_NLCG(:));
    s                     = measureShift(obj,imagor_NLCG);
    imagor_NLCG           = shiftImage(imagor_NLCG,s);
    sol_phasor_NLCG(:,k)  = imagor_NLCG(:);
    its_phasor_NLCG(k)    = size(his_phasor_NLCG.iters,2)-1;
    LS_phasor_NLCG(k)     = sum(his_phasor_NLCG.array(:,5));
    ROF_phasor_NLCG(k)    = his_phasor_NLCG.array(end,2)/his_phasor_NLCG.array(1,2);
    clear NonlinearCG;
    clear phasorObjFctn;

    % Run LBFGS for phase
    fctn = @(x) phaseObjFctn(x, A, bispec_phase,'weights',weights);
    [phase_BFGS, his_phase_BFGS] = LBFGS(fctn, phase_recur(:), 'maxIter',maxIter,...
                                         'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                         'iterSave',true);
    time_phase_BFGS(k) = toc();
    phase_BFGS  = phase_foldout(reshape(phase_BFGS,[256 256]), 0);
    image_BFGS  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_BFGS(:)),[256 256])))));
    image_BFGS          = image_BFGS/max(image_BFGS(:));
    s                   = measureShift(obj,image_BFGS);
    image_BFGS          = shiftImage(image_BFGS,s);
    sol_phase_BFGS(:,k) = image_BFGS(:);
    its_phase_BFGS(k)   = size(his_phase_BFGS.iters,2)-1;
    LS_phase_BFGS(k)    = sum(his_phase_BFGS.array(:,5));
    ROF_phase_BFGS(k)   = his_phase_BFGS.array(end,2)/his_phase_BFGS.array(1,2);
    clear LBFGS;
    clear phaseObjFctn;

    % Run BFGS for phasor
    fctn = @(x) phasorObjFctn(x, A, bispec_phase,'weights',weights);
    tic();
    [phasor_BFGS, his_phasor_BFGS] = LBFGS(fctn, phase_recur(:),'maxIter',maxIter,...
                                           'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                           'iterSave',true);
    time_phasor_BFGS(k) = toc();
    phasor_BFGS  = phase_foldout(reshape(phasor_BFGS,[256 256]), 0);
    imagor_BFGS  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_BFGS(:)),[256 256])))));
    imagor_BFGS           = imagor_BFGS/max(imagor_BFGS(:));
    s                     = measureShift(obj,imagor_BFGS);
    imagor_BFGS           = shiftImage(imagor_BFGS,s);
    sol_phasor_BFGS(:,k)  = imagor_BFGS(:);
    its_phasor_BFGS(k)    = size(his_phasor_BFGS.iters,2)-1;
    LS_phasor_BFGS(k)     = sum(his_phasor_BFGS.array(:,5));
    ROF_phasor_BFGS(k)   = his_phasor_BFGS.array(end,2)/his_phasor_BFGS.array(1,2);
    clear LBFGS;
    clear phasorObjFctn;


    % Run Gauss-Newton for phase
    fctn = @(x) phaseObjFctn(x, A, bispec_phase,'Hflag','full','weights',weights);
    tic();
    [phase_GNF, his_phase_GNF] = GaussNewtonProj(fctn, phase_recur(:),...
                                               'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                               'solver','bispPhFull','solverMaxIter',250,'solverTol',1e-1,...
                                               'tolN',tolN,'iterSave',true);
    time_phase_GNF(k) = toc();
    phase_GNF  = phase_foldout(reshape(phase_GNF,[256 256]), 0);
    image_GNF  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GNF(:)),[256 256])))));
    image_GNF          = image_GNF/max(image_GNF(:));
    s                  = measureShift(obj,image_GNF);
    image_GNF          = shiftImage(image_GNF,s);
    sol_phase_GNF(:,k) = image_GNF(:);
    its_phase_GNF(k)   = size(his_phase_GNF.iters,2)-1;
    LS_phase_GNF(k)    = sum(his_phase_GNF.array(:,7));
    ROF_phase_GNF(k)   = his_phase_GNF.array(end,2)/his_phase_GNF.array(1,2);
    clear GaussNewtonProj;
    clear phaseObjFctn;

    % Run Gauss-Newton for phasor
    fctn = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','full','weights',weights);
    tic();
    [phasor_GNF, his_phasor_GNF] = GaussNewtonProj(fctn, phase_recur(:),...
                                                 'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                 'solver','bispPhFull','solverMaxIter',250,'solverTol',1e-1,...
                                                 'tolN',tolN,'iterSave',true);
    time_phasor_GNF(k) = toc();
    phasor_GNF  = phase_foldout(reshape(phasor_GNF,[256 256]), 0);
    imagor_GNF  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GNF(:)),[256 256])))));
    imagor_GNF           = imagor_GNF/max(imagor_GNF(:));
    s                    = measureShift(obj,imagor_GNF);
    imagor_GNF           = shiftImage(imagor_GNF,s);
    sol_phasor_GNF(:,k)  = imagor_GNF(:);
    its_phasor_GNF(k)    = size(his_phasor_GNF.iters,2)-1;
    LS_phasor_GNF(k)     = sum(his_phasor_GNF.array(:,7));
    ROF_phasor_GNF(k)    = his_phasor_GNF.array(end,2)/his_phasor_GNF.array(1,2);
    clear GaussNewtonProj;
    clear phasorObjFctn;

    % Run Gauss-Newton for phase
    fctn = @(x) phaseObjFctn(x, A, bispec_phase,'Hflag','trunc','weights',weights);
    tic();
    [phase_GNT, his_phase_GNT] = GaussNewtonProj(fctn, phase_recur(:),...
                                               'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                               'solver','bispPhTrunc','solverMaxIter',250,'solverTol',1e-1,...
                                               'tolN',tolN,'iterSave',true);
    time_phase_GNT(k) = toc();
    phase_GNT  = phase_foldout(reshape(phase_GNT,[256 256]), 0);
    image_GNT  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GNT(:)),[256 256])))));
    image_GNT          = image_GNT/max(image_GNT(:));
    s                 = measureShift(obj,image_GNT);
    image_GNT          = shiftImage(image_GNT,s);
    sol_phase_GNT(:,k) = image_GNT(:);
    its_phase_GNT(k)   = size(his_phase_GNT.iters,2)-1;
    LS_phase_GNT(k)    = sum(his_phase_GNT.array(:,7));
    ROF_phase_GNT(k)   = his_phase_GNT.array(end,2)/his_phase_GNT.array(1,2);
    clear GaussNewtonProj;
    clear phaseObjFctn;
    
    % Run Gauss-Newton for phasor
    fctn = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','trunc','weights',weights);
    tic();
    [phasor_GNT, his_phasor_GNT] = GaussNewtonProj(fctn, phase_recur(:),...
                                                 'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                 'solver','bispPhTrunc','solverMaxIter',250,'solverTol',1e-1,...
                                                 'tolN',tolN,'iterSave',true);
    time_phasor_GNT(k) = toc();
    phasor_GNT  = phase_foldout(reshape(phasor_GNT,[256 256]), 0);
    imagor_GNT  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GNT(:)),[256 256])))));
    imagor_GNT           = imagor_GNT/max(imagor_GNT(:));
    s                   = measureShift(obj,imagor_GNT);
    imagor_GNT           = shiftImage(imagor_GNT,s);
    sol_phasor_GNT(:,k)  = imagor_GNT(:);
    its_phasor_GNT(k)    = size(his_phasor_GNT.iters,2)-1;
    LS_phasor_GNT(k)     = sum(his_phasor_GNT.array(:,7));
    ROF_phasor_GNT(k)   = his_phasor_GNT.array(end,2)/his_phasor_GNT.array(1,2);
    clear GaussNewtonProj;
    clear phasorObjFctn;


    % Run Gauss-Newton for phase
    fctn = @(x) phaseObjFctn(x, A, bispec_phase,'Hflag','ichol','weights',weights);
    tic();
    [phase_GNI, his_phase_GNI] = GaussNewtonProj(fctn, phase_recur(:),...
                                               'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                               'solver','bispPhIchol','solverMaxIter',250,'solverTol',1e-1,...
                                               'tolN',tolN,'iterSave',true);
    time_phase_GNI(k) = toc();
    phase_GNI  = phase_foldout(reshape(phase_GNI,[256 256]), 0);
    image_GNI  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_GNI(:)),[256 256])))));
    image_GNI          = image_GNI/max(image_GNI(:));
    s                 = measureShift(obj,image_GNI);
    image_GNI          = shiftImage(image_GNI,s);
    sol_phase_GNI(:,k) = image_GNI(:);
    its_phase_GNI(k)   = size(his_phase_GNI.iters,2)-1;
    LS_phase_GNI(k)    = sum(his_phase_GNI.array(:,7));
    ROF_phase_GNI(k)   = his_phase_GNI.array(end,2)/his_phase_GNI.array(1,2);
    clear GaussNewtonProj;
    clear phaseObjFctn;
    
    % Run Gauss-Newton for phasor
    fctn = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','ichol','weights',weights);
    tic();
    [phasor_GNI, his_phasor_GNI] = GaussNewtonProj(fctn, phase_recur(:),...
                                                 'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                 'solver','bispPhIchol','solverMaxIter',250,'solverTol',1e-1,...
                                                 'tolN',tolN,'iterSave',true);
    time_phasor_GNI(k) = toc();
    phasor_GNI  = phase_foldout(reshape(phasor_GNI,[256 256]), 0);
    imagor_GNI  = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phasor_GNI(:)),[256 256])))));
    imagor_GNI           = imagor_GNI/max(imagor_GNI(:));
    s                   = measureShift(obj,imagor_GNI);
    imagor_GNI           = shiftImage(imagor_GNI,s);
    sol_phasor_GNI(:,k)  = imagor_GNI(:);
    its_phasor_GNI(k)    = size(his_phasor_GNI.iters,2)-1;
    LS_phasor_GNI(k)     = sum(his_phasor_GNI.array(:,7));
    ROF_phasor_GNI(k)   = his_phasor_GNI.array(end,2)/his_phasor_GNI.array(1,2);
    clear GaussNewtonProj;
    clear phasorObjFctn;
    
end

%%
% Relative error and NCC

for k = 1:runs
    RE_phase_GD(k)    = norm((sol_phase_GD(:,k)) - obj(:))/norm(obj(:));   
    RE_phasor_GD(k)   = norm((sol_phasor_GD(:,k)) - obj(:))/norm(obj(:));   
    RE_phase_NLCG(k)  = norm((sol_phase_NLCG(:,k)) - obj(:))/norm(obj(:));  
    RE_phasor_NLCG(k) = norm((sol_phasor_NLCG(:,k)) - obj(:))/norm(obj(:));  
    RE_phase_BFGS(k)  = norm((sol_phase_BFGS(:,k)) - obj(:))/norm(obj(:));  
    RE_phasor_BFGS(k) = norm((sol_phasor_BFGS(:,k)) - obj(:))/norm(obj(:));  
    RE_phase_GNF(k)   = norm((sol_phase_GNF(:,k)) - obj(:))/norm(obj(:)); 
    RE_phasor_GNF(k)  = norm((sol_phasor_GNF(:,k)) - obj(:))/norm(obj(:));  
    RE_phase_GNT(k)   = norm((sol_phase_GNT(:,k)) - obj(:))/norm(obj(:));  
    RE_phasor_GNT(k)  = norm((sol_phasor_GNT(:,k)) - obj(:))/norm(obj(:));  
    RE_phase_GNI(k)   = norm((sol_phase_GNI(:,k)) - obj(:))/norm(obj(:));  
    RE_phasor_GNI(k)  = norm((sol_phasor_GNI(:,k)) - obj(:))/norm(obj(:));  
    
    NCC_phase_GD(k)    = 0.5 - 0.5*((sol_phase_GD(:,k)'*obj(:))^2/((sol_phase_GD(:,k)'*sol_phase_GD(:,k))*(obj(:)'*obj(:))));  
    NCC_phasor_GD(k)   = 0.5 - 0.5*((sol_phasor_GD(:,k)'*obj(:))^2/((sol_phasor_GD(:,k)'*sol_phasor_GD(:,k))*(obj(:)'*obj(:))));  
    NCC_phase_NLCG(k)  = 0.5 - 0.5*((sol_phase_NLCG(:,k)'*obj(:))^2/((sol_phase_NLCG(:,k)'*sol_phase_NLCG(:,k))*(obj(:)'*obj(:))));  
    NCC_phasor_NLCG(k) = 0.5 - 0.5*((sol_phasor_NLCG(:,k)'*obj(:))^2/((sol_phasor_NLCG(:,k)'*sol_phasor_NLCG(:,k))*(obj(:)'*obj(:))));  
    NCC_phase_BFGS(k)  = 0.5 - 0.5*((sol_phase_BFGS(:,k)'*obj(:))^2/((sol_phase_BFGS(:,k)'*sol_phase_BFGS(:,k))*(obj(:)'*obj(:))));  
    NCC_phasor_BFGS(k) = 0.5 - 0.5*((sol_phasor_BFGS(:,k)'*obj(:))^2/((sol_phasor_BFGS(:,k)'*sol_phasor_BFGS(:,k))*(obj(:)'*obj(:))));  
    NCC_phase_GNF(k)   = 0.5 - 0.5*((sol_phase_GNF(:,k)'*obj(:))^2/((sol_phase_GNF(:,k)'*sol_phase_GNF(:,k))*(obj(:)'*obj(:))));  
    NCC_phasor_GNF(k)  = 0.5 - 0.5*((sol_phasor_GNF(:,k)'*obj(:))^2/((sol_phasor_GNF(:,k)'*sol_phasor_GNF(:,k))*(obj(:)'*obj(:))));  
    NCC_phase_GNT(k)   = 0.5 - 0.5*((sol_phase_GNT(:,k)'*obj(:))^2/((sol_phase_GNT(:,k)'*sol_phase_GNT(:,k))*(obj(:)'*obj(:))));  
    NCC_phasor_GNT(k)  = 0.5 - 0.5*((sol_phasor_GNT(:,k)'*obj(:))^2/((sol_phasor_GNT(:,k)'*sol_phasor_GNT(:,k))*(obj(:)'*obj(:))));  
    NCC_phase_GNI(k)   = 0.5 - 0.5*((sol_phase_GNI(:,k)'*obj(:))^2/((sol_phase_GNI(:,k)'*sol_phase_GNI(:,k))*(obj(:)'*obj(:))));  
    NCC_phasor_GNI(k)  = 0.5 - 0.5*((sol_phasor_GNI(:,k)'*obj(:))^2/((sol_phasor_GNI(:,k)'*sol_phasor_GNI(:,k))*(obj(:)'*obj(:))));   
end

%%
% Print some results to the terminal window

fprintf('\n***** Relative Obj. Fctn. Minima *****\n');
fprintf('min(ROF_phase_GD)     = %1.4e \n', sum(ROF_phase_GD)/runs);
fprintf('min(ROF_phase_NLCG)   = %1.4e \n', sum(ROF_phase_NLCG)/runs);
fprintf('min(ROF_phase_LBFGS)  = %1.4e \n', sum(ROF_phase_BFGS)/runs);
fprintf('min(ROF_phase_GNF)    = %1.4e \n', sum(ROF_phase_GNF)/runs);
fprintf('min(ROF_phase_GNT)    = %1.4e \n', sum(ROF_phase_GNT)/runs);
fprintf('min(ROF_phase_GNI)    = %1.4e \n', sum(ROF_phase_GNI)/runs);

fprintf('\n***** Relative Error Minima *****\n');
fprintf('min(RE_phase_GD)     = %1.4e \n', sum(RE_phase_GD)/runs);
fprintf('min(RE_phase_NLCG)   = %1.4e \n', sum(RE_phase_NLCG)/runs);
fprintf('min(RE_phase_LBFGS)  = %1.4e \n', sum(RE_phase_BFGS)/runs);
fprintf('min(RE_phase_GNF)    = %1.4e \n', sum(RE_phase_GNF)/runs);
fprintf('min(RE_phase_GNT)    = %1.4e \n', sum(RE_phase_GNT)/runs);
fprintf('min(RE_phase_GNI)    = %1.4e \n', sum(RE_phase_GNI)/runs);

fprintf('\n***** Normalized Cross-Correlation Minima *****\n');
fprintf('min(NCC_phase_GD)     = %1.4e \n', sum(NCC_phase_GD)/runs);
fprintf('min(NCC_phase_NLCG)   = %1.4e \n', sum(NCC_phase_NLCG)/runs);
fprintf('min(NCC_phase_LBFGS)  = %1.4e \n', sum(NCC_phase_BFGS)/runs);
fprintf('min(NCC_phase_GNF)    = %1.4e \n', sum(NCC_phase_GNF)/runs);
fprintf('min(NCC_phase_GNT)    = %1.4e \n', sum(NCC_phase_GNT)/runs);
fprintf('min(NCC_phase_GNI)    = %1.4e \n', sum(NCC_phase_GNI)/runs);

fprintf('\n***** Outer Iterations til Convergence *****\n');
fprintf('iters(phase_GD)      = %.1f \n', sum(its_phase_GD)/runs);
fprintf('iters(phase_NLCG)    = %.1f \n', sum(its_phase_NLCG)/runs);
fprintf('iters(phase_LBFGS)   = %.1f \n', sum(its_phase_BFGS)/runs);
fprintf('iters(phase_GNF)     = %.1f \n', sum(its_phase_GNF)/runs);
fprintf('iters(phase_GNT)     = %.1f \n', sum(its_phase_GNT)/runs);
fprintf('iters(phase_GNI)     = %.1f \n', sum(its_phase_GNI)/runs);

fprintf('\n***** Total Time Elapsed *****\n');
fprintf('time(phase_GD)       = %1.4e \n', sum(time_phase_GD)/runs);
fprintf('time(phase_NLCG)     = %1.4e \n', sum(time_phase_NLCG)/runs);
fprintf('time(phase_LBFGS)    = %1.4e \n', sum(time_phase_BFGS)/runs);
fprintf('time(phase_GNF)      = %1.4e \n', sum(time_phase_GNF)/runs);
fprintf('time(phase_GNT)      = %1.4e \n', sum(time_phase_GNT)/runs);
fprintf('time(phase_GNI)      = %1.4e \n', sum(time_phase_GNI)/runs);

fprintf('\n***** Time per Iteration *****\n');
fprintf('time(phase_GD)/its     = %1.4e \n', sum(time_phase_GD)/sum(its_phase_GD));
fprintf('time(phase_NLCG)/its   = %1.4e \n', sum(time_phase_NLCG)/sum(its_phase_NLCG));
fprintf('time(phase_LBFGS)/its  = %1.4e \n', sum(time_phase_BFGS)/sum(its_phase_BFGS));
fprintf('time(phase_GNF)/its    = %1.4e \n', sum(time_phase_GNF)/sum(its_phase_GNF));
fprintf('time(phase_GNT)/its    = %1.4e \n', sum(time_phase_GNT)/sum(its_phase_GNT));
fprintf('time(phase_GNI)/its    = %1.4e \n', sum(time_phase_GNI)/sum(its_phase_GNI));

fprintf('\n***** Avg. Line Search Iterations per Outer Iteration *****\n');
fprintf('LS(phase_GD)/its     = %1.1f \n', sum(LS_phase_GD)/sum(its_phase_GD));
fprintf('LS(phase_NLCG)/its   = %1.1f \n', sum(LS_phase_NLCG)/sum(its_phase_NLCG));
fprintf('LS(phase_LBFGS)/its  = %1.1f \n', sum(LS_phase_BFGS)/sum(its_phase_BFGS));
fprintf('LS(phase_GNF)/its    = %1.1f \n', sum(LS_phase_GNF)/sum(its_phase_GNF));
fprintf('LS(phase_GNT)/its    = %1.1f \n', sum(LS_phase_GNT)/sum(its_phase_GNT));
fprintf('LS(phase_GNI)/its    = %1.1f \n', sum(LS_phase_GNI)/sum(its_phase_GNI));



fprintf('\n***** Relative Obj. Fctn. Minima *****\n');
fprintf('min(ROF_phasor_GD)    = %1.4e \n', sum(ROF_phasor_GD)/runs);
fprintf('min(ROF_phasor_NLCG)  = %1.4e \n', sum(ROF_phasor_NLCG)/runs);
fprintf('min(ROF_phasor_LBFGS) = %1.4e \n', sum(ROF_phasor_BFGS)/runs);
fprintf('min(ROF_phasor_GNF)   = %1.4e \n', sum(ROF_phasor_GNF)/runs);
fprintf('min(ROF_phasor_GNT)   = %1.4e \n', sum(ROF_phasor_GNT)/runs);
fprintf('min(ROF_phasor_GNI)   = %1.4e \n', sum(ROF_phasor_GNI)/runs);

fprintf('\n***** Relative Error Minima *****\n');
fprintf('min(RE_phasor_GD)    = %1.4e \n', sum(RE_phasor_GD)/runs);
fprintf('min(RE_phasor_NLCG)  = %1.4e \n', sum(RE_phasor_NLCG)/runs);
fprintf('min(RE_phasor_LBFGS) = %1.4e \n', sum(RE_phasor_BFGS)/runs);
fprintf('min(RE_phasor_GNF)   = %1.4e \n', sum(RE_phasor_GNF)/runs);
fprintf('min(RE_phasor_GNT)   = %1.4e \n', sum(RE_phasor_GNT)/runs);
fprintf('min(RE_phasor_GNI)   = %1.4e \n', sum(RE_phasor_GNI)/runs);

fprintf('\n***** Normalized Cross-Correlation Minima *****\n');
fprintf('min(NCC_phasor_GD)    = %1.4e \n', sum(NCC_phasor_GD)/runs);
fprintf('min(NCC_phasor_NLCG)  = %1.4e \n', sum(NCC_phasor_NLCG)/runs);
fprintf('min(NCC_phasor_LBFGS) = %1.4e \n', sum(NCC_phasor_BFGS)/runs);
fprintf('min(NCC_phasor_GNF)   = %1.4e \n', sum(NCC_phasor_GNF)/runs);
fprintf('min(NCC_phasor_GNT)   = %1.4e \n', sum(NCC_phasor_GNT)/runs);
fprintf('min(NCC_phasor_GNI)   = %1.4e \n', sum(NCC_phasor_GNI)/runs);

fprintf('\n***** Outer Iterations til Convergence *****\n');
fprintf('iters(phasor_GD)     = %.1f \n', sum(its_phasor_GD)/runs);
fprintf('iters(phasor_NLCG)   = %.1f \n', sum(its_phasor_NLCG)/runs);
fprintf('iters(phasor_LBFGS)  = %.1f \n', sum(its_phasor_BFGS)/runs);
fprintf('iters(phasor_GNF)    = %.1f \n', sum(its_phasor_GNF)/runs);
fprintf('iters(phasor_GNT)    = %.1f \n', sum(its_phasor_GNT)/runs);
fprintf('iters(phasor_GNI)    = %.1f \n', sum(its_phasor_GNI)/runs);

fprintf('\n***** Total Time Elapsed *****\n');
fprintf('time(phasor_GD)      = %1.4e \n', sum(time_phasor_GD)/runs);
fprintf('time(phasor_NLCG)    = %1.4e \n', sum(time_phasor_NLCG)/runs);
fprintf('time(phasor_LBFGS)   = %1.4e \n', sum(time_phasor_BFGS)/runs);
fprintf('time(phasor_GNF)     = %1.4e \n', sum(time_phasor_GNF)/runs);
fprintf('time(phasor_GNT)     = %1.4e \n', sum(time_phasor_GNT)/runs);
fprintf('time(phasor_GNI)     = %1.4e \n', sum(time_phasor_GNI)/runs);

fprintf('\n***** Time per Iteration *****\n');
fprintf('time(phasor_GD)/its    = %1.4e \n', sum(time_phasor_GD)/sum(its_phasor_GD));
fprintf('time(phasor_NLCG)/its  = %1.4e \n', sum(time_phasor_NLCG)/sum(its_phasor_NLCG));
fprintf('time(phasor_LBFGS)/its = %1.4e \n', sum(time_phasor_BFGS)/sum(its_phasor_BFGS));
fprintf('time(phasor_GNF)/its   = %1.4e \n', sum(time_phasor_GNF)/sum(its_phasor_GNF));
fprintf('time(phasor_GNT)/its   = %1.4e \n', sum(time_phasor_GNT)/sum(its_phasor_GNT));
fprintf('time(phasor_GNI)/its   = %1.4e \n', sum(time_phasor_GNI)/sum(its_phasor_GNI));

fprintf('\n***** Avg. Line Search Iterations per Outer Iteration *****\n');
fprintf('LS(phasor_GD)/its    = %1.1f \n', sum(LS_phasor_GD)/sum(its_phasor_GD));
fprintf('LS(phasor_NLCG)/its  = %1.1f \n', sum(LS_phasor_NLCG)/sum(its_phasor_NLCG));
fprintf('LS(phasor_LBFGS)/its = %1.1f \n', sum(LS_phasor_BFGS)/sum(its_phasor_BFGS));
fprintf('LS(phasor_GNF)/its   = %1.1f \n', sum(LS_phasor_GNF)/sum(its_phasor_GNF));
fprintf('LS(phasor_GNT)/its   = %1.1f \n', sum(LS_phasor_GNT)/sum(its_phasor_GNT));
fprintf('LS(phasor_GNI)/its   = %1.1f \n', sum(LS_phasor_GNI)/sum(its_phasor_GNI));


