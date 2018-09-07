%
%   This file tests optimization using projected Gauss-Newton with the
%   imphase and imphasor objective functions. 
%
clear all; close all;

% Setup Gauss-Newton parameters
n = 256^2;
upper_bound  = ones(n,1);
lower_bound  = zeros(n,1);
tolJ         = 1e-4;            
tolY         = 1e-3;           
tolG         = 1e1;
tolN         = 1e-3;
maxIter      = 100;
solverMaxIter= 250;              
solverTol    = 1e-1;
alphaPos     = 1e5;
alphaGrad    = 1e-2;

%%
% Set up loop of runs

runs = 50;

% Alot memory
sol_imphase_GD       = zeros(n,runs);
sol_imphasor_GD      = zeros(n,runs);
sol_imphase_NLCG     = zeros(n,runs);
sol_imphasor_NLCG    = zeros(n,runs);
sol_imphase_BFGS     = zeros(n,runs);
sol_imphasor_BFGS    = zeros(n,runs);
sol_imphase_GN       = zeros(n,runs);
sol_imphasor_GN      = zeros(n,runs);
sol_imphase_PGN      = zeros(n,runs);
sol_imphasor_PGN     = zeros(n,runs);
sol_imphase_PGNR     = zeros(n,runs);
sol_imphasor_PGNR    = zeros(n,runs);

ROF_imphase_GD       = zeros(runs,1);
ROF_imphasor_GD      = zeros(runs,1);
ROF_imphase_NLCG     = zeros(runs,1);
ROF_imphasor_NLCG    = zeros(runs,1);
ROF_imphase_BFGS     = zeros(runs,1);
ROF_imphasor_BFGS    = zeros(runs,1);
ROF_imphase_GN       = zeros(runs,1);
ROF_imphasor_GN      = zeros(runs,1);
ROF_imphase_PGN      = zeros(runs,1);
ROF_imphasor_PGN     = zeros(runs,1);
ROF_imphase_PGNR      = zeros(runs,1);
ROF_imphasor_PGNR     = zeros(runs,1);

RE_imphase_GD       = zeros(runs,1);
RE_imphasor_GD      = zeros(runs,1);
RE_imphase_NLCG     = zeros(runs,1);
RE_imphasor_NLCG    = zeros(runs,1);
RE_imphase_BFGS     = zeros(runs,1);
RE_imphasor_BFGS    = zeros(runs,1);
RE_imphase_GN       = zeros(runs,1);
RE_imphasor_GN      = zeros(runs,1);
RE_imphase_PGN      = zeros(runs,1);
RE_imphasor_PGN     = zeros(runs,1);
RE_imphase_PGNR     = zeros(runs,1);
RE_imphasor_PGNR    = zeros(runs,1);

NCC_imphase_GD       = zeros(runs,1);
NCC_imphasor_GD      = zeros(runs,1);
NCC_imphase_NLCG     = zeros(runs,1);
NCC_imphasor_NLCG    = zeros(runs,1);
NCC_imphase_BFGS     = zeros(runs,1);
NCC_imphasor_BFGS    = zeros(runs,1);
NCC_imphase_GN       = zeros(runs,1);
NCC_imphasor_GN      = zeros(runs,1);
NCC_imphase_PGN      = zeros(runs,1);
NCC_imphasor_PGN     = zeros(runs,1);
NCC_imphase_PGNR     = zeros(runs,1);
NCC_imphasor_PGNR    = zeros(runs,1);

time_imphase_GD       = zeros(runs,1);
time_imphasor_GD      = zeros(runs,1);
time_imphase_NLCG     = zeros(runs,1);
time_imphasor_NLCG    = zeros(runs,1);
time_imphase_BFGS     = zeros(runs,1);
time_imphasor_BFGS    = zeros(runs,1);
time_imphase_GN       = zeros(runs,1);
time_imphasor_GN      = zeros(runs,1);
time_imphase_PGN      = zeros(runs,1);
time_imphasor_PGN     = zeros(runs,1);
time_imphase_PGNR     = zeros(runs,1);
time_imphasor_PGNR    = zeros(runs,1);

its_imphase_GD       = zeros(runs,1);
its_imphasor_GD      = zeros(runs,1);
its_imphase_NLCG     = zeros(runs,1);
its_imphasor_NLCG    = zeros(runs,1);
its_imphase_BFGS     = zeros(runs,1);
its_imphasor_BFGS    = zeros(runs,1);
its_imphase_GN       = zeros(runs,1);
its_imphasor_GN      = zeros(runs,1);
its_imphase_PGN      = zeros(runs,1);
its_imphasor_PGN     = zeros(runs,1);
its_imphase_PGNR     = zeros(runs,1);
its_imphasor_PGNR    = zeros(runs,1);

LS_imphase_GD       = zeros(runs,1);
LS_imphasor_GD      = zeros(runs,1);
LS_imphase_NLCG     = zeros(runs,1);
LS_imphasor_NLCG    = zeros(runs,1);
LS_imphase_BFGS     = zeros(runs,1);
LS_imphasor_BFGS    = zeros(runs,1);
LS_imphase_GN       = zeros(runs,1);
LS_imphasor_GN      = zeros(runs,1);
LS_imphase_PGN      = zeros(runs,1);
LS_imphasor_PGN     = zeros(runs,1);
LS_imphase_PGNR     = zeros(runs,1);
LS_imphasor_PGNR    = zeros(runs,1);

for k = 1:runs
    
    % Setup data at each run
    path_SpeckleImagingCodes;
    [nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',100,'D_r0',30);
    setupBispectrumData;
    image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(1i*phase_recur(:)),[256 256])))));
    dims = size(image_recur);
    image_proj  = gdnnf_projection(image_recur, sum(image_recur(:))) + 1e-4;
    avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));
    obj = obj/max(obj(:));

    % Run gradient descent for imphase
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphase_GD, his_imphase_GD] = GradientDescentProj(fctn, image_recur(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'iterSave',true);
    time_imphase_GD(k)  = toc();
    imphase_GD          = reshape(imphase_GD,[256 256])/max(imphase_GD(:));
    s                   = measureShift(obj,imphase_GD);
    imphase_GD          = shiftImage(imphase_GD,s);
    sol_imphase_GD(:,k) = imphase_GD(:);
    its_imphase_GD(k)   = size(his_imphase_GD.iters,2)-1;
    LS_imphase_GD(k)    = sum(his_imphase_GD.array(:,6));
    ROF_imphase_GD(k)   = his_imphase_GD.array(end,2)/his_imphase_GD.array(1,2);
    clear GradientDescentProj;
    clear imphaseObjFctn;

    % Run gradient descent for imphasor
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphasor_GD, his_imphasor_GD] = GradientDescentProj(fctn, image_recur(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'iterSave',true);                        
    time_imphasor_GD(k)  = toc();
    imphasor_GD          = reshape(imphasor_GD, [256 256])/max(imphasor_GD(:));;
    s                    = measureShift(obj,imphasor_GD);
    imphasor_GD          = shiftImage(imphasor_GD,s);
    sol_imphasor_GD(:,k) = imphasor_GD(:);
    its_imphasor_GD(k)   = size(his_imphasor_GD.iters,2)-1;
    LS_imphasor_GD(k)    = sum(his_imphasor_GD.array(:,6));
    ROF_imphasor_GD(k)   = his_imphasor_GD.array(end,2)/his_imphasor_GD.array(1,2);
    clear GradientDescentProj;
    clear imphasorObjFctn;


    % Run NLCG for imphase
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphase_NLCG, his_imphase_NLCG] = NonlinearCG(fctn, image_recur(:), 'maxIter',maxIter,...
                                            'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                            'iterSave',true);
    time_imphase_NLCG(k)  = toc();
    imphase_NLCG          = reshape(imphase_NLCG,[256 256])/max(imphase_NLCG(:));
    s                     = measureShift(obj,imphase_NLCG);
    imphase_NLCG          = shiftImage(imphase_NLCG,s);
    sol_imphase_NLCG(:,k) = imphase_NLCG(:);
    its_imphase_NLCG(k)   = size(his_imphase_NLCG.iters,2)-1;
    LS_imphase_NLCG(k)    = sum(his_imphase_NLCG.array(:,5));
    ROF_imphase_NLCG(k)   = his_imphase_NLCG.array(end,2)/his_imphase_NLCG.array(1,2);
    clear NonlinearCG;
    clear imphaseObjFctn;

    % Run NLCG for imphasor
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphasor_NLCG, his_imphasor_NLCG] = NonlinearCG(fctn, image_recur(:),'maxIter',maxIter,...
                                              'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                              'iterSave',true);
    time_imphasor_NLCG(k)  = toc();
    imphasor_NLCG          = reshape(imphasor_NLCG, [256 256])/max(imphasor_NLCG(:));
    s                      = measureShift(obj,imphasor_NLCG);
    imphasor_NLCG          = shiftImage(imphasor_NLCG,s);
    sol_imphasor_NLCG(:,k) = imphasor_NLCG(:);
    its_imphasor_NLCG(k)   = size(his_imphasor_NLCG.iters,2)-1;
    LS_imphasor_NLCG(k)    = sum(his_imphasor_NLCG.array(:,5));
    ROF_imphasor_NLCG(k)   = his_imphasor_NLCG.array(end,2)/his_imphasor_NLCG.array(1,2);
    clear NonlinearCG;
    clear imphasorObjFctn;


    % Run LBFGS for imphase
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphase_BFGS, his_imphase_BFGS] = LBFGS(fctn, image_recur(:), 'maxIter',maxIter,...
                                            'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                            'iterSave',true);
    time_imphase_BFGS(k)  = toc();
    imphase_BFGS          = reshape(imphase_BFGS,[256 256])/max(imphase_BFGS(:));
    s                     = measureShift(obj,imphase_BFGS);
    imphase_BFGS          = shiftImage(imphase_BFGS,s);
    sol_imphase_BFGS(:,k) = imphase_BFGS(:);
    its_imphase_BFGS(k)   = size(his_imphase_BFGS.iters,2)-1;
    LS_imphase_BFGS(k)    = sum(his_imphase_BFGS.array(:,5));
    ROF_imphase_BFGS(k)   = his_imphase_BFGS.array(end,2)/his_imphase_BFGS.array(1,2);
    clear LBFGS;
    clear imphaseObjFctn;

    % Run LBFGS for imphasor
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphasor_BFGS, his_imphasor_BFGS] = LBFGS(fctn, image_recur(:),'maxIter',maxIter,...
                                              'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                              'iterSave',true);
    time_imphasor_BFGS(k)  = toc();
    imphasor_BFGS          = reshape(imphasor_BFGS,[256 256])/max(imphasor_BFGS(:));
    s                      = measureShift(obj,imphasor_BFGS);
    imphasor_BFGS          = shiftImage(imphasor_BFGS,s);
    sol_imphasor_BFGS(:,k) = imphasor_BFGS(:);
    its_imphasor_BFGS(k)   = size(his_imphasor_BFGS.iters,2)-1;
    LS_imphasor_BFGS(k)    = sum(his_imphasor_BFGS.array(:,5));
    ROF_imphasor_BFGS(k)   = his_imphasor_BFGS.array(end,2)/his_imphasor_BFGS.array(1,2);
    clear LBFGS;
    clear imphasorObjFctn;


    % Run Gauss-Newton for imphase
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphase_GN, his_imphase_GN] = GaussNewtonProj(fctn, image_recur(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                    'tolN',tolN,'iterSave',true);
    time_imphase_GN(k)  = toc();
    imphase_GN          = reshape(imphase_GN,[256 256])/max(imphase_GN(:));
    s                   = measureShift(obj,imphase_GN);
    imphase_GN          = shiftImage(imphase_GN,s);
    sol_imphase_GN(:,k) = imphase_GN(:);
    its_imphase_GN(k)   = size(his_imphase_GN.iters,2)-1;
    LS_imphase_GN(k)    = sum(his_imphase_GN.array(:,7));
    ROF_imphase_GN(k)   = his_imphase_GN.array(end,2)/his_imphase_GN.array(1,2);
    clear GaussNewtonProj;
    clear imphaseObjFctn;

    % Run Gauss-Newton for imphasor
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase,dims, pupil_mask,'alpha',alphaPos,'regularizer','pos','weights',weights);
    tic();
    [imphasor_GN, his_imphasor_GN] = GaussNewtonProj(fctn, image_recur(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                    'tolN',tolN,'iterSave',true);
    time_imphasor_GN(k)  = toc();
    imphasor_GN          = reshape(imphasor_GN,[256 256])/max(imphasor_GN(:));
    s                    = measureShift(obj,imphasor_GN);
    imphasor_GN          = shiftImage(imphasor_GN,s);
    sol_imphasor_GN(:,k) = imphasor_GN(:);
    its_imphasor_GN(k)   = size(his_imphasor_GN.iters,2)-1;
    LS_imphasor_GN(k)    = sum(his_imphasor_GN.array(:,7));
    ROF_imphasor_GN(k)   = his_imphasor_GN.array(end,2)/his_imphasor_GN.array(1,2);
    clear GaussNewtonProj;
    clear imphasorObjFctn;


    % Run projected Gauss-Newton for imphase
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', 0.0,'regularizer','pos','weights',weights);
    tic();
    [imphase_PGN, his_imphase_PGN] = GaussNewtonProj(fctn, image_proj(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                    'tolN',tolN,'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    time_imphase_PGN(k)  = toc();
    imphase_PGN          = reshape(imphase_PGN,[256 256])/max(imphase_PGN(:));
    s                    = measureShift(obj,imphase_PGN);
    imphase_PGN          = shiftImage(imphase_PGN,s);
    sol_imphase_PGN(:,k) = imphase_PGN(:);
    its_imphase_PGN(k)   = size(his_imphase_PGN.iters,2)-1;
    LS_imphase_PGN(k)    = sum(his_imphase_PGN.array(:,7));
    ROF_imphase_PGN(k)   = his_imphase_PGN.array(end,2)/his_imphase_PGN.array(1,2);
    clear GaussNewtonProj;
    clear imphaseObjFctn;

    % Run projected Gauss-Newton for imphasor
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', 0.0,'regularizer','pos','weights',weights);
    tic();
    [imphasor_PGN, his_imphasor_PGN] = GaussNewtonProj(fctn, image_proj(:),...
                                                      'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                      'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                      'tolN',tolN,'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    time_imphasor_PGN(k)  = toc();
    imphasor_PGN          = reshape(imphasor_PGN,[256 256])/max(imphasor_PGN(:));
    s                     = measureShift(obj,imphasor_PGN);
    imphasor_PGN          = shiftImage(imphasor_PGN,s);
    sol_imphasor_PGN(:,k) = imphasor_PGN(:);
    its_imphasor_PGN(k)   = size(his_imphasor_PGN.iters,2)-1;
    LS_imphasor_PGN(k)    = sum(his_imphasor_PGN.array(:,7));
    ROF_imphasor_PGN(k)   = his_imphasor_PGN.array(end,2)/his_imphasor_PGN.array(1,2);
    clear GaussNewtonProj;
    clear imphasorObjFctn;

    % Run projected Gauss-Newton for imphase
    fctn = @(x) imphaseObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaGrad,'regularizer','grad','weights',weights);
    tic();
    [imphase_PGNR, his_imphase_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                    'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                    'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                    'tolN',tolN,'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    time_imphase_PGNR(k)  = toc();
    imphase_PGNR          = reshape(imphase_PGNR,[256 256])/max(imphase_PGNR(:));
    s                     = measureShift(obj,imphase_PGNR);
    imphase_PGNR          = shiftImage(imphase_PGNR,s);
    sol_imphase_PGNR(:,k) = imphase_PGNR(:);
    its_imphase_PGNR(k)   = size(his_imphase_PGNR.iters,2)-1;
    LS_imphase_PGNR(k)    = sum(his_imphase_PGNR.array(:,7)); 
    ROF_imphase_PGNR(k)   = his_imphase_PGNR.array(end,2)/his_imphase_PGNR.array(1,2);
    clear GaussNewtonProj;
    clear imphaseObjFctn;

    % Run projected Gauss-Newton for imphasor
    fctn = @(x) imphasorObjFctn(x,A, bispec_phase, dims, pupil_mask,'alpha', alphaGrad,'regularizer','grad','weights',weights);
    tic();
    [imphasor_PGNR, his_imphasor_PGNR] = GaussNewtonProj(fctn, image_proj(:),...
                                                      'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                      'solver','bispIm','solverMaxIter',250,'solverTol',1e-1,...
                                                      'tolN',tolN,'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
    time_imphasor_PGNR(k)   = toc();
    imphasor_PGNR          = reshape(imphasor_PGNR,[256 256])/max(imphasor_PGNR(:));
    s                      = measureShift(obj,imphasor_PGNR);
    imphasor_PGNR          = shiftImage(imphasor_PGNR,s);
    sol_imphasor_PGNR(:,k) = imphasor_PGNR(:);
    its_imphasor_PGNR(k)   = size(his_imphasor_PGNR.iters,2)-1;
    LS_imphasor_PGNR(k)    = sum(his_imphasor_PGNR.array(:,7));
    ROF_imphasor_PGNR(k)   = his_imphasor_PGNR.array(end,2)/his_imphasor_PGNR.array(1,2);
    clear GaussNewtonProj;
    clear imphasorObjFctn;
end
    
%%
% Relative error and NCC

for k = 1:runs
    RE_imphase_GD(k)    = norm((sol_imphase_GD(:,k)) - obj(:))/norm(obj(:));   
    RE_imphasor_GD(k)   = norm((sol_imphasor_GD(:,k)) - obj(:))/norm(obj(:));   
    RE_imphase_NLCG(k)  = norm((sol_imphase_NLCG(:,k)) - obj(:))/norm(obj(:));  
    RE_imphasor_NLCG(k) = norm((sol_imphasor_NLCG(:,k)) - obj(:))/norm(obj(:));  
    RE_imphase_BFGS(k)  = norm((sol_imphase_BFGS(:,k)) - obj(:))/norm(obj(:));  
    RE_imphasor_BFGS(k) = norm((sol_imphasor_BFGS(:,k)) - obj(:))/norm(obj(:));  
    RE_imphase_GN(k)    = norm((sol_imphase_GN(:,k)) - obj(:))/norm(obj(:)); 
    RE_imphasor_GN(k)   = norm((sol_imphasor_GN(:,k)) - obj(:))/norm(obj(:));  
    RE_imphase_PGN(k)   = norm((sol_imphase_PGN(:,k)) - obj(:))/norm(obj(:));  
    RE_imphasor_PGN(k)  = norm((sol_imphasor_PGN(:,k)) - obj(:))/norm(obj(:));  
    RE_imphase_PGNR(k)  = norm((sol_imphase_PGNR(:,k)) - obj(:))/norm(obj(:));  
    RE_imphasor_PGNR(k) = norm((sol_imphasor_PGNR(:,k)) - obj(:))/norm(obj(:));  
    
    NCC_imphase_GD(k) = 0.5 - 0.5*((sol_imphase_GD(:,k)'*obj(:))^2/((sol_imphase_GD(:,k)'*sol_imphase_GD(:,k))*(obj(:)'*obj(:))));  
    NCC_imphasor_GD(k) = 0.5 - 0.5*((sol_imphasor_GD(:,k)'*obj(:))^2/((sol_imphasor_GD(:,k)'*sol_imphasor_GD(:,k))*(obj(:)'*obj(:))));  
    NCC_imphase_NLCG(k) = 0.5 - 0.5*((sol_imphase_NLCG(:,k)'*obj(:))^2/((sol_imphase_NLCG(:,k)'*sol_imphase_NLCG(:,k))*(obj(:)'*obj(:))));  
    NCC_imphasor_NLCG(k) = 0.5 - 0.5*((sol_imphasor_NLCG(:,k)'*obj(:))^2/((sol_imphasor_NLCG(:,k)'*sol_imphasor_NLCG(:,k))*(obj(:)'*obj(:))));  
    NCC_imphase_BFGS(k) = 0.5 - 0.5*((sol_imphase_BFGS(:,k)'*obj(:))^2/((sol_imphase_BFGS(:,k)'*sol_imphase_BFGS(:,k))*(obj(:)'*obj(:))));  
    NCC_imphasor_BFGS(k) = 0.5 - 0.5*((sol_imphasor_BFGS(:,k)'*obj(:))^2/((sol_imphasor_BFGS(:,k)'*sol_imphasor_BFGS(:,k))*(obj(:)'*obj(:))));  
    NCC_imphase_GN(k) = 0.5 - 0.5*((sol_imphase_GN(:,k)'*obj(:))^2/((sol_imphase_GN(:,k)'*sol_imphase_GN(:,k))*(obj(:)'*obj(:))));  
    NCC_imphasor_GN(k) = 0.5 - 0.5*((sol_imphasor_GN(:,k)'*obj(:))^2/((sol_imphasor_GN(:,k)'*sol_imphasor_GN(:,k))*(obj(:)'*obj(:))));  
    NCC_imphase_PGN(k) = 0.5 - 0.5*((sol_imphase_PGN(:,k)'*obj(:))^2/((sol_imphase_PGN(:,k)'*sol_imphase_PGN(:,k))*(obj(:)'*obj(:))));  
    NCC_imphasor_PGN(k) = 0.5 - 0.5*((sol_imphasor_PGN(:,k)'*obj(:))^2/((sol_imphasor_PGN(:,k)'*sol_imphasor_PGN(:,k))*(obj(:)'*obj(:))));  
    NCC_imphase_PGNR(k) = 0.5 - 0.5*((sol_imphase_PGNR(:,k)'*obj(:))^2/((sol_imphase_PGNR(:,k)'*sol_imphase_PGNR(:,k))*(obj(:)'*obj(:))));  
    NCC_imphasor_PGNR(k) = 0.5 - 0.5*((sol_imphasor_PGNR(:,k)'*obj(:))^2/((sol_imphasor_PGNR(:,k)'*sol_imphasor_PGNR(:,k))*(obj(:)'*obj(:))));   
end

%%
% Print some results to the terminal window

fprintf('\n***** Relative Obj. Fctn. Minima *****\n');
fprintf('min(ROF_imphase_GD)     = %1.4e \n', sum(ROF_imphase_GD)/runs);
fprintf('min(ROF_imphasor_GD)    = %1.4e \n', sum(ROF_imphasor_GD)/runs);
fprintf('min(ROF_imphase_NLCG)   = %1.4e \n', sum(ROF_imphase_NLCG)/runs);
fprintf('min(ROF_imphasor_NLCG)  = %1.4e \n', sum(ROF_imphasor_NLCG)/runs);
fprintf('min(ROF_imphase_LBFGS)  = %1.4e \n', sum(ROF_imphase_BFGS)/runs);
fprintf('min(ROF_imphasor_LBFGS) = %1.4e \n', sum(ROF_imphasor_BFGS)/runs);
fprintf('min(ROF_imphase_GN)     = %1.4e \n', sum(ROF_imphase_GN)/runs);
fprintf('min(ROF_imphasor_GN)    = %1.4e \n', sum(ROF_imphasor_GN)/runs);
fprintf('min(ROF_imphase_PGN)    = %1.4e \n', sum(ROF_imphase_PGN)/runs);
fprintf('min(ROF_imphasor_PGN)   = %1.4e \n', sum(ROF_imphasor_PGN)/runs);
fprintf('min(ROF_imphase_PGNR)   = %1.4e \n', sum(ROF_imphase_PGNR)/runs);
fprintf('min(ROF_imphasor_PGNR)  = %1.4e \n', sum(ROF_imphasor_PGNR)/runs);

fprintf('\n***** Relative Error Minima *****\n');
fprintf('min(RE_imphase_GD)     = %1.4e \n', sum(RE_imphase_GD)/runs);
fprintf('min(RE_imphasor_GD)    = %1.4e \n', sum(RE_imphasor_GD)/runs);
fprintf('min(RE_imphase_NLCG)   = %1.4e \n', sum(RE_imphase_NLCG)/runs);
fprintf('min(RE_imphasor_NLCG)  = %1.4e \n', sum(RE_imphasor_NLCG)/runs);
fprintf('min(RE_imphase_LBFGS)  = %1.4e \n', sum(RE_imphase_BFGS)/runs);
fprintf('min(RE_imphasor_LBFGS) = %1.4e \n', sum(RE_imphasor_BFGS)/runs);
fprintf('min(RE_imphase_GN)     = %1.4e \n', sum(RE_imphase_GN)/runs);
fprintf('min(RE_imphasor_GN)    = %1.4e \n', sum(RE_imphasor_GN)/runs);
fprintf('min(RE_imphase_PGN)    = %1.4e \n', sum(RE_imphase_PGN)/runs);
fprintf('min(RE_imphasor_PGN)   = %1.4e \n', sum(RE_imphasor_PGN)/runs);
fprintf('min(RE_imphase_PGNR)   = %1.4e \n', sum(RE_imphase_PGNR)/runs);
fprintf('min(RE_imphasor_PGNR)  = %1.4e \n', sum(RE_imphasor_PGNR)/runs);

fprintf('\n***** Normalized Cross-Correlation Minima *****\n');
fprintf('min(NCC_imphase_GD)    = %1.4e \n', sum(NCC_imphase_GD)/runs);
fprintf('min(NCC_imphasor_GD)   = %1.4e \n', sum(NCC_imphasor_GD)/runs);
fprintf('min(NCC_imphase_NLCG)  = %1.4e \n', sum(NCC_imphase_NLCG)/runs);
fprintf('min(NCC_imphasor_NLCG) = %1.4e \n', sum(NCC_imphasor_NLCG)/runs);
fprintf('min(NCC_imphase_LBFGS) = %1.4e \n', sum(NCC_imphase_BFGS)/runs);
fprintf('min(NCC_imphasor_LBFGS)= %1.4e \n', sum(NCC_imphasor_BFGS)/runs);
fprintf('min(NCC_imphase_GN)    = %1.4e \n', sum(NCC_imphase_GN)/runs);
fprintf('min(NCC_imphasor_GN)   = %1.4e \n', sum(NCC_imphasor_GN)/runs);
fprintf('min(NCC_imphase_PGN)   = %1.4e \n', sum(NCC_imphase_PGN)/runs);
fprintf('min(NCC_imphasor_PGN)  = %1.4e \n', sum(NCC_imphasor_PGN)/runs);
fprintf('min(NCC_imphase_PGNR)  = %1.4e \n', sum(NCC_imphase_PGNR)/runs);
fprintf('min(NCC_imphasor_PGNR) = %1.4e \n', sum(NCC_imphasor_PGNR)/runs);

fprintf('\n***** Total Time Elapsed *****\n');
fprintf('time(imphase_GD)       = %1.4e \n', sum(time_imphase_GD)/runs);
fprintf('time(imphasor_GD)      = %1.4e \n', sum(time_imphasor_GD)/runs);
fprintf('time(imphase_NLCG)     = %1.4e \n', sum(time_imphase_NLCG)/runs);
fprintf('time(imphasor_NLCG)    = %1.4e \n', sum(time_imphasor_NLCG)/runs);
fprintf('time(imphase_LBFGS)    = %1.4e \n', sum(time_imphase_BFGS)/runs);
fprintf('time(imphasor_LBFGS)   = %1.4e \n', sum(time_imphasor_BFGS)/runs);
fprintf('time(imphase_GN)       = %1.4e \n', sum(time_imphase_GN)/runs);
fprintf('time(imphasor_GN)      = %1.4e \n', sum(time_imphasor_GN)/runs);
fprintf('time(imphase_PGN)      = %1.4e \n', sum(time_imphase_PGN)/runs);
fprintf('time(imphasor_PGN)     = %1.4e \n', sum(time_imphasor_PGN)/runs);
fprintf('time(imphase_PGNR)     = %1.4e \n', sum(time_imphase_PGNR)/runs);
fprintf('time(imphasor_PGNR)    = %1.4e \n', sum(time_imphasor_PGNR)/runs);

fprintf('\n***** Time per Iteration *****\n');
fprintf('time(imphase_GD)/its    = %1.4e \n', sum(time_imphase_GD)/sum(its_imphase_GD));
fprintf('time(imphasor_GD)/its   = %1.4e \n', sum(time_imphasor_GD)/sum(its_imphasor_GD));
fprintf('time(imphase_NLCG)/its  = %1.4e \n', sum(time_imphase_NLCG)/sum(its_imphase_NLCG));
fprintf('time(imphasor_NLCG)/its = %1.4e \n', sum(time_imphasor_NLCG)/sum(its_imphasor_NLCG));
fprintf('time(imphase_LBFGS)/its = %1.4e \n', sum(time_imphase_BFGS)/sum(its_imphase_BFGS));
fprintf('time(imphasor_LBFGS)/its= %1.4e \n', sum(time_imphasor_BFGS)/sum(its_imphasor_BFGS));
fprintf('time(imphase_GN)/its    = %1.4e \n', sum(time_imphase_GN)/sum(its_imphase_GN));
fprintf('time(imphasor_GN)/its   = %1.4e \n', sum(time_imphasor_GN)/sum(its_imphasor_GN));
fprintf('time(imphase_PGN)/its   = %1.4e \n', sum(time_imphase_PGN)/sum(its_imphase_PGN));
fprintf('time(imphasor_PGN)/its  = %1.4e \n', sum(time_imphasor_PGN)/sum(its_imphasor_PGN));
fprintf('time(imphase_PGNR)/its  = %1.4e \n', sum(time_imphase_PGNR)/sum(its_imphase_PGNR));
fprintf('time(imphasor_PGNR)/its = %1.4e \n', sum(time_imphasor_PGNR)/sum(its_imphasor_PGNR));

fprintf('\n***** Outer Iterations til Convergence *****\n');
fprintf('iters(imphase_GD)      = %.1f \n', sum(its_imphase_GD)/runs);
fprintf('iters(imphasor_GD)     = %.1f \n', sum(its_imphasor_GD)/runs);
fprintf('iters(imphase_NLCG)    = %.1f \n', sum(its_imphase_NLCG)/runs);
fprintf('iters(imphasor_NLCG)   = %.1f \n', sum(its_imphasor_NLCG)/runs);
fprintf('iters(imphase_LBFGS)   = %.1f \n', sum(its_imphase_BFGS)/runs);
fprintf('iters(imphasor_LBFGS)  = %.1f \n', sum(its_imphasor_BFGS)/runs);
fprintf('iters(imphase_GN)      = %.1f \n', sum(its_imphase_GN)/runs);
fprintf('iters(imphasor_GN)     = %.1f \n', sum(its_imphasor_GN)/runs);
fprintf('iters(imphase_PGN)     = %.1f \n', sum(its_imphase_PGN)/runs);
fprintf('iters(imphasor_PGN)    = %.1f \n', sum(its_imphasor_PGN)/runs);
fprintf('iters(imphase_PGNR)    = %.1f \n', sum(its_imphase_PGNR)/runs);
fprintf('iters(imphasor_PGNR)   = %.1f \n', sum(its_imphasor_PGNR)/runs);

fprintf('\n***** Avg. Line Search Iterations per Outer Iteration *****\n');
fprintf('LS(imphase_GD)/its     = %1.1f \n', sum(LS_imphase_GD)/sum(its_imphase_GD));
fprintf('LS(imphasor_GD)/its    = %1.1f \n', sum(LS_imphasor_GD)/sum(its_imphasor_GD));
fprintf('LS(imphase_NLCG)/its   = %1.1f \n', sum(LS_imphase_NLCG)/sum(its_imphase_NLCG));
fprintf('LS(imphasor_NLCG)/its  = %1.1f \n', sum(LS_imphasor_NLCG)/sum(its_imphasor_NLCG));
fprintf('LS(imphase_LBFGS)/its  = %1.1f \n', sum(LS_imphase_BFGS)/sum(its_imphase_BFGS));
fprintf('LS(imphasor_LBFGS)/its = %1.1f \n', sum(LS_imphasor_BFGS)/sum(its_imphasor_BFGS));
fprintf('LS(imphase_GN)/its     = %1.1f \n', sum(LS_imphase_GN)/sum(its_imphase_GN));
fprintf('LS(imphasor_GN)/its    = %1.1f \n', sum(LS_imphasor_GN)/sum(its_imphasor_GN));
fprintf('LS(imphase_PGN)/its    = %1.1f \n', sum(LS_imphase_PGN)/sum(its_imphase_PGN));
fprintf('LS(imphasor_PGN)/its   = %1.1f \n', sum(LS_imphasor_PGN)/sum(its_imphasor_PGN));
fprintf('LS(imphase_PGNR)/its   = %1.1f \n', sum(LS_imphase_PGNR)/sum(its_imphase_PGNR));
fprintf('LS(imphasor_PGNR)/its  = %1.1f \n', sum(LS_imphasor_PGNR)/sum(its_imphasor_PGNR));
