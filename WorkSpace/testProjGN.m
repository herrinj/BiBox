%
%   This file tests optimization using projected Gauss-Newton with the
%   imphase and imphasor objective functions. 
%

% Setup data
path_SpeckleImagingCodes;
[nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50,'D_r0',30);
setupBispectrumData;
image_recur = real(fftshift(ifft2(fftshift(reshape(pospec(:).*exp(i*phase_recur(:)),[256 256])))));
image_proj  = gdnnf_projection(image_recur, sum(image_recur(:))) + 1e-4;
avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));

% Setup Gauss-Newton parameters
upper_bound = ones(numel(image_proj),1);
lower_bound = zeros(numel(image_proj),1);
tolJ         = 1e-5;            
tolY         = 1e-5;           
tolG         = 1e-5;
maxIter      = 50;
solverMaxIter= 250;              
solverTol    = 1e-1;

%%
% Run damped Newton for imphase
ADA = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
obj_func = @(image) imphase_rec2(image,A,bispec_phase,weights, pupil_mask, ADA, 100.0,'pos',pospec);
tic();
[imphase_GN, his_imphase_GN] = GaussNewtonProj(obj_func, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'solver','bispectrum','solverMaxIter',250,'solverTol',1e-1,...
                                                'iterSave',true);
time_imphase_GN = toc();
imphase_GN = reshape(imphase_GN,[256 256]);

% Run damped Newton for imphasor
obj_func = @(image) imphasor_rec2(image,A,bispec_phase, weights, ADA, pupil_mask, 100.0,'pos',pospec);
tic();
[imphasor_GN, his_imphasor_GN] = GaussNewtonProj(obj_func, image_recur(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'solver','bispectrum','solverMaxIter',250,'solverTol',1e-1,...
                                                'iterSave',true);
time_imphasor_GN = toc();
imphasor_GN = reshape(imphasor_GN, [256 256]);

%%
% Run projected Gauss-Newton for imphase
ADA = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
obj_func = @(image) imphase_rec2(image,A,bispec_phase,weights, pupil_mask, ADA, 0.0,'pos',pospec);
tic();
[imphase_PGN, his_imphase_PGN] = GaussNewtonProj(obj_func, image_proj(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'solver','bispectrum','solverMaxIter',250,'solverTol',1e-1,...
                                                'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphase_PGN = toc();
imphase_PGN = reshape(imphase_PGN,[256 256]);

% Run projected Gauss-Newton for imphasor
obj_func = @(image) imphasor_rec2(image,A,bispec_phase,weights, ADA, pupil_mask, 0.0,'pos',pospec);
tic();
[imphasor_PGN, his_imphasor_PGN] = GaussNewtonProj(obj_func, image_proj(:),...
                                                  'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                  'solver','bispectrum','solverMaxIter',250,'solverTol',1e-1,...
                                                  'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphasor_PGN = toc();
imphasor_PGN = reshape(imphasor_PGN,[256 256]);

%%
% Run projected Gauss-Newton for imphase
ADA = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
obj_func = @(image) imphase_rec2(image,A,bispec_phase,weights, pupil_mask, ADA, 0.000001,'pow',pospec);
tic();
[imphase_PGNR, his_imphase_PGNR] = GaussNewtonProj(obj_func, image_proj(:),...
                                                'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                'solver','bispectrum','solverMaxIter',250,'solverTol',1e-1,...
                                                'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphase_PGNR = toc();
imphase_PGNR = reshape(imphase_PGNR,[256 256]);

% Run projected Gauss-Newton for imphasor
obj_func = @(image) imphasor_rec2(image,A,bispec_phase,weights, ADA, pupil_mask, 0.000001,'pow',pospec);
tic();
[imphasor_PGNR, his_imphasor_PGNR] = GaussNewtonProj(obj_func, image_proj(:),...
                                                  'maxIter',maxIter, 'tolJ', tolJ, 'tolY',tolY,'tolG',tolG,...
                                                  'solver','bispectrum','solverMaxIter',250,'solverTol',1e-1,...
                                                  'upper_bound',upper_bound,'lower_bound',lower_bound,'iterSave',true);
time_imphasor_PGNR = toc();
imphasor_PGNR = reshape(imphasor_PGNR,[256 256]);

%%
% Look at some results
obj             = obj/max(obj(:));
image_recur     = image_recur/max(image_recur(:));
image_proj      = image_proj/max(image_proj(:));
avg_data_frame  = avg_data_frame/max(avg_data_frame(:));
imphase_GN      = imphase_GN/max(imphase_GN(:));
imphase_PGN     = imphase_PGN/max(imphase_PGN(:));
imphase_PGNR     = imphase_PGNR/max(imphase_PGNR(:));
imphasor_GN     = imphasor_GN/max(imphasor_GN(:));
imphasor_PGN    = imphasor_PGN/max(imphasor_PGN(:));
imphasor_PGNR    = imphasor_PGNR/max(imphasor_PGNR(:));

figure; 
subplot(3,4,1); imagesc(reshape(obj,[256 256])); axis image; axis off; A = colorbar; title('truth'); 
subplot(3,4,2); imagesc(reshape(image_recur,[256 256])); axis image; axis off; colorbar; title('recur');
subplot(3,4,3); imagesc(reshape(image_proj, [256 256])); axis image; axis off; colorbar;  title('proj. recur');
subplot(3,4,4); imagesc(avg_data_frame); axis image; axis off; colorbar;  title('avg. blurred frame');

subplot(3,4,5); imagesc(imphase_GN); axis image; axis off; colorbar;  title('Imphase - DN');
subplot(3,4,6); imagesc(imphase_PGN); axis image; axis off; colorbar;  title('Imphase - PGN');
subplot(3,4,7); imagesc(imphasor_GN); axis image; axis off; colorbar;  title('Imphasor - DN');
subplot(3,4,8); imagesc(imphasor_PGN); axis image; axis off; colorbar;  title('Imphasor - PGN');

subplot(3,4,10); imagesc(imphase_PGNR); axis image; axis off; colorbar;  title('Imphase - PGNR');
subplot(3,4,12); imagesc(imphasor_PGNR); axis image; axis off; colorbar;  title('Imphasor - PGNR');

%%
% Relative objective function

figure();
plot((0:size(his_imphase_GN.array,1)-1)',his_imphase_GN.array(:,2)/his_imphase_GN.array(1,2),'ro-'); 
hold on;
plot((0:size(his_imphasor_GN.array,1)-1)',his_imphasor_GN.array(:,2)/his_imphasor_GN.array(1,2),'b*-'); 
plot((0:size(his_imphase_PGN.array,1)-1)',his_imphase_PGN.array(:,2)/his_imphase_PGN.array(1,2),'kd-'); 
plot((0:size(his_imphasor_PGN.array,1)-1)',his_imphasor_PGN.array(:,2)/his_imphasor_PGN.array(1,2),'mh-');
plot((0:size(his_imphase_PGNR.array,1)-1)',his_imphase_PGNR.array(:,2)/his_imphase_PGNR.array(1,2),'g^-'); 
plot((0:size(his_imphasor_PGNR.array,1)-1)',his_imphasor_PGNR.array(:,2)/his_imphasor_PGNR.array(1,2),'cp-');
leg = legend('E1-GN-pen.reg', 'E2-GN-pen.reg.','E1-PGN','E2-PGN','E1-PGN-R','E2-PGN-R');
leg.FontSize = 14;
tit = title('Rel. Obj. Func: ||J||/||J(0)||');
tit.FontSize = 16;
%%
% Relative error plots
its_imphase_GN      = his_imphase_GN.iters;
its_imphasor_GN     = his_imphasor_GN.iters;
its_imphase_PGN     = his_imphase_PGN.iters;
its_imphasor_PGN    = his_imphasor_PGN.iters;
its_imphase_PGNR    = his_imphase_PGNR.iters;
its_imphasor_PGNR   = his_imphasor_PGNR.iters;

RE_imphase_GN       = zeros(size(its_imphase_GN,2),1);
RE_imphasor_GN      = zeros(size(its_imphasor_GN,2),1);
RE_imphase_PGN      = zeros(size(its_imphase_PGN,2),1);
RE_imphasor_PGN     = zeros(size(its_imphasor_PGN,2),1);
RE_imphase_PGNR     = zeros(size(its_imphase_PGNR,2),1);
RE_imphasor_PGNR    = zeros(size(its_imphasor_PGNR,2),1);

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
plot((0:length(RE_imphase_GN)-1)',RE_imphase_GN,'ro-'); 
hold on;
plot((0:length(RE_imphasor_GN)-1)' ,RE_imphasor_GN,'b*-'); 
plot((0:length(RE_imphase_PGN)-1)' ,RE_imphase_PGN,'kd-'); 
plot((0:length(RE_imphasor_PGN)-1)',RE_imphasor_PGN,'mh-'); 
plot((0:length(RE_imphase_PGNR)-1)' ,RE_imphase_PGNR,'g^-'); 
plot((0:length(RE_imphasor_PGNR)-1)',RE_imphasor_PGNR,'cp-');
leg = legend('E1-GN-pen.reg', 'E2-GN-pen.reg.','E1-PGN','E2-PGN','E1-PGN-R','E2-PGN-R');
leg.FontSize = 14;
tit = title('RE: ||x-xtrue||^2/||xtrue||^2')
tit.FontSize = 16;

%%
% Normalized cross-correlation
its_imphase_GN      = his_imphase_GN.iters;
its_imphasor_GN     = his_imphasor_GN.iters;
its_imphase_PGN     = his_imphase_PGN.iters;
its_imphasor_PGN    = his_imphasor_PGN.iters;
its_imphase_PGNR    = his_imphase_PGNR.iters;
its_imphasor_PGNR   = his_imphasor_PGNR.iters;

NCC_imphase_GN       = zeros(size(its_imphase_GN,2),1);
NCC_imphasor_GN      = zeros(size(its_imphasor_GN,2),1);
NCC_imphase_PGN      = zeros(size(its_imphase_PGN,2),1);
NCC_imphasor_PGN     = zeros(size(its_imphasor_PGN,2),1);
NCC_imphase_PGNR     = zeros(size(its_imphase_PGNR,2),1);
NCC_imphasor_PGNR    = zeros(size(its_imphasor_PGNR,2),1);


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
plot((0:length(NCC_imphase_GN)-1)',NCC_imphase_GN,'ro-'); 
hold on;
plot((0:length(NCC_imphasor_GN)-1)' ,NCC_imphasor_GN,'b*-'); 
plot((0:length(NCC_imphase_PGN)-1)' ,NCC_imphase_PGN,'kd-'); 
plot((0:length(NCC_imphasor_PGN)-1)',NCC_imphasor_PGN,'mh-'); 
plot((0:length(NCC_imphase_PGNR)-1)' ,NCC_imphase_PGNR,'g^-'); 
plot((0:length(NCC_imphasor_PGNR)-1)',NCC_imphasor_PGNR,'cp-');
leg = legend('E1-GN-pen.reg', 'E2-GN-pen.reg.','E1-PGN','E2-PGN','E1-PGN-R','E2-PGN-R');
leg.FontSize = 14;
tit = title('NCC: 0.5*(1 - <x,xtrue>^2/(||x||^2*||xtrue||^2)');
tit.FontSize = 16;
%%
% Print some results to the terminal window
fprintf('\n***** Relative Error Minima *****\n');
fprintf('min(RE_imphase_GN)     = %1.4e \n', min(RE_imphase_GN));
fprintf('min(RE_imphasor_GN)    = %1.4e \n', min(RE_imphasor_GN));
fprintf('min(RE_imphase_PGN)    = %1.4e \n', min(RE_imphase_PGN));
fprintf('min(RE_imphasor_PGN)   = %1.4e \n', min(RE_imphasor_PGN));
fprintf('min(RE_imphase_PGNR)    = %1.4e \n', min(RE_imphase_PGNR));
fprintf('min(RE_imphasor_PGNR)   = %1.4e \n', min(RE_imphasor_PGNR));

fprintf('\n***** Normalized Cross-Correlation Minima *****\n');
fprintf('min(NCC_imphase_GN)    = %1.4e \n', min(NCC_imphase_GN));
fprintf('min(NCC_imphasor_GN)   = %1.4e \n', min(NCC_imphasor_GN));
fprintf('min(NCC_imphase_PGN)   = %1.4e \n', min(NCC_imphase_PGN));
fprintf('min(NCC_imphasor_PGN)  = %1.4e \n', min(NCC_imphasor_PGN));
fprintf('min(NCC_imphase_PGNR)   = %1.4e \n', min(NCC_imphase_PGNR));
fprintf('min(NCC_imphasor_PGNR)  = %1.4e \n', min(NCC_imphasor_PGNR));

fprintf('\n***** Total Time Elapsed *****\n');
fprintf('time(imphase_GN)       = %1.4e \n', time_imphase_GN);
fprintf('time(imphasor_GN)      = %1.4e \n', time_imphasor_GN);
fprintf('time(imphase_PGN)      = %1.4e \n', time_imphase_PGN);
fprintf('time(imphasor_PGN)     = %1.4e \n', time_imphasor_PGN);
fprintf('time(imphase_PGNR)      = %1.4e \n', time_imphase_PGNR);
fprintf('time(imphasor_PGNR)     = %1.4e \n', time_imphasor_PGNR);

fprintf('\n***** Time per Iteration *****\n');
fprintf('time(imphase_GN)/its   = %1.4e \n', time_imphase_GN/size(its_imphase_GN,2));
fprintf('time(imphasor_GN)/its  = %1.4e \n', time_imphasor_GN/size(its_imphasor_GN,2));
fprintf('time(imphase_PGN)/its  = %1.4e \n', time_imphase_PGN/size(its_imphase_PGN,2));
fprintf('time(imphasor_PGN)/its = %1.4e \n', time_imphasor_PGN/size(its_imphasor_PGN,2));
fprintf('time(imphase_PGNR)/its  = %1.4e \n', time_imphase_PGNR/size(its_imphase_PGNR,2));
fprintf('time(imphasor_PGNR)/its = %1.4e \n', time_imphasor_PGNR/size(its_imphasor_PGNR,2));

fprintf('\n***** Outer Iterations til Convergence *****\n');
fprintf('iters(imphase_GN)      = %d \n', size(its_imphase_GN,2)-1);
fprintf('iters(imphasor_GN)     = %d \n', size(its_imphasor_GN,2)-1);
fprintf('iters(imphase_PGN)     = %d \n', size(its_imphase_PGN,2)-1);
fprintf('iters(imphasor_PGN)    = %d \n', size(its_imphasor_PGN,2)-1);
fprintf('iters(imphase_PGNR)     = %d \n', size(its_imphase_PGNR,2)-1);
fprintf('iters(imphasor_PGNR)    = %d \n', size(its_imphasor_PGNR,2)-1);

fprintf('\n***** Avg. Line Search Iterations per Outer Iteration *****\n');
fprintf('LS(imphase_GN)/its     = %1.1f \n', sum(his_imphase_GN.array(:,6)/(size(its_imphase_GN,2)-1)));
fprintf('LS(imphasor_GN)/its    = %1.1f \n', sum(his_imphasor_GN.array(:,6)/(size(its_imphasor_GN,2)-1)));
fprintf('LS(imphase_PGN)/its    = %1.1f \n', sum(his_imphase_PGN.array(:,6)/(size(its_imphase_PGN,2)-1)));
fprintf('LS(imphasor_PGN)/its   = %1.1f \n', sum(his_imphasor_PGN.array(:,6)/(size(its_imphasor_PGN,2)-1)));
fprintf('LS(imphase_PGNR)/its    = %1.1f \n', sum(his_imphase_PGNR.array(:,6)/(size(its_imphase_PGNR,2)-1)));
fprintf('LS(imphasor_PGNR)/its   = %1.1f \n', sum(his_imphasor_PGNR.array(:,6)/(size(its_imphasor_PGNR,2)-1)));
