% This script makes L-curve and discrepancy principle plots for the
% regularation iterations saved in the files regInfo_50_30_?.mat and


clear all; close all;
setupBispectrumData;
load('regInfo_50_30_imphase_c.mat');

% First, plot discrepancy principle
max_gd = max(image_imphase_gd_alpha);
max_nlcg = max(image_imphase_nlcg_alpha);
max_newt = max(image_imphase_newt_alpha);
for j=1:11
    image_imphase_gd_alpha(:,j) = image_imphase_gd_alpha(:,j)/max_gd(j);
    image_imphase_nlcg_alpha(:,j) = image_imphase_nlcg_alpha(:,j)/max_nlcg(j);
    image_imphase_newt_alpha(:,j) = image_imphase_newt_alpha(:,j)/max_newt(j);
end

for j = 1:11
    res_alpha_gd_p(j) = norm(image_imphase_gd_alpha(:,j) - obj(:))^2;
    res_alpha_nlcg_p(j) = norm(image_imphase_nlcg_alpha(:,j) - obj(:))^2;
    res_alpha_newt_p(j) = norm(image_imphase_newt_alpha(:,j) - obj(:))^2;
end
    
figure; semilogx(alpha, res_alpha_gd_p,'ro'); hold on; 
plot(alpha, res_alpha_nlcg_p,'bs'); plot(alpha, res_alpha_newt_p,'kd');

% Next, plot the L-curve
figure; loglog(image_norm_gd.^2, res_alpha_gd_p,'ro'); hold on;
plot(image_norm_nlcg.^2, res_alpha_nlcg_p,'bs'); 
plot(image_norm_newt.^2, res_alpha_newt_p,'kd'); axis tight;