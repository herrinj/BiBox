%
% Plot the results for some images for the paper
%

setupBispectrumData; % 50 frames, 3e6 photo events for data, 5000 photo events for star, sigma_rn = 5, Satellite.jpg, alpha = 100 for reg.
load iterInfo_50_30_reg;

figure; imagesc(reshape(obj,[256 256])/max(obj(:))); axis image; colorbar; title('True Object');

avg_data_frame = sum(data,3)/size(data,3); avg_data_frame = avg_data_frame/max(avg_data_frame(:));
figure; imagesc(avg_data_frame); axis image; colorbar; title('Avg. Data Frame');

image_phase = real(fftshift(ifft2(fftshift(pospec.*exp(1i*reshape(iters_phase_gd(:,end),[256 256]))))));
image_phase = image_phase/max(image_phase(:));
figure; imagesc(image_phase); axis image; colorbar; title('E_1(\phi)');

image_phasor = real(fftshift(ifft2(fftshift(pospec.*exp(1i*reshape(iters_phasor_gd(:,end),[256 256]))))));
image_phasor = image_phasor/max(image_phasor(:));
figure; imagesc(image_phasor); axis image; colorbar; title('E_2(\phi)');

figure; imagesc(reshape(iters_imphase_gd(:,end),[256 256])/max(iters_imphase_gd(:,end))); axis image; colorbar; title('E_1(f)');

figure; imagesc(reshape(iters_imphasor_gd(:,end),[256 256])/max(iters_imphasor_gd(:,end))); axis image; colorbar; title('E_2(f)');