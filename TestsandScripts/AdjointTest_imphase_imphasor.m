%
% Adjoint test for dphi_di operator used in imphasor_rec and imphase_rec
% objective functions 
%
setupBispectrumData;
image_recur = real(fftshift(ifft2(fftshift(pospec.*exp(i*phase_recur)))));
IMAGE = fftshift(fft2(fftshift(image_recur))).*pupil_mask;

v = rand(256,256);
w = rand(256,256);

dims = size(pupil_mask);
inds = find(abs(IMAGE) ~= 0);
zero_inds = find(abs(IMAGE) == 0);

dph_w = reshape(w,dims);
dph_w = fftshift(fft2(fftshift(dph_w)))/256;
dph_w = real(IMAGE).*imag(dph_w) - imag(IMAGE).*real(dph_w);
dph_w(inds) = dph_w(inds)./(abs(IMAGE(inds)).^2);
dph_w(zero_inds) = 0;
dph_w = dph_w.*pupil_mask;

term1 = v(:)'*dph_w(:);

% Lastly, evaluate the adjoint dph_di^* times Hdph_pk
dph_v1 = zeros(size(IMAGE));
dph_v2 = zeros(size(IMAGE));

dph_v1(inds) = v(inds)./(abs(IMAGE(inds)).^2);
dph_v1(zero_inds) = 0;
dph_v1 = dph_v1.*real(IMAGE);
dph_v2(inds) = v(inds)./(abs(IMAGE(inds)).^2);
dph_v2 = -dph_v2.*imag(IMAGE);
dph_v2(zero_inds) = 0;

dph_v1 = imag(fftshift(fft2(fftshift(dph_v1.*pupil_mask))))/256;
dph_v2 = real(fftshift(fft2(fftshift(dph_v2.*pupil_mask))))/256;
dph_v = dph_v1(:) + dph_v2(:);

term2 = dph_v(:)'*w(:);

% Term 2 for efficiency
dph_v_b = zeros(size(IMAGE));
dph_v_b(inds) = v(inds)./IMAGE(inds);
dph_v_b(zero_inds) = 0;
dph_v_b = imag(fftshift(fft2(fftshift(dph_v_b.*pupil_mask))))/256;

term3 = dph_v_b(:)'*w(:);

% Term 1 for efficiency
dph_w = reshape(w,dims);
dph_w_b = fftshift(fft2(fftshift(dph_w)))/256;
dph_w_b(inds) = imag(dph_w_b(inds)./IMAGE(inds));
dph_w_b(zero_inds) = 0;
dph_w_b = dph_w_b.*pupil_mask;

term4 = v(:)'*dph_w_b(:);

norm(term1 - term2)