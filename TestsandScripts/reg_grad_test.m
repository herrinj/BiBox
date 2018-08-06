clear all; close all;
setupBispectrumData;
IMAGE_recur = (pospec + 5*rand(256)).*exp(1i*phase_recur);
image_recur = real(fftshift(ifft2(fftshift(IMAGE_recur))));
im = reshape(image_recur,[256 256]);
pospec = reshape(pospec,[256 256]);

v  = randn(size(im));
Fim = fftshift(fft2(im));
inside = Fim.*conj(Fim).*pupil_mask - pospec.^2;
f  = 0.5*sum(inside(:).^2);
df = 2*real(reshape(fft2(fftshift(conj(Fim).*inside.*pupil_mask)),[],1))'*v(:);
df2 = v(:)'*real(2*reshape(fft2(fftshift(conj(Fim).*real(2*conj(Fim).*fftshift(fft2(v)).*pupil_mask).*pupil_mask)),[],1));

error = zeros(10,3);
for j=1:10
    ft = 0.5*sum(reshape(fftshift(fft2(im+10.0^(-j)*v)).*conj(fftshift(fft2(im+10.0^(-j)*v))).*pupil_mask - pospec.^2,[],1).^2);
    error(j,1) = norm(f(:) - ft(:))/norm(f(:));
    error(j,2) = norm(f(:) + 10.0^(-j)*df(:)- ft(:))/norm(f(:));
    error(j,3) = norm(f(:) + 10.0^(-j)*df(:) + 10.0^(-j*2)*df2(:) - ft(:))/norm(f(:));
    fprintf('h=%1.2e\t %1.4e\t %1.4e\t %1.4e\t \n', 10.0^(-j), error(j,1), error(j,2), error(j,3));
end

loglog(logspace(-1,-10,10),error);
legend('|f - ft|', '|f + h*df*v - ft|','|f + h*df*v + h^2*v*df2*v - ft|');

% Adjoint test
w = randn(size(im));
z = randn(size(im));

dfw = real(2*reshape(conj(Fim).*fftshift(fft2(w)).*pupil_mask ,[],1));
check1 = z(:)'*dfw(:);
dfz = real(2*reshape(fft2(fftshift(conj(Fim).*z.*pupil_mask)),[],1));
check2 = dfz(:)'*w(:);
