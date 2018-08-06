function [ phz] = ft_phase_screen(r0, N, delta, L0, l0)
%  function phz = ft_phase_screen(r0, N, delta, L0, l0)
%  Input
%   r0: Fried paramter;
%   N:  number of pixels along one dimension;
%   delta: side length of pixels;
%   L0: the outer scale;
%   l0: the inner scale.
%
%  Output
%   phz: the generated wavefront phase.

%  By Numerical Simulation of Optical Wave Propagation. Adjusted so that
%  the Fourier coefficients are Hermitian symmetric - when applying the
%  inverse Fourier transform, the results (wavefront phase) are directly
%  real.

% last updated Sept 25, 2011.

del_f = 1/(N*delta);
fx = (-N/2:N/2-1) * del_f;

[fx, fy] = meshgrid(fx);
[th, f] = cart2pol(fx, fy);
fm = 5.92/l0/(2*pi);
f0 = 1/L0;

PSD_phi = 0.023*r0^(-5/3) * exp((-f/fm).^2) ...
    ./ (f.^2 + f0^2).^(11/6);
PSD_phi(N/2+1, N/2+1) = 0;

% generated Hermitian symmetric Fourier coefficients .
cn_old = (randn(N) + i *randn(N)) .* sqrt(PSD_phi)*del_f;
%cn = (randn(N) + i*randn(N)).* sqrt(PSD_phi)*del_f;
cn = randn(N) + i*randn(N);
temp = diag((cn_old + cn_old')/2);
for k = 1:N
    cn(k,k) = temp(k);
end

if mod(N,2) == 1
    for k = 1:ceil(N/2)
        for j = 1:N
            cn(k,j) = conj(cn(mod(N-k+1,N)+1, mod(N-j+1,N)+1));
        end
    end
else
    cn(1,N/2+1) = cn(2,2);
    cn(N/2+1,1) = cn(3,3);
    for k = 1:N/2+1
        for j = 1:N
            cn(k,j) = conj(cn(mod(N-k+1,N)+1,mod(N-j+1,N)+1));
        end
    end
end
% randn('seed',1);
% cn_real = randn(N);
% randn('seed',2);
% cn_img = randn(N);
% cn = (cn_real + i *cn_img) .* sqrt(PSD_phi)*del_f;

cn = cn.* sqrt(PSD_phi)*del_f;

%phz_old = real(ift2(cn_old,1));
phz = ifftshift(ifft2(ifftshift(cn)))*N^2;

%figure(1), imshow(phz_old, []), colormap(jet), colorbar, title('unSymm generated phase')
%figure(2), imshow(phz, []), colormap(jet), colorbar, title('Symm generated phase')