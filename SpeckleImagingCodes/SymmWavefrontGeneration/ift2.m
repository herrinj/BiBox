function g = ift2(G,delta_f)
%  g = ift2(G,delta_f)

% By Numerical Simulation of Optical Wave Propagation

N = size(G,1);
g = ifftshift(ifft2(ifftshift(G))) * (N * delta_f)^2;
