function [phasor] = recur_phase(bispec, bindex, N, weights)
%
%
%   Input:    bisp - accumulated bispectrum phasor of the data, given in the
%                    form exp(i*theta) with modulus 1
%           bindex - index structure for vectorizing the bispectrum
%                    accumulation and recursive algorithm
%                N - size of expected image NxN for phase recovery
%          weights - weights for the reliability of each bispectral element
%           
%   Output: phasor - recovered phase from the bispectrum, output in phasor
%                    form 
%
%
if nargin <4
    weights = ones(length(bispec),1);
end
    
u_inds = bindex.u;
v_inds = bindex.v;
uv_inds = bindex.u_v;

phasor = zeros(N,N);

% Initial values 
phasor(N/2 + 1, N/2 + 1) = 1;
phasor(N/2, N/2 + 1) = 1;
phasor(N/2 + 2, N/2 + 1) = 1;
phasor(N/2 + 1, N/2) = 1;
phasor(N/2 + 1, N/2 + 2) = 1;


% This implements the recursive phase algorithm Eq. 2 in Tyler & Schulze, 2004
for j = 1:length(uv_inds)
    if (phasor(u_inds(j)) ~=0 & phasor(v_inds(j)) ~=0)
        summand =  weights(j)*bispec(j)/(phasor(u_inds(j))*phasor(v_inds(j))); % need to multiply this quantity by w(u,v) in future
        summand = conj(summand);
        phasor(uv_inds(j)) = phasor(uv_inds(j)) + summand;
    end    
end





