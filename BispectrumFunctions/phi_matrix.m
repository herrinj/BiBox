function [A] = phi_matrix(bindex,N)
%
%   Makes sparse matrix A corresponding to the u,v, u+v phase triples that
%   correspond to the object bispectum phase in Eq. 28, 42 in Regagnon
%   
%   Input: bindex - index structure where the kth entry is the kth row of 
%                   the matrix A corresponding to the kth u,v,u+v triple
%               N - dimension of phase to retrieve, typically n^2 for an 
%                   nxn image
%
%   Output:     A - sparse matrix with dimension MxN where M is the number 
%                   u,v,u+v triples in the bispectrum index and N is the
%                   dimension of the object phase
%

if nargin <2
    runMinimalExample;
    return;
end

u_inds = bindex.u;
v_inds = bindex.v;
s_inds = bindex.u_v;

M = length(u_inds);

I = [1:M, 1:M, 1:M];
J = [u_inds, v_inds, s_inds];
V = [ones(1,M), ones(1,M), -1*ones(1,M)];

A = sparse(I,J,V,M,N);

end

function runMinimalExample
    b = bindex(16,8,3,0);
    N = 16^2;
    A = phi_matrix(b,N);
    spy(A); title('A matrix for a 16x16 image');
end

