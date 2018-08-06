% 
%   This produces the figures for the Hessian treatment for phase_rec and
%   phasor_rec objective functions with incomplete Cholesky
%
clear; close all;

% Complete the various changes to the Hessian
b = bindex(256, 64, 5);
A = phi_matrix(b, 256^2);
H = A'*A;
p = symamd(H);
Hp = H(p,p);
[i,j] = find(Hp~=0);
Hp2 = Hp(1:max(i),1:max(j));
L = ichol(Hp2);

% Produce the figures
figure; spy(H); set(gca,'XTick',[],'YTick',[]); title('Full Hessian');
figure; spy(Hp); set(gca,'XTick',[],'YTick',[]); title('Permutation');
figure; spy(Hp2); set(gca,'XTick',[],'YTick',[]); title('Truncation');
figure; spy(L); set(gca,'XTick',[],'YTick',[]); title('Factorization');

