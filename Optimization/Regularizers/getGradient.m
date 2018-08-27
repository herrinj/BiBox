function R = getGradient(omega,m)
% =========================================================================
% function R = getGradient(omega,m)
%
% build gradient operator for images
%
% Input:
%  omega  - description of computational domain
%     mf  - spatial discretization
%
% Output:
%     R   - gradient operator, sparse
%
% =========================================================================
dim = length(omega)/2;
h     = (omega(2:2:end)-omega(1:2:end))./m;
dxi   = @(i) spdiags(ones(m(i),1)*[-1 1],0:1,m(i)-1,m(i))/h(i);
E     = @(i) speye(m(i));
switch dim
    case 1
        R = dxi(1);
    case 2
        R = [...
               kron(E(2)  ,dxi(1)); ...
               kron(dxi(2),E(1))
            ];
    case 3
        R = [...
               kron(E(3)  ,kron(E(2)  ,dxi(1))); ...
               kron(E(3)  ,kron(dxi(2),E(1))); ...
               kron(dxi(3),kron(E(2)  ,E(1))); ...
            ];
 
end
end

