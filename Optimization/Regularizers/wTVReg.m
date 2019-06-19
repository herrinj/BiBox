function [Rc, dR, d2R] = wTVReg(x, xref, dims)
%
%   Total variation regularizer for regular 2D image on domain [0 1 0 1]
%
%   Input:    x - image
%          xref - reference image
%          dims - dimensions of image
%
%   Output:  Rc - regularizer (scalar)
%            dR - gradient of regularizer (column vector)
%           d2R - Hessian of regularizer (matrix)

if nargin == 0
   runMinimalExample;
   return;
end

eps = 1e-3; 
h   = ones(size(dims))./dims;

dx  = x - xref;
S   = getGradient([0 1 0 1],dims); % recall that Div = Grad'

v   = prod(h)*ones(1, prod(dims));   % constant vector for rectangular grid

A1  = kron(speye(dims(2)),spdiags([0.5*ones(dims(1)), 0.5*ones(dims(1))],[0 1],dims(1), dims(1)-1));
A2  = kron(spdiags([0.5*ones(dims(2)) 0.5*ones(dims(2))],[0 1],dims(2),dims(2)-1),speye(dims(1)));
Af  = [A1, A2];

wTV = sqrt(Af*(S*dx).^2 + eps);

Rc  = v*wTV;
D   = spdiags(Af'*(v(:)./wTV),0,size(Af,2),size(Af,2));
d2R = S'*(D*S);
dR  = d2R*dx;

end

function runMinimalExample
%
%   Runs minimal example with derivative check for function.
%
    n    = 128;
    dims = [n n];
    x    = phantom(n); x = x(:);
    xref = zeros(size(x));
    fctn = @(y) wTVReg(y, xref, dims);
    [f0,df,d2f] = fctn(x);
    
    h = logspace(-1,-10,10);
    v = randn(numel(x),1);

    dvf     = reshape(df,1,[])*v;
    d2vf    = v'*(d2f*v);
    
    T0 = zeros(1,10);
    T1 = zeros(1,10);
    T2 = zeros(1,10);

    for j=1:length(h)
        ft      = feval(fctn,x+h(j)*v);                         % function value
        T0(j)   = norm(f0-ft);                               % TaylorPoly 0
        T1(j)   = norm(f0+h(j)*dvf - ft);                    % TaylorPoly 1
        T2(j)   = norm(f0+h(j)*dvf+0.5*h(j)*h(j)*d2vf - ft); % TaylorPoly 2
        fprintf('h=%12.4e     T0=%12.4e    T1=%12.4e    T2=%12.4e\n', h(j), T0(j), T1(j), T2(j));
    end    
    
    ph = loglog( h, [T0;T1;T2]); set(ph(2),'linestyle','--')
    th = title(sprintf('%s: T0,T1,T2 vs. h',mfilename));
    return;
end