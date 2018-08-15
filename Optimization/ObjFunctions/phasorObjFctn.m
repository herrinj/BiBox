function [Jc, para, dJ, H] = phaseObjFctn(x, A, bispec_phase,varargin)
%
%   Evaluates objective function, gradient, and approximate Hessian for the
%   phase recovery objective function given in Regagnon, Eq. 45 which is 
%
%   J(phase) = min 0.5*sum([real(delta)^2 - imag(delta)^2]*weights)
%              where delta = exp(i*bispec_phase) - exp(i*(A*x))
%
%   Inputs:     x - input phase value
%               A - sparse matrix with 3 non-zero entried per row
%                   corresponding to a u, v, u+v triple in the bispectrum
%    bispec_phase - accumulated phase of the data bispectrum
%         weights - weights based on the signal-to-noise ratio of the
%                   given bispectum elements
%           alpha - regularization parameter
%   Optional Inputs:
%           Hflag - switch indicating which Hessian option to use
%                       0,'full'  - full approximate Hessian as a matrix
%                       1,'trunc' - truncated Hessian after symmetric
%                                   minimum degree reordering as a struct
%                       2,'ichol' - incomplete Cholesky with reduced 
%                                   sparcity pattern of truncated Hessian 
%                                   after a minimum degree reordering
%
%   Outputs:   Jc - objective function value for input phase
%            para - any parameters you want to be output (default empty)
%              dJ - gradient vector for input phase
%               H - approximate Hessian (independent of input phase) as
%                   a function handle
%
if nargin<1
    runMinimalExample;
    return;
end

% Set up default parameters for optional inputs
weights     = ones(length(bispec_phase),1);
Hflag       = 'ichol';

para = [];
doGrad = nargout > 2;   
doHess = nargout > 3;

for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

% Objective function
cosb = cos(bispec_phase);
sinb = sin(bispec_phase);
Ap = A*x(:);
cosAp = cos(Ap);
sinAp = sin(Ap);

Jc = 0.5*sum(weights.*((cosb - cosAp).^2 + (sinb - sinAp).^2));

% Gradient of objective function
if doGrad
    dJ = A'*(weights.*(cosb.*sinAp - sinb.*cosAp));
    dJ = dJ';
end

% Note that the Hessian does not depend on phi if we omit the diagonal term 
% below. For the constant Hessian, we compute the Hessian only once as a 
% persistent variable and only have to avoid computing it more than once. 
% The diagonal, if wanted, is...
% diagonal = sparse(1:m,1:m,bispec_cos.*Aphase_cos + bispec_sin.*Aphase_sin); % include?
if doHess
    persistent Hper;
    if isempty(Hper)
        Hper = struct();
        switch Hflag
            case{'full',0} % returns the full, untruncated Hessian as a struct
                Hper.matrix= A'*spdiags(weights,0,size(A,1),size(A,1))*A;
                Hper.flag  = Hflag;
        
            case{'trunc',1} % returns a reordered, truncated Hessian as a struct
                ADA     = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
                p       = symamd(ADA);
                ADAp    = ADA(p,p);
                [i,j]   = find(ADAp~=0);
                ADAp    = ADAp(1:max(i),1:max(j));
                dim     = max(j);
            
                % Load the truncated Hessian, permutation, dimension, and flag
                Hper.matrix= ADAp;
                Hper.perm  = p;
                Hper.dim   = dim; 
                Hper.flag  = Hflag;
       
            case{'ichol',2} % iChol of the reordered, truncated Hessian as a struct
                ADA     = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
                p       = symamd(ADA);
                ADAp    = ADA(p,p);
                [i,j]   = find(ADAp~=0);
                ADAp    = ADAp(1:max(i),1:max(j));
                dim     = max(j);
                L       = ichol(ADAp);

                % Load the truncated Hessian, permutation, dimension, and flag
                Hper.matrix = L;
                Hper.perm = p;
                Hper.dim = dim;
                Hper.flag = Hflag;
        end
    end
    H = Hper;
end

end

function runMinimalExample
    [nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50);
    setupBispectrumData;
    fctn  = @(x) phasorObjFctn(x, A, bispec_phase,'Hflag','full');

    x = randn(numel(phase_recur),1);
    [f0,~,df,d2f] = fctn(x);

    h = logspace(-1,-10,10);
    v = randn(numel(x),1);

    dvf     = df*v;
    d2vf    = v'*(d2f.matrix*v);
    
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
