function [func, grad, hess] = phase_rec(phase, A, bispec_phase, weights, alpha, hess_opts, hess_const)
%
%   Evaluates objective function, gradient, and approximate Hessian for the
%   phase recovery objective function given in Regagnon, Eq. 44 which is 
%
%   J(phase) = min 0.5*sum(mod_2pi(bispec_phase - A*phase)^2*weights^2)
%
%   Inputs: phase - input phase value
%               A - sparse matrix with 3 non-zero entried per row
%                   corresponding to a u, v, u+v triple in the bispectrum
%    bispec_phase - accumulated phase of the data bispectrum
%         weights - weights based on the signal-to-noise ratio of the
%                   given bispectum elements
%           alpha - regularization parameter
%       hess_opts - switch indicating which Hessian option to use
%                       0,'full'  - full approximate Hessian as a matrix
%                       1,'trunc' - truncated Hessian after symmetric
%                                   minimum degree reordering as a struct
%                       2,'ichol' - incomplete Cholesky with reduced 
%                                   sparcity pattern of truncated Hessian 
%                                   after a minimum degree reordering
%                       3,'const' - Hessian constant, supplied in function
%                                   argument hess_const
%      hess_const - in the case of a pre-computable, constant Hessian which
%                   is independent of the current phase, input the
%                   matrix/structure/function here and select hess_opts = 3
%
%
%   Outputs: func - objective function value for input phase
%            grad - gradient vector for input phase
%            hess - approximate Hessian (independent of input phase) as
%                   a struct with fields depending on hess_opts
%
if nargin<1
    runMinimalExample;
    return;
end

% Account for case when weights are not given
if isempty(weights)
    weights = ones(length(bispec_phase),1);
end

doGrad = nargout > 1;   
doHess = nargout > 2;

% Objective function
phase = phase(:);
phase_diff = mod(bispec_phase - A*phase + pi,2*pi) - pi;
func = 0.5*sum(weights.*(phase_diff.^2));

% Gradient of objective function
if doGrad
    grad = -A'*(weights.*phase_diff);
end

% Note that the approximate Hessian does not depend on phi, thus we often
% only compute and factor this once for a given Gauss-Newton call
if doHess
    switch hess_opts
        case{'full',0} % returns the full, untruncated Hessian as a struct
            hess.matrix = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
            hess.flag = 'full';
        
        case{'trunc',1} % returns a reorders, truncated Hessian as a struct
            H = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
            p = symamd(H);
            Hp = H(p,p);
            [i,j] = find(Hp~=0);
            Hp = Hp(1:max(i),1:max(j));
            dim = max(j);
            
            % Load the truncated Hessian, permutation, dimension, and type
            % flag into a structure
            hess.matrix = Hp;
            hess.perm = p;
            hess.dim = dim; 
            hess.flag = 'trunc';
       
        case{'ichol',2} % iChol of the reordered, truncated Hessian as a struct
            H = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
            p = symamd(H);
            Hp = H(p,p);
            [i,j] = find(Hp~=0);
            Hp = Hp(1:max(i),1:max(j));
            dim = max(j);
            L = ichol(Hp);
            
            % Load the truncated Hessian, permutation, dimension, and type
            % flag into a structure
            hess.matrix = L;
            hess.perm = p;
            hess.dim = dim;
            hess.flag = 'ichol';
        case{'const',3}
            if nargin < 7 || isempty(hess_const)
                fprintf('Error: Must supply hess_const for hess_opts = const \n');
            end
            hess = hess_const;
    end
end

end

function runMinimalExample
    [nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50);
    setupBispectrumData;
    fctn = @(phase) phase_rec(phase,A,bispec_phase,weights,0,0);
    checkDerivative(fctn, rand(prod(size(phase_recur)),1)); % Pro-tip: to test Hessian in proximity to solution, use phase_recur(:) for guess
end
