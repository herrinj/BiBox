function [Jc, para, dJ, H] = imphaseObjFctn(x, A, bispec_phase, dims, pupil_mask, varargin)
%
%   Evaluates objective function, gradient, and approximate Hessian for the
%   phase recovery objective function given in Glindemann & Dainty, Eq. 11  
%
%   J(phase) = min 0.5*sum(mod_2pi(bispec - A*phase(x))^2*weights^2)
%
%   and phase is the current phase of the current input image
%
%   Inputs:     x - input image value
%               A - sparse matrix with 3 non-zero entried per row
%                   corresponding to a u, v, u+v triple in the bispectrum
%    bispec_phase - accumulated phase of the data bispectrum
%            dims - dimension of image and pupil mask
%      pupil_mask - pupil mask for current problem 
%   
%   Variable Inputs:      
%         weights - weights based on the signal-to-noise ratio of the
%                   given bispectum elements
%           alpha - regularization parameter
%     regularizer - flag dictating which regularization option to use
%                       0,'pos' - enforce positivity of solution image as
%                                 proposed by Glindemann, Dainty
%                       1,'pow' - ties solution image's power spectrum to
%                                 a fixed, calculated power spectrum given
%                                 as argument 8
%                       2,'grad'- discrete gradient for smoothness
%                       3,'eye  - identity regularizer
%          pospec - fixed power spectrum for regularization option 1 
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
alpha       = 0;
regularizer = 'eye';

para = [];
doGrad = nargout > 2;   
doHess = nargout > 3;


for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

% Evaluate pieces for obj. function and derivatives
x = reshape(x, dims);
X = fftshift(fft2(fftshift(x)))/256; % Scale appropriately
phase = angle(X);

%   Calculate the difference between the calculated phase and the
%   bispectrum phase, modulo 2pi
phase_diff = bispec_phase - A*phase(:);
phase_diff = mod(phase_diff + pi, 2*pi) - pi;

Jc = 0.5*sum(weights.*(phase_diff.^2));

% Regularization term for enforcing non-negativity
if alpha > 0
    switch regularizer
        case{'pos',0}
            neg_inds = find(x < 0);
            Sx = abs(x(neg_inds)).^2;
            Sc = 0.5*alpha*sum(Sx(:));
        case{'pow',1}
            Fx = fftshift(fft2(x));
            Sx = Fx.*conj(Fx).*pupil_mask - pospec.^2;
            Sc = 0.5*alpha*sum(Sx(:).^2);
        case{'grad',2}
            S = getGradient([0 1 0 1],dims);
            Sx = S*x(:);
            Sc = 0.5*alpha*(Sx'*Sx);
        case{'eye',3}
            Sc = 0.5*alpha*(x(:)'*x(:));
    end
    Jc = Jc + Sc;
end


if doGrad % take the derivative of the objective function w.r.t. the image
    % First, gradient w.r.t the phase (same as phase_rec objective)
    phase_diff = -A'*(weights.*phase_diff);
    phase_diff = reshape(phase_diff, dims);

       % Next, take gradient of phase with respect to the image in direction
    piece = zeros(dims);
    inds = find(abs(X) ~=0); % avoid dividing by zero
    piece(inds) = phase_diff(inds)./X(inds);
    dJ = imag(fftshift(fft2(fftshift(piece.*pupil_mask))))/256;
    dJ = reshape(dJ,1,[]);
    
    % Lastly, combine and add the gradient of regularization 
    if alpha > 0
        switch regularizer
            case{'pos',0}
                dJ(neg_inds) = dJ(neg_inds) + alpha*x(neg_inds)'; 
            case{'pow',1}
                dS= 2*real(reshape(fft2(fftshift(conj(Fx).*Sx.*pupil_mask)),1,[]));
                dJ = dJ + alpha*dS(:)';
            case{'grad',2}
                dJ = dJ + alpha*(Sx'*S);
            case{'eye',3}
                dJ = dJ + alpha*x(:)';
        end
    end
end

if doHess % Returns function handle for Hessian action on a vector. Symmetrix so H' = H
    
    % Note: we omit the following diagonal from ADA so that we can reuse it
    % as a constant without recalculating at each iterate.
    % diagonal = sparse(1:length(bispec_phase),1:length(bispec_phase), bispec_cos.*Aphase_cos + bispec_sin.*Aphase_sin);
    persistent ADA;
    if isempty(ADA)
        ADA = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
    end
    
    op = @(p) hessOp(p,X,ADA,pupil_mask,dims);

    if alpha > 0
        switch regularizer
            case{'pos',0}
                D = sparse(neg_inds, neg_inds, ones(length(neg_inds),1), prod(dims), prod(dims));
                H = @(p) op(p) + alpha*D*p;
            case{'pow',1}
                regOp = @(p) real(4*reshape(fft2(fftshift(conj(Fx).*real(conj(Fx).*fftshift(fft2(reshape(p,dims))).*pupil_mask).*pupil_mask)),[],1));
                H = @(p) op(p) + alpha*regOp(p);
            case{'grad',2}
                H = @(p) op(p) + alpha*(S'*(S*p));
            case{'eye',3}
                H = @(p) op(p) + alpha*p;
        end
    else
        H = op;
    end
end

end

function runMinimalExample
%
%   Runs minimal example with derivative check for function.
%
    [nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50);
    setupBispectrumData;
    image_recur     = real(fftshift(ifft2(fftshift(pospec.*exp(1i*phase_recur)))));
    dims            = size(image_recur);
    fctn  = @(x) imphaseObjFctn(x, A, bispec_phase, dims, pupil_mask, 'regularizer','pos','alpha',100);

    x = randn(numel(image_recur),1);
    [f0,~,df,d2f] = fctn(x);

    h = logspace(-1,-10,10);
    v = randn(numel(x),1);

    dvf     = df*v;
    d2vf    = v'*d2f(v);
    
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

function [Hp] = hessOp(p, X, ADA, pupil_mask, dims)
% 
%   [Hp] = hessOp(p, X, ADA, pupil_mask);
%
%   Computes the symmetric Hessian operator in the direction pk.
%
%           H = (dph_di)^* ADA * (dph_di)
%
%   where dph_di is the derivative of the phase w.r.t the image. This
%   function returns H*pk
%
%   Inputs:        pk - direction vector for Hessian operator to act on
%                   X - in Fourier spaceimage calculated as fftshift(fft2(fftshift(x)))/256
%                 ADA - sparse matrix A'*D*A where A has 3 non-zero entried per row
%                       corresponding to a u, v, u+v triple in the bispectrum
%                       and D is a diagonal matrix depending on the
%                       objective function (SNR weights for imphase_rec and
%                       SNR weights times diagonal term for imphasor_rec)
%          pupil_mask - pupil mask for current problem
%
%   Outputs:       Hp - vector of Hessian operator in the direction pk 
%

% First, evaluate the dph_di operator times the direction pk
X       = reshape(X,dims);
mask    = reshape(pupil_mask,dims);
X       = X.*mask;
inds    = find(abs(X) ~= 0);
zinds   = find(abs(X) == 0);

dph_p       = reshape(p,dims);
dph_p       = fftshift(fft2(fftshift(dph_p)))/256;
dph_p(inds) = imag(dph_p(inds)./X(inds)); % Avoids division by zero
dph_p(zinds)= 0;
dph_p       = dph_p.*mask;

% Next, multiply dph_p times the operator A'*diagonal*A 
ADAp = reshape(ADA*dph_p(:),dims);

% Lastly, evaluate the adjoint dph_di^* times Hdph_p
Hp          = zeros(size(ADAp));
Hp(inds)    = ADAp(inds)./X(inds);
Hp(zinds)   = 0;
Hp          = imag(fftshift(fft2(fftshift(Hp.*mask))))/256;
Hp          = Hp(:);

end