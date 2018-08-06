function [func, grad, hess] = imphase_rec(image, A, bispec_phase, weights, pupil_mask, ADA, alpha, reg_flag, pospec)
%
%   Evaluates objective function, gradient, and approximate Hessian for the
%   phase recovery objective function given in Glindemann & Dainty, Eq. 11  
%
%   J(phase) = min 0.5*sum(mod_2pi(bispec - A*phase)^2*weights^2)
%
%   and phase is the current phase of the current input image
%
%   Inputs: image - input image value
%               A - sparse matrix with 3 non-zero entried per row
%                   corresponding to a u, v, u+v triple in the bispectrum
%    bispec_phase - accumulated phase of the data bispectrum
%         weights - weights based on the signal-to-noise ratio of the
%                   given bispectum elements
%      pupil_mask - pupil mask for current problem 
%             ADA - constant part of the Hessian A'*diag(weights)*A
%           alpha - regularization parameter
%        reg_flag - flag dictating which regularization option to use
%                       0,'pos' - enforce positivity of solution image as
%                                 proposed by Glindemann, Dainty
%                       1,'pow' - ties solution image's power spectrum to
%                                 a fixed, calculated power spectrum given
%                                 as argument 8
%                       2,'comb'- combine both of the regularization
%                                 options above
%          pospec - fixed power spectrum for regularization options 1,2 
%
%   Outputs: func - objective function value for input phase
%            grad - gradient vector for input phase
%            hess - approximate Hessian (independent of input phase) as
%                   a function handle
%

if nargin<1
    runMinimalExample;
    return;
end

if isempty(weights)
    weights = ones(length(bispec_phase),1);
end

doGrad = nargout > 1;   
doHess = nargout > 2;

dims = size(pupil_mask);
image = reshape(image, dims);
IMAGE = fftshift(fft2(fftshift(image)))/256;
phase = angle(IMAGE);

%   Calculate the difference between the calculated phase and the
%   bispectrum phase, modulo 2pi
phase_diff = bispec_phase - A*phase(:);
phase_diff = mod(phase_diff + pi, 2*pi) - pi;

% Regularization term for enforcing non-negativity
reg_term = 0;
neg_inds = find(image < 0);
if alpha > 0
    switch reg_flag
        case{'pos',0}
            reg_term = abs(image(neg_inds)).^2;
            reg_term = sum(reg_term(:));
        case{'pow',1}
            Fim = fftshift(fft2(image));
            inside = Fim.*conj(Fim).*pupil_mask - pospec.^2;
            reg_term = sum(inside(:).^2);
    end
end

func = 0.5*sum(weights.*(phase_diff.^2))+ 0.5*alpha*reg_term;

if doGrad % take the derivative of the objective function w.r.t. the image
    % First, gradient w.r.t the phase (same as phase_rec objective)
    phase_diff = -A'*(weights.*phase_diff);
    phase_diff = reshape(phase_diff, dims);

    % Next, take gradient of phase with respect to the image in direction
    piece = zeros(dims);
    inds = find(abs(IMAGE) ~=0); % avoid dividing by zero
    piece(inds) = phase_diff(inds)./IMAGE(inds);
    piece = imag(fftshift(fft2(fftshift(piece.*pupil_mask))))/256;
    grad = piece(:);
    
    % Lastly, combine and add the gradient of regularization 
    if alpha > 0
        switch reg_flag
            case{'pos',0}
                grad(neg_inds) = grad(neg_inds) + alpha*image(neg_inds);
                grad = grad(:);   
            case{'pow',1}
                reg_grad = 2*real(reshape(fft2(fftshift(conj(Fim).*inside.*pupil_mask)),[],1));
                grad = grad(:) + alpha*reg_grad(:);
        end
    end

end

if doHess % returns the full, untruncated Hessian-matrix multiplication in
          % direction pk as a function handle
    
    switch reg_flag
        case{'pos',0}
            hess.operator = @(pk) image_rec_hess_mult(pk, IMAGE, ADA, pupil_mask, alpha, reg_flag, neg_inds);
            hess.flag = 'oper';
        case{'pow',1}
            hess.operator = @(pk) image_rec_hess_mult(pk, IMAGE, ADA, pupil_mask, alpha, reg_flag, image);
            hess.flag = 'oper';
    end               
end

end

function runMinimalExample
    [nfr, D_r0, image_name, K_n, sigma_rn] = setupBispectrumParams('nfr',50);
    setupBispectrumData;
    image_recur = real(fftshift(ifft2(fftshift(pospec.*exp(i*phase_recur)))));
    ADA = A'*spdiags(weights,0,size(A,1),size(A,1))*A;
    fctn = @(image) imphase_rec(image, A, bispec_phase, weights, ADA, pupil_mask, 1.0, 'pos', pospec);
    checkDerivative(fctn, rand(prod(size(image_recur)),1)); % Pro-tip: to test Hessian in proximity to solution, use image_recur(:) for guess
end