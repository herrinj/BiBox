function [Hpk] = image_rec_hess_mult(pk, IMAGE, ADA, pupil_mask, alpha, reg_flag, reg_param)
% 
%   [Hpk] = image_rec_hess_mult(pk, IMAGE, A, pupil_mask, diagonal, alpha, neg_inds);
%
%   Computes the symmetric Hessian operator in the direction pk for use in
%   the objective functions imphase_rec.m and imphasor_rec.m . The Hessian
%   will have the form:
%
%           H = (dph_di)^* A' * diagonal* A * (dph_di)
%
%   where dph_di is the derivative of the phase w.r.t the image. This
%   function returns H*pk
%
%   Inputs:        pk -direction vector for Hessian operator to act on
%               IMAGE - image calculated as fftshift(fft2(fftshift(image)))/256
%                 ADA - sparse matrix A'*D*A where A has 3 non-zero entried per row
%                       corresponding to a u, v, u+v triple in the bispectrum
%                       and D is a diagonal matrix depending on the
%                       objective function (SNR weights for imphase_rec and
%                       SNR weights times diagonal term for imphasor_rec)
%          pupil_mask - pupil mask for current problem
%               alpha - regularization parameter
%            reg_flag - flag dictating which regularization option to use
%                       0,'pos' - enforce positivity of solution image as
%                                 proposed by Glindemann, Dainty
%                       1,'pow' - ties solution image's power spectrum to
%                                 a fixed, calculated power spectrum given
%                                 as argument 8      
%           reg_param - necessary parameters for taking the J^T*J for the 
%                       approximate Hessian
%                       0,'pos' --> indices of negative image values for 
%                                   Hessian of regularizer
%                       1,'pow' --> image necessary for misfit J^T*J
%
%   Outputs:      Hpk - vector of Hessian operator in the direction pk 
%

% First, evaluate the dph_di operator times the direction pk
dims = size(pupil_mask);
IMAGE = IMAGE.*pupil_mask;
inds = find(abs(IMAGE) ~= 0);
zero_inds = find(abs(IMAGE) == 0);

dph_pk = reshape(pk,dims);
dph_pk = fftshift(fft2(fftshift(dph_pk)))/256;
dph_pk(inds) = imag(dph_pk(inds)./IMAGE(inds));
dph_pk(zero_inds) = 0;
dph_pk = dph_pk.*pupil_mask;

% Next, multiply dph_pk times the operator A'*diagonal*A 
ADApk = reshape(ADA*dph_pk(:),dims);

% Lastly, evaluate the adjoint dph_di^* times Hdph_pk
Hpk = zeros(size(ADApk));
Hpk(inds) = ADApk(inds)./IMAGE(inds);
Hpk(zero_inds) = 0;
Hpk = imag(fftshift(fft2(fftshift(Hpk.*pupil_mask))))/256;
Hpk = Hpk(:);

% To finish, add on the Hessian for the regularizer 
if alpha > 0
    switch reg_flag
        case{'pos',0}
            D = sparse(reg_param, reg_param, alpha*ones(length(reg_param),1), prod(dims),prod(dims));
            Hpk = Hpk + D*pk(:);
        case{'pow',1}
            Fim = fftshift(fft2(reg_param));
            reg_term = real(4*reshape(fft2(fftshift(conj(Fim).*real(conj(Fim).*fftshift(fft2(reshape(pk,dims))).*pupil_mask).*pupil_mask)),[],1));
            Hpk = Hpk + alpha*reg_term;
    end
end



end

