function [bisp, bisp_phase, snr_weights] = bispec_accum(DATA, bindex)
%
%
%   Input:         DATA - image frames from which to accumulate the 
%                         bispectrum (in Fourier domain)
%                bindex - index structure for vectorizing the bispectrum
%                         accumulation and recursive algorithm
%           
%   Output:        bisp - accumulated ensemble average of the bispectrum 
%                         of data frames 
%           bisp_phase  - accumulated ensemble average of the bispectrum
%                         phase of the data frames
%           snr_weights - SNR weights for least squares objective function 
%                         from Eq.40 in Regagnon
%


u_inds = bindex.u;
v_inds = bindex.v;
uv_inds = bindex.u_v;

% Accumulate the bispectrum using the relationship bisp = IM(u)*IM(v)*conj(IM(u+v))
% over the appropriate indices u,v,u+v. See Eq. 26 in Regagnon, 1996 and others
nfr = size(DATA,3);
DATA = reshape(DATA,[],nfr);
DATA_phase = angle(DATA);
bisp = DATA(u_inds,:).*DATA(v_inds,:).*conj(DATA(uv_inds,:));

R2 = sum(real(bisp).^2,2)/nfr;
I2 = sum(imag(bisp).^2,2)/nfr;
IR = sum(imag(bisp).*real(bisp),2)/nfr;

bisp = sum(bisp,2)/nfr;
bisp_phase = angle(bisp);
%inds = find(bisp_phase <0);
%bisp_phase(inds) = bisp_phase(inds) + 2*pi;


% Calculate the SNR weights
sigma_R = R2 - real(bisp).^2;
sigma_I = I2 - imag(bisp).^2;
cov_IR = IR - imag(bisp).*real(bisp);
beta = angle(sum(bisp)/length(bisp));
snr_weights = abs(bisp)./(sigma_I*cos(beta)*cos(beta) + sigma_R*sin(beta)*sin(beta) - cov_IR*sin(2*beta)).^0.5;


