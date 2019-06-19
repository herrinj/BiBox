function [nfr, D_r0, image_name, K_n, sigma_rn, fourier_rad, second_rad] = setupBispectrumParams(varargin)
%
% This function sets the parameters necessary to run the script
% setupBispectrumData. The number of inputs can vary to override the
% default parameters
%

% Default parameters
nfr = 50; % number of data frames
D_r0 = 30; % D/r0 value (10,20,30,40,50...)
image_name = 'Satellite';
K_n = 3e6; % number of photoevents per frame (estimate for simulated data)
sigma_rn = 5; % standard deviation of zero-mean white Gaussian read noise from CCD device
fourier_rad = 96;
second_rad  = 5;

for k=1:2:length(varargin)
   eval([varargin{k}, '=varargin{',int2str(k+1),'};']); 
end
