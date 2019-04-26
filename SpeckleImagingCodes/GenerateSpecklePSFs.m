function [phase_frames, PSF_frames, pupil_mask, scaled_pupil_mask] = GenerateSpecklePSFs(n, n_frames, D_over_r0, seed)
%
%  This function will generate some speckle imaging PSF test data.
%
%  Input:            
%             n - size of the image, in pixels (assumed to be square),
%                 for example, n = 256.  Default is n = 256;
%      n_frames - number of desired frames of data.  Default is 
%                 n_frames = 30;
%     D_over_r0 - defines seeing conditions (amount of blurring).
%                 This should be set to one of the following values:
%                    1, 5, 10, 15, 20, 30, 40, 50, 70.
%                 Default is 30.
%          seed - specifies the seed for the random number generator. If
%                 empty or nargin<4, seed is set to 0;
%
%  Output: 
%          phase_frames - three dimensional array, where 
%                         phase_frames(:,:,k) is the phase for frame k.
%          PSF_frames   - three dimensional array, where 
%                         PSF_frames(:,:,k) is the PSF for frame k.
%          pupil_mask   - unscaled pupil mask
%     scaled_pupil_mask - scaled pupil mask -- this is what is used
%                              to construct the PSFs.
%

switch nargin
    case 0
        n = []; n_frames = []; D_over_r0 = [];
    case 1
        n_frames = []; D_over_r0 = [];
    case 2
        D_over_r0 = [];
end
if isempty(n), n = 256; end
if isempty(n_frames), n_frames = 30; end
if isempty(D_over_r0), D_over_r0 = 30; end

%
% Set some paths to codes we'll need to generate the data.
% The SymmWavefrontGeneration codes were written mainly by Qing Chu,
% with some modifications made by Jim Nagy.
% The FrozenFlow codes were written by Jim Nagy.
%
% path(path, 'SymmWavefrontGeneration/');
% path(path, 'FrozenFlow/');

if isempty(seed)
    rng(0);
else
    rng(seed);
end

%
%  We'll assume the aperture diameter (in number of pixels) is half the
%  size of the image size.  So if we have a 256-by-256 image, we'll use
%  an aperture size of 128-by-128.
%
%%%n_ap = n/2;
n_ap = n/2;
if n_ap ~= fix(n_ap)
    error('Use even integers for input n')
end

%
%  Need to define some things define a frozen flow wavefront.  We'll move
%  the flow across the telescope aperture, and then extract the various
%  wavefronts.
%  First we need information to define the wind speed, and then convert 
%  these to deltax and deltay shifts.
%
r = 3;
theta = pi/4;
wind_vecs = [r theta];
deltax = wind_vecs(:,1) .* cos(wind_vecs(:,2));
deltay = wind_vecs(:,1) .* sin(wind_vecs(:,2));

switch D_over_r0
  %
  %  Using the default D (here D2) = 0.5, then:
  %                   Cn2 = 3.8023e-18 should give D_over_r0 = 1
  %                   Cn2 = 5.5590e-17 should give D_over_r0 = 5
  %                
  %                 Note: D_over_r0 = 5 corresponds to good seeing
  %                 conditions.
  %
  %                   Cn2 = 1.7649e-16 should give D_over_r0 = 10
  %                   Cn2 = 3.4690e-16 should give D_over_r0 = 15
  %                   Cn2 = 5.6031e-16 should give D_over_r0 = 20
  %
  %                 Note: At least according to to one reference,
  %                       D_over_r0 = 20 corresponds to very poor seeing 
  %                       conditions -- beyond the capabilities of many AO 
  %                       systems to correct.
  %
  case 1,  Cn2 = 3.8023e-18;
  case 5,  Cn2 = 1.09504e-18;
  case 10, Cn2 = 3.4765e-18;
  case 15, Cn2 = 6.8332e-18;
  case 20, Cn2 = 1.10371e-17;
  case 30, Cn2 = 2.16941e-17;
  case 40, Cn2 = 3.50407e-17;
  case 50, Cn2 = 5.08265e-17;
  case 70, Cn2 = 8.90505e-17;
  otherwise
    warning('invaled D/r0 term. Using default value of D/r0 = 30.')
    Cn2 = 2.16941e-17;
end

%
%  We need to find a composite grid size, and the amount of padding needed
%  to get from n_ap to n_comp.  This will give the size of the global
%  wavefront, which will then flow across the aperture.
%
[n_comp, n_comp_pad] = GetCompositeGridSize(n_ap, n_frames, deltax, deltay);

[phase_layers, ~] = GenerateWavefront(n_comp, [], Cn2);
phase_global = phase_layers{1};

pupil_mask0 = MakeMask(n_ap,1);
pupil_mask = padarray(pupil_mask0, [n_ap/2, n_ap/2], 'both');
%
% pupil_mask0 is an n_ap -by- n_ap circle mask, which comes to the
% boundaries of this array.  If you want to use this to build an 
% n-by-ny PSF, you need to pad this to size n-by-n.  See below.
%
% These next matrices will be used to move the global wavefront across
% the aperture, and extract the individual phases for each frame.
%
W = WindowMatrix(n_ap, n_comp, pupil_mask0, n_comp_pad+1, n_comp_pad+1);
A = MotionMatrix(deltax, deltay, n_comp, n_ap, n_frames);
phase_frames_vec = kron(speye(n_frames),W)*A*phase_global(:);
phase_frames0 = zeros(n_ap, n_ap, n_frames);
phase_frames = zeros(n, n, n_frames);
PSF_frames = zeros(n, n, n_frames);

% To get the normalized PSF (so that the pixel values sum to one), 
% normalize the pupil_mask.  
%
scaled_pupil_mask = n*pupil_mask/sqrt(sum(pupil_mask(:)));
for k = 1:n_frames
    idx_start = n_ap*n_ap*(k-1)+1;
    idx_end   = n_ap*n_ap*k;
    phase_frames0(:,:,k) = reshape(phase_frames_vec(idx_start:idx_end),n_ap,n_ap);
    phase_frames(:,:,k) = padarray(phase_frames0(:,:,k), [n_ap/2, n_ap/2], 'both');
    PSF_shifted = abs( ifft2(scaled_pupil_mask.*exp(sqrt(-1)*phase_frames(:,:,k))) ).^2;
    PSF_frames(:,:,k) = fftshift(PSF_shifted);
end


