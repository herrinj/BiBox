function [n_comp, n_comp_pad] = GetCompositeGridSize(n_ap, n_frames, deltax, deltay)
%
%  [n_comp, n_comp_pad] = GetCompositeGridSize(n_ap, n_frames, deltax, deltay);
%  
% This function computes the grid size needed to store the composite
% high resolution gradients and phases.
%
% Input:
%   n_ap       - number of pixels across the aperture (diameter)
%   n_frames   - number of frames
%   deltax     - motion information of each of the layers; number of
%                high resolution pixels shifting in x - direction.
%   deltay     - motion information of each of the layers; number of
%                high resolution pixels shifting in y - direction.
%
% Output:
%   n_comp     - composit grid size
%   n_comp_pad - amount the aperture needs to be padded to get to the
%                composite grid size
%

%
%  J. Nagy
%  August, 2012
%

n_comp_pad = max( max( ceil( abs([deltax, deltay])*n_frames ) ) );
n_comp = n_ap + 2*n_comp_pad;


