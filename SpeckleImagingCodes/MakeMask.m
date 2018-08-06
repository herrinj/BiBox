function [mask] = MakeMask(n, r1, r0)
%
%  [mask] = MakeMask(n, r1, r0)
%  [mask] = MakeMask(n, r1)
%
%  Constructs an n by n mask with outer radius r1.  If only two input
%  arguments are given, the mask is circular.  Otherwise, it is annular
%  with inner radius r0.  The largest circle contained in the n by n square
%  has radius r1 = 1.
%
%  We assume n is an even integer. The central pixel has index
%  (n+1)/2.
%
%  Inputs:
%    n - size of mask to be constructed
%    r1 - outer radius
%    r0 - inner radius, if annular
%
%  Outputs:
%    mask - array of 1s and 0s representing the mask
%
% Author: John Bardsley, Sarah Knepper
% Date Created: 27 September 2009
% Date Last Modified: 27 September 2009
%

h = 2/n;
x = (-1:h:1-h)';
onevec = ones(n,1);
r = sqrt((x*onevec').^2 + (onevec*x').^2);
if nargin == 2
  mask = (r <= r1); 
else
  mask = (r0 <= r) & (r <= r1);
end
