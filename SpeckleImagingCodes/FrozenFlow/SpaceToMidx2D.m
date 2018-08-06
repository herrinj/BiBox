function [I, J] = SpaceToMidx2D(X, Y, m, n)
%
%  Convert the Euclidean spatial coordinates given by (X,Y) to
%  MATLAB indices.  That is, assume the image array is situated
%  on the Eucldean axes:
%
%              y ^
%                |
%                |
%                |-------->
%                         x
%
if nargin < 3
  m = size(Y,1);
  n = size(X,2);
elseif nargin < 2
  n = m;
end
I = m - Y;
J = X + 1;
