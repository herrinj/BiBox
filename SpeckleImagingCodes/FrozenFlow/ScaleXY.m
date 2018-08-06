function [Xr, Yr] = ScaleXY(X, Y, s1, s2)
%
%              [Xr, Yr] = ScaleXY(X, Y, s1, s2);
%
%  Scale a given set of (x,y) points by values s1, s2.
%  It is assumed that the points lie in the unit square [0,1]X[0,1]
%
%  Input:  X     = array containing x coordinates of original points
%          Y     = array containing y coordinates of original points
%          s1    = scaling of x-coordinates
%          s2    = scaling of y-coordinates
%
%  Output: Xr = array containing scaled x coordinates
%          Yr = array containing scaled y coordinates
%

%
%  Create affine transformation to do the rotation of coordinates:
%
T = [s1, 0, 0;0, s2, 0; 0, 0, 1];

%
%  Note that we need to shift the center from (0.5,0.5) to (0,0)
%  before scaling, then shift back.
%
SL = [1 0 0;0 1 0; -0.5,  -0.5, 1];
SR = [1 0 0;0 1 0; 0.5,  0.5, 1];

Z = (([X(:), Y(:), ones(length(X(:)), 1)] * SL) * T ) * SR;

Xr = reshape(Z(:,1), size(X));
Yr = reshape(Z(:,2), size(Y));

