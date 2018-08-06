function [X, Y] = GetPixelCenters2D(m,n)
%
%           [X, Y] = GetPixelCenters(m,n);
%
%  Find the (x,y) coordinates of the centers of pixels of an image.
%  NOTE:  Here it is assumed that the image boundaries are defined by the
%         standard Euclidean coordinate system:
%              y ^
%                |
%                |
%                |-------->
%                         x
%         and the centers of the pixels are given at (x,y), where
%         x = 0, 1, 2, ..., n-1, 
%         y = 0, 1, 2, ..., m-1
%
%  Input:  m, n = dimension of the image
%
%  Output: X = x-coordinates of centers of the pixels
%          Y = y-coordinates of centers of the pixels
%

[X, Y] = meshgrid(0:n-1, m-1:-1:0);