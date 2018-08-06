function [Xnew, Ynew] = TransformCoordinates2D(T, X, Y)
%
%  Transform (x,y) coordinates given by X and Y using the
%  affine transformation given by T.  That is,
%     [xnew, ynew] = [x, y, 1]*T
%

W = [X(:), Y(:), ones(length(X(:)), 1)] * T;

Xnew = reshape(W(:,1), size(X));
Ynew = reshape(W(:,2), size(Y));
