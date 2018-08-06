function S = BuildInterpMatrix(I, J, m, n)
%
%  Given pixel coordinates, (I,J) = (row, col), which
%  are not necessarily integers, this function computes
%  an interpolation matrix S so that y = S*x interpolates
%  pixel values in the image x to give pixel values in
%  the image y.
%
%  Here we use linear interpolation.
%
%  Input:
%    I and J are arrays of size m-by-n (same as image dimensions).
%       These contain coordinates of an image transformation.
%    m, n is the size of the image.
%
%  Output:
%    S is a sparse matrix
%

% First find integer coordinates that surround each
% (I(i,j),J(i,j))
% These will be the bilinear interpolation coordinates.
%
i0 = floor(I(:));
i1 = ceil(I(:));
j0 = floor(J(:));
j1 = ceil(J(:));

i1 = i0+1;
j1 = j0+1;

%
% To avoid playing games with indices, we are very sloppy
% around the border.  Interpolation is only done if ALL of the
% points that surround (I(i,j),J(i,j)) are within the image
% boarders.  This makes it easier to compute the interpolation
% weights without having to use special conditions near
% the borders.  If we assume black areas near the border,
% then this should not cause any problems.
%
% The first step, then, is to find the rows that will contain
% weights.
%
row_idx = find(1<=i0 & i1<=m & 1<=j0 & j1<=n);
i = [row_idx; row_idx; row_idx; row_idx];

% Since we only consider interior pixel values, then each
% row will have exactly four weights.  So next we find the
% four column indices where these weights should be put.
% We are assuming column ordering of the image.
%
i0 = i0(row_idx);, i1 = i1(row_idx);, j0 = j0(row_idx);, j1 = j1(row_idx);

col_idx1 = i0 + m*(j0-1);
col_idx2 = i1 + m*(j0-1);
col_idx3 = i0 + m*(j1-1);
col_idx4 = i1 + m*(j1-1);
j = [col_idx1; col_idx2; col_idx3; col_idx4];

%
% Now we compute the weights that go into the matrix.
%
deltai = I(row_idx) - i0;
deltaj = J(row_idx) - j0;
w1 = (1 - deltai).*(1 - deltaj);
w2 = (1 - deltaj).*deltai;
w3 = (1 - deltai).*deltaj;
w4 = deltai.*deltaj;
s = [w1; w2; w3; w4];


%
% Now let's put the weights in the sparse matrix.
%
S = sparse(i, j, s, m*n, m*n);


