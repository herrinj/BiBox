function W = WindowMatrix(n_ap, n_comp, pupil_mask, row_start, col_start)
%
%  W = WindowMatrix(n_ap, n_comp, pupil_mask, row_start, col_start);
%
%  This function constructs a sparse matrix that operates as a mask 
%  to grab a certain region from an image array.
%
%  Input:
%    n_ap       - size of high resolution image domain at telescope 
%                       (e.g., 128-by-128)
%    n_comp     - size of large (global, or composite) image
%    pupil_mask - pupil mask of telescope
%    row_start  -  These tell were the rectangular winow, which contains
%    col_start  /  the pupil mask, should begin in the composite grid.
%
%  Output:
%    W          - sparse matrix
%

%
%  J. Nagy
%  August, 2012
%

jw1 = kron((0:n_ap-1)',n_comp*ones(n_ap,1));
%jw2 = kron(ones(n_ap,1),(1:n_ap)');
t1 = n_comp*(col_start-1)+row_start;
jw2 = kron(ones(n_ap,1),(t1:t1+n_ap-1)');
jw = jw1+jw2;
iw = (1:n_ap*n_ap)';
W = sparse(iw, jw, pupil_mask(:), n_ap*n_ap, n_comp*n_comp);
%Wh = sparse(iw, jw, ones(n_ap*n_ap,1), n_ap*n_ap, n_comp*n_comp);
