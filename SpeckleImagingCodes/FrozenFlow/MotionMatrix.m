function A = MotionMatrix(deltax, deltay, n_comp, n_ap, n_frames)
%
%     A = MotionMatrix(deltax, deltay, n_comp, n_ap, n_frames);
%
%  Use the motion information, as given by deltax and deltay,
%  to create a sparse matrix that, when multiplied to the large 
%  global wavefront (or gradient fields) will move them to get the
%  wavefronts (or gradient fields) in the position corresponding to
%  each frame.
%
%  Input:
%   deltax     - motion information of each of the layers; number of
%                high resolution pixels shifting in x - direction.
%   deltay     - motion information of each of the layers; number of
%                high resolution pixels shifting in y - direction.
%   n_comp     - composit grid size
%   n_ap       - number of pixels across the aperture (diameter)
%   n_frames   - number of frames
%
%  Output:
%   A          - sparse matrix
%

%
%  J. Nagy
%  August, 2012
%
n_layers = size(deltax,1);

%
%  Get affine transformations for the motion:
%
T = AffineTransform(n_ap, n_frames, deltax, deltay);

A = [];
for L = 1:n_layers 
  %
  % Get pixel centers of composite image.
  %
  [X, Y] = GetPixelCenters2D(n_comp, n_comp);

  S = cell(n_frames,1);
  S2 = cell(n_frames,1);
  %h = waitbar(0, sprintf('Building motion matrices for layer %d', L));
  for k = 1:n_frames
    
    %
    %  Use affine transfromation to transfrom coordinates:
    %
    [Xnew, Ynew] = TransformCoordinates2D(T(:,:,k,L), X, Y); 
    
    %
    %  Get MATLAB indices corresponding to these new coordinates:
    %
    [I, J] = SpaceToMidx2D(Xnew, Ynew);
    
    %
    % Now build the sparse matrix that does the geometric
    % transformation on the image.
    %
    S{k} = BuildInterpMatrix(I, J, n_comp, n_comp);  
    %waitbar(k/n_frames)
  end
  AA = cat(1,S{:});
  %close(h)
  A = [A, AA];
end
