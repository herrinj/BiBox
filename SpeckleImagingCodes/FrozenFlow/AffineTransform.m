function T = AffineTransform(n_ap, n_frames, deltax, deltay)
%
%    T = AffineTransform(n_ap, n_frames, deltax, deltay);
%
%  This function generates affine tranformations from given motion data.
%
%  Input: 
%   n_ap       - number of pixels across the aperture (diameter)
%   n_frames   - number of frames
%   deltax     - motion information of each of the layers; number of
%                high resolution pixels shifting in x - direction.
%   deltay     - motion information of each of the layers; number of
%                high resolution pixels shifting in y - direction.
%
%  Output:
%   T          - affine tranformations
%

%
%  Remark: We could include rotation in this as well, but for now we
%          only implement only shifts. If later we want to include
%          rotations, then we will need to edit this code as follows:
%          * include an additional input: delta_theta
%          * uncomment the lines below that pertain to rotation
%

n_layers = size(deltax,1);

if size(deltax,2) > 1
  % The velocity is not constant, so this should be n_frames-1 in
  % length
  if size(deltax,2) ~= n_frames-1
    error('incorrect deltax for nonconstant velocity')
  end
  deltax_vec = deltax;
else
  deltax_vec = deltax(:,ones(1,n_frames-1));
end
if size(deltay,2) > 1
  % The velocity is not constant, so this should be n_frames-1 in
  % length
  if size(deltay,2) ~= n_frames-1
    error('incorrect deltay for nonconstant velocity')
  end
  deltay_vec = deltay;
else
  deltay_vec = deltay(:,ones(1,n_frames-1));
end
%----------------------------------
% ROTATION STATMENTS:
%
% If we edit this code to allow for rotation, uncomment the following:
%
%if size(delta_theta,2) > 1
%  % The velocity is not constant, so this should be n_frames-1 in
%  % length
%  if size(delta_theta,2) ~= n_frames-1
%    error('incorrect delta_theta for nonconstant velocity')
%  end
%  delta_theta_vec = delta_theta;
%else
%  delta_theta_vec = delta_theta(:,ones(1,n_frames-1));
%end
%
%----------------------------------

T = zeros(3, 3, n_frames, n_layers);

for L = 1:n_layers
  %
  %  The first frame is the reference frame, so the affine transformation
  %  should be the identity.
  %
  T(:,:,1,L) = eye(3);
  
  for k = 2:n_frames
    dx = sum(deltax_vec(L,1:k-1)); % the shifting = integral of velocity * delta_time
    dy = sum(deltay_vec(L,1:k-1));
    %----------------------------------
    % ROTATION CODES: If we want to use rotations, uncomment this line:
    %
    % theta = sum(delta_theta_vec(L,1:k-1));
    %
    %----------------------------------
        
    %
    % Get spatial coordinates of center points of pixels:
    %   Note: If we assume the image is defined on a standard Euclidean
    %         coordinate system:
    %
    %              y ^
    %                |
    %                |
    %                |-------->
    %                         x
    %
    %          then an affine
        
    %
    % Now shift the coordinate system:
    % Create affine transformation to do the shifting of coordinates:
    % Note: According to our coordinate system, dx > 0 means move right,
    %       and dy > 0 means moves up.
    %
    % The affine transformation for this is:
    %
    T_shift = [1 0 0; 0 1 0; -dx -dy 1];

    %----------------------------------
    % ROTATION CODES: If we want to include rotation, uncomment these lines
    %
    % % Now consider a rotation about the center of the image.
    % %  Rotation is clockwise by an angle theta.
    % %  and that the center of rotation is the point ((n+1)/2, (m+1)/2)
    % %  where the image is m-by-n pixels.
    % %
    % %  Note that standard rotation is about (0,0).  So we first need
    % %  to shift the center of the image to the coordinate (0,0), then
    % %  rotate, then shift back.
    % %
    % %  Shfit to (0,0):
    % %
    % SL = [1 0 0;0 1 0;-(n-1)/2 -(n_ap-1)/2 1];
    % %  Rotate by angle theta.
    % TR = [cos(theta), sin(theta), 0;-sin(theta), cos(theta), 0; 0, 0, 1];
    % %  Now shift back.
    % SR = [1 0 0;0 1 0; (n-1)/2,  (m-1)/2, 1];
    % % The combined transformation is:
    % T_rot_center = SL*TR*SR;
    %
    %----------------------------------
    %
    % ROTATION CODES: If we want to includ rotation, comment out this
    %                 next line:
    %
    T_rot_center = eye(3);
    %
    %----------------------------------
    %
    %  Finally, the combined tranformation that shifts and rotates about
    %  center is:
    %
    T(:,:,k,L) = T_shift * T_rot_center;
    %   
  end
end
