function [err, NoiseInfo] = IRnoise(b, kind, level)
%
% Notes: no seed is used! The level is RELATIVE!
%
% Initialization: set the default inputs
default_kind = 'gauss';
default_level = 0.01;
%
switch nargin
    case 0
        error('Not enough input arguments')
    case 1
        kind = []; level = [];
    case 2
        level = [];
    case 3
    otherwise
        error('Too many input arguments')
end
%
% Note: b might be empty
if isempty(kind)
    kind = default_kind;
end
if isempty(level)
    level = default_level;
end
n=length(b);
if strcmp(kind, 'gauss')
    r=randn(n,1);
    err = ((level*norm(b))/norm(r))*r;
elseif strcmp(kind, 'poisson')
    err = poissonNoise(b,level);    
else
    error('Invalid noise type')
end
NoiseInfo.kind = kind;
NoiseInfo.level = level;
%
% Subfunctions
%
function err = poissonNoise(inarray,level)
%
if nargin <  1, 
    error('Requires at least one input argument.'); 
end

thresh=32;

out=inarray;

% High-count pixels - use Gaussian approach
gtthresh=find(inarray>thresh);
if ~isempty(gtthresh),
        out(gtthresh)=inarray(gtthresh) + ...
sqrt(inarray(gtthresh)).*randn(size(inarray(gtthresh)));
        out(gtthresh)=round(max(0,out(gtthresh)));
end

% Low-count pixels - this goes into the counting-experiment method

ltthresh=find(inarray<=thresh);
if length(ltthresh)>0  % segregate low-value pixels to speed computation
        lamda=inarray(ltthresh); 
     % Now dealing with 1-D column vector to merge into n-D array out later on
     % Initialize r to zero.
        r = zeros(size(lamda));  % output array for ltthresh pixels
        p = zeros(size(lamda));
        done = ones(size(lamda));
        
        while any(done) ~= 0 % note, do repeatedly calculate over all of lamda
            p = p - log(rand(size(lamda)));  
            kc = [1:length(lamda)]';
            k = find(p < lamda); % Q: does this k index over 
            if any(k)
                r(k) = r(k) + 1;
            end
            kc(k) = [];
            done(kc) = zeros(size(kc));
        end
            
% Return NaN if lamda not positive -- to do this, un-comment what 
% follows (gives zero now).      
%       tmp = NaN;
%       if any(any(any(lamda <= 0)));
%           if prod(size(lamda) == 1),   % i.e., a single pixel?
%               r = tmp(ones(size(lamda)));
%           else
%               k = find(lamda <= 0);
%               r(k) = tmp(ones(size(k)));
%           end
%       end

% Merge low-value-pixel results with large-value-pixel results
        out(ltthresh)=r;  
        err = inarray-out;
        err = (err/norm(err))*level*norm(inarray);
        
end % of if length(ltthresh)>0

