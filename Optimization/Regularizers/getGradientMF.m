function [Rx] = getGradientMF(x,omega,m,flag)
%==========================================================================
%
%   Matrix-free evaluation of R*x where the matrix R is the discrete 
%   gradient operator in d = 1,2,3 dimensions.
%
%   Input:      x - vector to take gradient
%           omega - 2dx1 array with domain of x 
%               m - dx1 array with dimensions of x
%            flag - set to 'transp' or 'notransp'
%
%   Output:    Rx - output vector R*x or R'*x where R is the discrete 
%                   gradient operator in the appropriate dimension
%
% =========================================================================

dim = length(omega)/2;
h     = (omega(2:2:end)-omega(1:2:end))./m;

switch flag
    case 'notransp' % R*x
        x = reshape(x,m);
        switch dim
            case 1
                Rx  = (x(2:end) - x(1:end-1))/h(1);
            case 2
                Rx1 = (x(2:end,:) - x(1:end-1,:))/h(1);
                Rx2 = (x(:,2:end) - x(:,1:end-1))/h(2);
                Rx  = [Rx1(:);Rx2(:)];
            case 3
                Rx1 = (x(2:end,:,:) - x(1:end-1,:,:))/h(1);
                Rx2 = (x(:,2:end,:) - x(:,1:end-1,:))/h(2);
                Rx3 = (x(:,:,2:end) - x(:,:,1:end-1))/h(3);
                Rx  = [Rx1(:);Rx2(:);Rx3(:)];
        end
    case 'transp' % R'*x
        switch dim
            case 1
                xp  = padarray(x,1,'both');
                Rx  = (xp(1:end-1) - xp(2:end))/h(1);
            case 2
                xp1 = reshape(x(1:(m(1)-1)*m(2)),m(1)-1,m(2));
                xp2 = reshape(x((m(1)-1)*m(2)+1:end),m(1),m(2)-1);
                xp1 = padarray(xp1,[1 0],'both');
                xp2 = padarray(xp2,[0 1],'both');
                Rx1 = (xp1(1:end-1,:) - xp1(2:end,:))/h(1);
                Rx2 = (xp2(:,1:end-1) - xp2(:,2:end))/h(2);
                Rx  = Rx1(:) + Rx2(:);
            case 3
                xp1 = reshape(x(1:(m(1)-1)*m(2)*m(3)),m(1)-1,m(2),m(3));
                xp2 = reshape(x((m(1)-1)*m(2)*m(3)+1:(m(1)-1)*m(2)*m(3)+m(1)*(m(2)-1)*m(3)),m(1),m(2)-1,m(3));
                xp3 = reshape(x((m(1)-1)*m(2)*m(3)+m(1)*(m(2)-1)*m(3)+1:end),m(1),m(2),m(3)-1);
                xp1 = padarray(xp1,[1 0 0],'both');
                xp2 = padarray(xp2,[0 1 0],'both');
                xp3 = padarray(xp3,[0 0 1],'both');
                Rx1 = (xp1(1:end-1,:,:) - xp1(2:end,:,:))/h(1);
                Rx2 = (xp2(:,1:end-1,:) - xp2(:,2:end,:))/h(2);
                Rx3 = (xp3(:,:,1:end-1) - xp3(:,:,2:end))/h(3);
                Rx  = Rx1(:) + Rx2(:) + Rx3(:);
        end
end
end


