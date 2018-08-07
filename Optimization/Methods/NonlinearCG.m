function [yc,his] = NLCG(fctn, y0, varargin)
%
% function [yc,his] = NLCG(fctn,y0,varargin)
%
% Nonlinear conjugate gradient with armijo line search for minimizing 
% J = fctn(yc)
% 
% Input:
%   fctn      - function handle
%   yc        - starting guess (required), should be within feasible region
%   varargin  - optional parameter, see below
%
% Output:
%   yc        - numerical optimizer (current iterate)
%   his       - iteration history
%
%==============================================================================

if nargin==0
   runMinimalExample;
   return;
end

% Gradient descent parameters
maxIter      = 100;
yStop        = [];
Jstop        = [];
paraStop     = [];
tolJ         = 1e-4;            % stopping tolerance, objective function
tolY         = 1e-4;            % stopping tolerance, norm of solution
tolG         = 1e-4;            % stopping tolerance, norm of gradient

% Line search parameters
lsMaxIter    = 100;          % maximum number of line search iterations
lsReduction  = 1e-4;         % reduction constant for Armijo condition
lineSearch   = @armijo;      % Could potentially call to other projected LS

% Method options
verbose      = true;         % flag to print out
iterSave     = 0;            % flag to save iterations
stop         = zeros(5,1);   % vector for stopping criteria
Plots        = @(iter,para) []; % for plots;

% Overwrite default parameters above using varargin
for k=1:2:length(varargin)     
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

% Evaluate objective function for stopping criteria
if isempty(yStop) 
    yStop = y0; 
end

if (isempty(Jstop) || isempty(paraStop))
    [Jstop,paraStop] = fctn(yStop); Jstop = abs(Jstop) + (Jstop == 0); 
    Plots('stop',paraStop);
end

% History output
his          = [];
hisArray     = zeros(maxIter+1,5);
hisStr       = {'iter','J','Jold-J','|dy|','LS'};

yc = y0;
[Jc,para,dJ] = fctn(yc); 
Plots('start',para);
iter = 0; yOld = 0*yc; Jold = Jc; beta = 0; dy = 0*yc;
hisArray(1,:) = [0 , Jc, Jc, norm(y0), 0];


% Save iterates
if iterSave
    iterArray      =  zeros(numel(yc), maxIter+1);
    iterArray(:,1) = yc;
end

% Print stuff
if verbose
    fprintf('%s %s %s\n',ones(1,20)*char('='),mfilename,ones(1,20)*char('='));
    fprintf('[ maxIter=%s / tolJ=%s / tolU=%s / tolG=%s / length(yc)=%d ]\n',...
    num2str(maxIter),num2str(tolJ),num2str(tolY),num2str(tolG),length(yc));
    fprintf('%4s %-12s %-12s %-12s %4s \n %s \n', hisStr{:},char(ones(1,64)*'-'));
    dispHis = @(var) fprintf('%4d %-12.4e %-12.3e %-12.3e %4d \n',var);
    dispHis(hisArray(1,:));
end

% Start projected gradient descent iteration
while 1
   
    % Check stopping criteria
    stop(1) = (iter>0) && abs(Jold-Jc)   <= tolJ*(1+abs(Jstop));
    stop(2) = (iter>0) && (norm(yc-yOld) <= tolY*(1+norm(y0)));
    stop(3) = norm(dJ)              <= tolG*(1+abs(Jstop));
    stop(4) = norm(dJ)              <= 1e6*eps;
    stop(5) = (iter >= maxIter);
    if (all(stop(1:3)) || any(stop(4:5)))
        break;  
    end
    
    iter = iter+1;  
    
    % Step direction
    dy = -dJ(:) + beta*dy(:);
    
    % Line search
    [yt, exitFlag, lsIter] = lineSearch(fctn, yc, dy, Jc, dJ, lsMaxIter, lsReduction); 
    
    % Save old values and re-evaluate objective function
    yOld = yc; Jold = Jc; yc = yt; dJold = dJ;
    beta = (norm(dJ)^2)/(norm(dJold)^2); % Fletcher-Reeves
%     beta  = (reshape(dJ,[],1)*(dJ(:)-dJold(:)))/(norm(dJold)^2); % Polak-Ribiere
    [Jc,para,dJ] = fctn(yc);
      
    
    % Some output
    hisArray(iter+1,:) = [iter, Jc, Jold-Jc, norm(yc-yOld), lsIter];
    if verbose
        dispHis(hisArray(iter+1,:));
    end
    if iterSave
        iterArray(:,iter+1) = yc;
    end
    
    % Exit if line search failed
    if exitFlag==0 
        break;
    end
    
    para.normdY = norm(yc - yOld);
    Plots(iter,para);
end

Plots(iter,para);

% Clean up and output
his.str = hisStr;
his.array = hisArray(1:iter+1,:);
if iterSave
   his.iters = iterArray(:,1:iter+1); 
end

if verbose
    fprintf('STOPPING:\n');
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(1),...
    '(Jold-Jc)',(Jold-Jc),'tolJ*(1+|Jstop|)',tolJ*(1+abs(Jstop)));
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(2),...
    '|yc-yOld|',norm(yc-yOld),'tolY*(1+norm(yc)) ',tolY*(1+norm(yc)));
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(3),...
    '|dJ|',norm(dJ),'tolG*(1+abs(Jstop))',tolG*(1+abs(Jstop)));
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(4),...
    'norm(dJ)',norm(dJ),'eps',1e6*eps);
    fprintf('%d[ %-10s=  %-14d >= %-25s=  %-14d]\n',stop(5),...
    'iter',iter,'maxIter',maxIter);

    %FAIRmessage([mfilename,' : done !'],'=');
end
    
end

function [yt, exitFlag, iter] = armijo(fctn, yc, dy, Jc, dJ, lsMaxIter, lsReduction) 
%
%   This function peforms a projected Armijo line search obeying the bounds
%   
%   Input:      ftcn - objective function from proj. Gauss-Newton iteration
%                 yc - current iterate
%                 dy - update direction for currant iterate
%                 Jc - current objective function value
%                 dJ - current gradient 
%       
%          lsMaxIter - maximum number of line search iterations
%        lsReduction - required reduction for Armijo condition
%   
%   Output:       yt - updated iterate yc + t*dy
%           exitFlag - flag, 0 failure, 1 success
%               iter - number of line search iterations
%
persistent t;
if isempty(t)
    t = 1; % initial step 1 for Gauss-Newton
end
iter = 1;
cond = zeros(2,1);

while 1
    yt = yc + t*dy;
    Jt = fctn(yt);
    
    % check Armijo condition
    cond(1) = (Jt<Jc + t*lsReduction*(reshape(dJ,1,[])*dy));
    cond(2) = (iter >=lsMaxIter);
    
    if cond(1)
        if (iter==1)
            t = 2*t;
        end
        exitFlag = 1;
        break; 
    elseif cond(2)
        exitFlag = 0;
        yt = yc; % No progress
        fprintf('Line search fail: maximum iterations = %d \n',lsMaxIter);
        break;
    end
        
    t = t/2; % sterp reduction factor
    iter = iter+1;
end
end

function [f,para,df,d2f] = Rosenbrock(x)
x = reshape(x,2,[]);
para = [];
f = (1-x(1,:)).^2 + 100*(x(2,:) - (x(1,:)).^2).^2;

if nargout>1 && size(x,2)==1
    df = [2*(x(1)-1) - 400*x(1)*(x(2)-(x(1))^2); ...
        200*(x(2) - (x(1))^2)];
end

if nargout>2 && size(x,2)==1
    n= 2;
    d2f=zeros(n);
    d2f(1,1)=400*(3*x(1)^2-x(2))+2; d2f(1,2)=-400*x(1);
    for j=2:n-1
        d2f(j,j-1)=-400*x(j-1);
        d2f(j,j)=200+400*(3*x(j)^2-x(j+1))+2;
        d2f(j,j+1)=-400*x(j);
    end
    d2f(n,n-1)=-400*x(n-1); d2f(n,n)=200;
end

end

function runMinimalExample
   fctn = @Rosenbrock;
   
   % Solve with no bounds
   [y_nb,his] = GradientDescentProj(fctn, [0.0;0.0],'verbose',1,'maxIter',1000);
   
   fprintf('numerical solution: y = [%1.4f, %1.4f]\n',y_nb);
   
   [X,Y] = ndgrid(linspace(-1,2,101));
   F = reshape(fctn([X(:),Y(:)]'),size(X));
   figure; 
   contour(X,Y,F,200); hold on;
   plot(y_nb(1),y_nb(2)  ,'ro');
   plot(1.0,1.0  , 'r*');

   
end