function [yc,his] = GaussNewtonProj(fctn,y0,varargin)
%
% function [yc,his] = GaussNewtonProj(fctn,y0,varargin)
%
% Projected Gauss-Newton with Projected Armijo line search for minimizing 
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
   fctn = @Rosenbrock;
   
   % Solve with no bounds
   [y_nb,his] = GaussNewtonProj(fctn, [4;2],'verbose',0);
   
   % Solve with lower bound
   lower_bound = [1.1; -1.0];
   y_lb = GaussNewtonProj(fctn, [4;2],'verbose',0, 'lower_bound', lower_bound);

   % Solve with upper bound
   upper_bound = [4.0; 0.9];
   y_ub = GaussNewtonProj(fctn,[0;-1],'verbose',0, 'upper_bound', upper_bound);
   
   % Solve with both
   y_bb = GaussNewtonProj(fctn,[4.0; 0.0],'verbose',0, 'lower_bound', lower_bound, 'upper_bound', upper_bound);
   
   
   fprintf('numerical solution: y = [%1.4f, %1.4f]\n',y_nb);
   fprintf('numerical solution: y_lb = [%1.4f, %1.4f]\n',y_lb);
   fprintf('numerical solution: y_ub = [%1.4f, %1.4f]\n',y_ub);
   fprintf('numerical solution: y_bb = [%1.4f, %1.4f]\n',y_bb);

   
   [X,Y] = ndgrid(linspace(-1,2,101));
   F = reshape(fctn([X(:),Y(:)]'),size(X));
   figure; 
   contour(X,Y,F,200); hold on;
   plot(y_nb(1),y_nb(2)  ,'ro');
   plot(y_lb(1),y_lb(2),'rd');
   plot(y_ub(1),y_ub(2),'r*');
   plot(y_bb(1),y_bb(2),'rh');
   
   yc   = [];
   his  = [];
   return;
end

% Gauss-Newton parameters
maxIter      = 50;
yStop        = [];
Jstop        = [];
paraStop     = [];
tolJ         = 1e-4;            % stopping tolerance, objective function
tolY         = 1e-4;            % stopping tolerance, norm of solution
tolG         = 1e-4;            % stopping tolerance, norm of gradient

% Gauss-Newton step solve parameters
solver       = [];             
solverMaxIter= 50;              
solverTol    = 1e-1;

% Bound constraints (do not work with complex variables)
upper_bound  = Inf*ones(numel(y0),1);   
lower_bound  = -Inf*ones(numel(y0),1);

% Line search parameters
lsMaxIter    = 10;           % maximum number of line search iterations
lsReduction  = 1e-4;         % reduction constant for Armijo condition
lineSearch   = @proj_armijo; % Could potentially call to other projected LS

% Method options
verbose      = true;         % flag to print out
iterSave     = 0;            % flag to save iterations
iterVP       = 0;            % flag to save VarPro linear variables from para structure
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
hisArray     = zeros(maxIter+1,7);
hisStr       = {'iter','J','Jold-J','|proj_dJ|','|dy|','LS','Active'};

yc = y0;
active = (yc <= lower_bound)|(yc >= upper_bound);
[Jc,para,dJ,op] = fctn(yc); 
proj_dJ = proj_grad(dJ, yc, lower_bound, upper_bound);
Plots('start',para);
iter = 0; yOld = 0*yc; Jold = Jc;
hisArray(1,:) = [0 , Jc, Jc, norm(proj_dJ), norm(y0), 0, sum(active>0)];


% Save iterates
if iterSave
    if iterVP
        iterArray      =  zeros(numel(para.Tc) + numel(yc), maxIter+1);
        iterArray(:,1) = [para.Tc(:); yc];
    else
        iterArray      =  zeros(numel(yc), maxIter+1);
        iterArray(:,1) = yc;
    end
end

% Print stuff
if verbose
    fprintf('%s %s %s\n',ones(1,20)*char('='),mfilename,ones(1,20)*char('='));
    fprintf('[ maxIter=%s / tolJ=%s / tolU=%s / tolG=%s / length(yc)=%d ]\n',...
    num2str(maxIter),num2str(tolJ),num2str(tolY),num2str(tolG),length(yc));
    fprintf('%4s %-12s %-12s %-12s %-12s %4s %-8s \n %s \n', hisStr{:},char(ones(1,64)*'-'));
    dispHis = @(var) fprintf('%4d %-12.4e %-12.3e %-12.3e %-12.3e %4d %-8d \n',var);
    dispHis(hisArray(1,:));
end

% Start projected Gauss-Newton iteration
while 1
   
    % Check stopping criteria
    stop(1) = (iter>0) && abs(Jold-Jc)   <= tolJ*(1+abs(Jstop));
    stop(2) = (iter>0) && (norm(yc-yOld) <= tolY*(1+norm(y0)));
    stop(3) = norm(proj_dJ)              <= tolG*(1+abs(Jstop));
    stop(4) = norm(proj_dJ)              <= 1e6*eps;
    stop(5) = (iter >= maxIter);
    if (all(stop(1:3)) || any(stop(4:5)))
        break;  
    end
    
    iter = iter+1;  
    
    % Gauss-Newton step on inactive set
    [dy_in, solverInfo] = stepGN(op,-dJ,solver, 'active', active, 'solverMaxIter', solverMaxIter, 'solverTol', solverTol);
    
    % Pull out updated gradient
    if isfield(solverInfo,'dJ')
        dJ = solverInfo.dJ;
    end
    
    % Projected gradient descent on active set
    dy_act = zeros(size(yc));
    dy_act(yc == lower_bound) = -dJ(yc == lower_bound); 
    dy_act(yc == upper_bound) = -dJ(yc == upper_bound);    
    
    % Combine the steps
    if sum(active)==0
        nu = 0;
    else
        nu = max(abs(dy_in))/max(abs(dy_act)); % scaling factor (Haber, Geophysical Electromagnetics, p.110)
    end
    dy = dy_in + nu*dy_act;
    
    % Line search
    [yt, exitFlag, lsIter] = lineSearch(fctn, yc, dy, Jc, proj_dJ, lower_bound, upper_bound, lsMaxIter, lsReduction); 
    
    % Save old values and re-evaluate objective function
    yOld = yc; Jold = Jc; yc = yt;
    active = (yc <= lower_bound)|(yc >= upper_bound);      % update active set
    [Jc,para,dJ,op] = fctn(yc);                              % evalute objective function
    proj_dJ = proj_grad(dJ, yc, lower_bound, upper_bound);  % projected gradient
  
    % Some output
    hisArray(iter+1,:) = [iter, Jc, Jold-Jc, norm(proj_dJ), norm(yc-yOld), lsIter, sum(active>0)];
    if verbose
        dispHis(hisArray(iter+1,:));
    end
    if iterSave
        if iterVP
            iterArray(:,iter+1) = [para.Tc(:); yc];
        else
            iterArray(:,iter+1) = yc;
        end
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
    '|dJ|',norm(proj_dJ),'tolG*(1+abs(Jstop))',tolG*(1+abs(Jstop)));
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(4),...
    'norm(dJ)',norm(proj_dJ),'eps',1e3*eps);
    fprintf('%d[ %-10s=  %-14d >= %-25s=  %-14d]\n',stop(5),...
    'iter',iter,'maxIter',maxIter);

    %FAIRmessage([mfilename,' : done !'],'=');
end
    
end

function [yt, exitFlag, iter] = proj_armijo(fctn, yc, dy, Jc, proj_dJ, lower_bound, upper_bound, lsMaxIter, lsReduction) 
%
%   This function peforms a projected Armijo line search obeying the bounds
%   
%   Input:      ftcn - objective function from proj. Gauss-Newton iteration
%                 yc - current iterate
%                 dy - update direction for currant iterate
%                 Jc - current objective function value
%            proj_dJ - projected gradient
%       
%          lsMaxIter - maximum number of line search iterations
%        lsReduction - required reduction for Armijo condition
%   
%   Output:       yt - updated iterate yc + t*dy
%           exitFlag - flag, 0 failure, 1 success
%               iter - number of line search iterations
%

t = 1; % initial step 1 for Gauss-Newton
iter = 1;
cond = zeros(2,1);

while 1
    yt = yc + t*dy;
    active = (yt <= lower_bound)|(yt >= upper_bound);
    if sum(active)>0
        yt = min(max(yt,lower_bound),upper_bound); % Only impose bounds on the image
    end
    Jt = fctn(yt);
    
    % check Armijo condition
    cond(1) = (Jt<Jc + t*lsReduction*(reshape(proj_dJ,1,[])*dy));
    cond(2) = (iter >=lsMaxIter);
    
    if cond(1)
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

function [proj_dJ] = proj_grad(dJ, yc, lower_bound, upper_bound)
% 
%   This function projects the gradient
%
%   Input:      dJ - unprojected gradient
%               yc - current image and motion parameters to detect which 
%                    variables are on the bounds
%      lower_bound - vector containing elementwise lower bounds on yc
%      upper_bound - vector containint elementwise upper bounds on yc
%
%   Output:
%          proj_dJ - projected gradient
%

proj_dJ = dJ(:);
proj_dJ(yc == lower_bound) = min(dJ(yc == lower_bound),0);
proj_dJ(yc == upper_bound) = max(dJ(yc == upper_bound),0);

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
