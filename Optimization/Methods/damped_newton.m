function [x,flag,his,X] = damped_newton(obj_func, x0, maxIter, tol, verbose)
%
%   [x,flag,his,X] = damped_newton(obj_func, x0, maxIter, tol, verbose)
%
%   Damped Newton method 
%
%   Inputs:
%           obj_func - function handle for evaluation of objective function
%                      f and gradient df
%                 x0 - initial guess for method
%            maxIter - default 20
%                tol - default 1e^-8
%            verbose - default 0 (off) change to 1 to print iteration
%                      information
%
%
%   Outputs: x - solution vector
%            flag - indicates how method stops
%            his - matrix containing info about method iterations
%                  his(:,1) - objective function at each iteration
%                  his(:,2) - abs. value of change in obj. function divided
%                             by the size of x
%                  his(:,3) - norm of the gradient at each iteration
%                  his(:,4) - norm of change in gradient from previous
%                             iteration
%                  his(:,5) - number of line search iterations per method
%                             iteration
%                  his(:,6) - tic/toc seconds per iteration
%            X - iteration history where X(:,i) is x for the ith iteration
%

% Need dimension of the problem
if isempty(x0)
    fprintf('Please supply initial guess x0 of correct dimension \n');
    return;
end
   
% Define necessary method parameters
if isempty(maxIter)
    maxIter = 50;
end

if isempty(tol)
    tol = 1e-8;
end

if isempty(verbose)
    verbose = 0;
end

if nargout == 4
    saveIter = 1;
else 
    saveIter =0;
end

his = zeros(maxIter,5);
if saveIter
    X = zeros(length(x0(:)), maxIter+1);
end

i=1; flag = -1; x = x0(:);
[~,df_old] = feval(obj_func,x);
fc_old = 0;

while i<=maxIter
    tic;
    [fc, df, d2f] = feval(obj_func, x);
    his(i,1:4) = [fc abs(fc_old - fc)/prod(size(x)) norm(df) norm(df - df_old)];
    
    if saveIter
        X(:,i) = x;
    end

    % Stopping criteria is norm of gradient
    if norm(df) < tol || abs(fc_old - fc)/prod(size(x)) < tol
        flag = 0;
        his = his(1:i,:);
        break;
    end
        
    % Get search direction by Hessian solve   
    pk  = newton_hess_solve(d2f, -df, verbose); % needs to handle all types of Hessian
    
    % Armijo line search to find step size a_k
    [ak,his(i,5)] = armijo(obj_func,fc,df,x,pk); 
    if his(i,5) == -1
        flag = -2;
        his  = his(1:i,:);
        break;
    end
    
    if verbose
        fprintf('iter=%04d\t |f|=%1.2e\t |f_old - f|=%1.2e\t |df|=%1.2e\t |df - df_old|=%1.2e\t LS=%d \n', i, his(i,1), his(i,2), his(i,3), his(i,4), his(i,5));
    end 
    
    % Update x
    x = x + ak*pk;
    his(i,6) = toc;
    i = i + 1;   
    df_old = df;
    fc_old = fc;
end

i = min(maxIter,i);

% Print the reason for method termination
if verbose
    switch flag 
        case{0}
            fprintf('Damped Newton achieved desired tolerance of tol=%1.2e at iteration %d. \n',tol,i);
        case{-1}
            fprintf('Damped Newton performed the maximum number of iterations (=%d) but did not reach tol=%1.2e. \n',i,tol);
        case{-2}
            fprintf('Damped Newton stopped due to a failed line search at iteration %d. \n',i);
    end
    fprintf('Total time elapsed was %1.2e secs with %1.2e secs/iter. \n', sum(his(:,6)), sum(his(:,6))/i);
end

if saveIter
    X = X(:,1:i); 
end

end