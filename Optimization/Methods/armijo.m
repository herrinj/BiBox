function [t,LS] = armijo(func,fc,df,xc,pc,maxIter,b)
%
%   Armijo line search
%
%   Inputs: func - function handle for function evals
%             fc - current objective function value
%             df - current gradient value
%             xc - current x value
%             pc - current search direction
%        maxIter - maximum number of line search iterations 
%              b - backtracking ratio for the Armijo search
%
%   Outputs:  t - step size
%            LS - flag/iter, LS > 1 indicates number of line search iterations,
%                            LS = -1 indicates line search failure
%

if nargin < 6
   maxIter = 100;
end

c1 = 1e-4; 

if nargin < 7
    b = 0.5; % Backtracking ratio for Armijo
end
    
LS = 1; 
t  = 1;

while LS<=maxIter
    ft = feval(func, xc + t*pc);
    if ft <= fc + t*c1*(df'*pc)
        break
    end
    t = t*b;
    LS = LS + 1;
end
if LS>maxIter
	LS= -1;
	t = 0.0;
end
    return
end