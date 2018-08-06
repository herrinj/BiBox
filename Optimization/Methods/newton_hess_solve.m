function [pk] = newton_hess_solve(hess, grad, verbose)
%
%   [pk] = newton_hess_solve(hess, grad)
%
%   This function solves the system H*pk = -df for a Newton step based
%   optimization. H is given in the variable 'hess' and is approximated by 
%   the symmetrics J^T*J operator where J is the objective function 
%   Jacobian. 'grad' contains the negative gradient, -df.
%
%   Input: 
%           hess - hess is a variable matrix/struct/function handle which
%                  contains the approximate Hessian operator for the
%                  objective functions phase_rec.m, phasor_rec.m,
%                  imphase_rec.m, imphasor_rec.m
%           grad - negative gradient as a column vector
%        verbose - prints out pertenent information related to Hessian 
%                  solve
%
%   Output:
%             pk - search direction given by the solution so H*pk = -df
%

if isstruct(hess)
   flag = hess.flag;
   switch flag
       case{'full'}
           H = hess.matrix; % full Hessian
           pk = H\grad;
           
       case{'trunc'}
           H = hess.matrix; % truncated, permuted Hessian
           perm = hess.perm;
           dim = hess.dim;
           pk = zeros(length(grad(:)),1);
           grad_perm = grad(perm);
           grad_perm = grad_perm(1:dim);
           pk_perm = H\grad_perm;
           pk(perm(1:dim)) = pk_perm;
       
       case{'ichol'}
           H = hess.matrix; % incomplete Cholesky factor of truncated, permuted Hessian
           perm = hess.perm;
           dim = hess.dim;
           pk = zeros(length(grad(:)),1);
           grad_perm = grad(perm);
           grad_perm = grad_perm(1:dim);
           y = H\grad_perm;
           pk_perm  = H'\y;
           pk(perm(1:dim)) = pk_perm;
       
       case{'oper'}
           if isfield(hess,'preconditioner')
                [pk,pcg_flag,relres,iter] = pcg(hess.operator,grad,1e-1,250,hess.preconditioner);
           else
                [pk,pcg_flag,relres,iter] = pcg(hess.operator,grad,1e-1,250);
           end
           if verbose
                fprintf('PCG terminated with flag = %d after %d iterations \n', pcg_flag, iter);
           end
   end
end


end

