function [dy,solveInfo] = stepGN(op,rhs,solver,varargin)
%
%   Input:      op - Hessian or Jacobian matrix/function handle/structure
%              rhs - (should be) negative gradient 
%           solver - string/function handle indicating which solver to use
%    
%   Output:     dy - solution for Gauss-Newton step
%        solveInfo - structure containing information relevant to solver
%


active          = [];
solverMaxIter   = 25;
solverTol       = 1e-1;


for k=1:2:length(varargin), % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

if isempty(solver) && isnumeric(op)
    solver = 'backslash';
elseif isa(solver,'function_handle')
    if isempty(inactive)
        dy = solver(rhs,op,solverMaxIter,solverTol);
    else
        dy = solver(rhs(inactive),op(inactive,inactive),solverMaxIter,solverTol);
    end
    return;
end

switch solver
    case 'backslash' 
        %
        %   Assumes operator is Hessian, not Jacobian
        %
        if (isempty(active) || sum(active) == 0)
            dy = op\reshape(rhs,[],1);
        else
            dy = zeros(size(reshape(rhs,[],1)));
            dy(~active) = op(~active,~active)\reshape(rhs(~active),[],1);
        end
        solveInfo = [];
    
    case 'mbFULLlsdir'
        %
        %   Matrix-based, fully coupled (unprojected) least-squares solver
        %   for Gauss-Newton step using direct regularization with fixed
        %   alpha 
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x
        Jw      = op.Jw; % Jacobian w.r.t. w 
        res     = op.res; % current residual
        yc      = op.yc; % current iterate
        alpha   = op.alpha; % regularization parameter
        S       = op.S;  % regularizer
        xdim    = op.xdim;
        wdim    = op.wdim;
        xc      = yc(1:xdim);
        activex = active(1:xdim);
        activew = active(xdim+1:end);
        
        % Set up system
        Jx      = Jx(:,not(activex));   % restrict Jx to inactive set
        Jw      = Jw(:,not(activew));   % restrict Jw to inactive set
        xc      = xc(not(activex));
        S       = S(:,not(activex));    % restrict regularizer
        
        rhs     = [-res; -sqrt(alpha)*S*xc];
        op      = [Jx, Jw; sqrt(alpha)*S, zeros(size(S,1),size(Jw,2))];
        
        % Solve for dy = [dx;dw] directly
        dy      = zeros(xdim+wdim,1);
        [dy(not(active)),~,~,iters] = lsqr(op,rhs,solverTol,solverMaxIter);        
        
        % Output relevant info
        solveInfo.iters = iters;
        
    case 'mbLAPlsdir' 
        %
        %   Matrix-based, coupled LAP least-squares solver for Gauss-Newton 
        %   step using direct regularization with fixed alpha 
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x
        Jw      = op.Jw; % Jacobian w.r.t. w 
        res     = op.res; % current residual
        yc      = op.yc; % current iterate
        alpha   = op.alpha; % regularization parameter
        S       = op.S;  % regularizer
        xdim    = op.xdim;
        wdim    = op.wdim;
        xc      = yc(1:xdim);
        activex = active(1:xdim);
        activew = active(xdim+1:end);
        
        % Set up system
        Jx      = Jx(:,not(activex));   % restrict Jx to inactive set
        Jw      = Jw(:,not(activew));   % restrict Jw to inactive set
        xc      = xc(not(activex));
        L       = chol(Jw'*Jw,'lower');  % store Cholesky factors for speed 
        S       = S(:,not(activex));    % restrict regularizer

        proj_rhs= [-res + Jw*(L'\(L\(Jw'*res))); -sqrt(alpha)*S*xc];
        proj_Jx = [Jx - Jw*(L'\(L\(Jw'*Jx))); sqrt(alpha)*S] ;         
        dx      = zeros(xdim,1);
        dw      = zeros(wdim,1);
        
        % Solve for dx iteratively
        [dx(not(activex)),~,~,iters] = lsqr(proj_Jx,proj_rhs,solverTol,solverMaxIter);        
        
        % Substitute directly for dw
        dw(not(activew)) = -real(L'\(L\(Jw'*Jx*dx(not(activex)) + Jw'*res)));
        
        % Combine and return results
        dy = [dx(:);dw(:)];
        
        % Output relevant info
        solveInfo.iters = iters;
        
    case 'mfLAPlsdir'       
        %
        %   Matrix-free, coupled LAP least-squares solver for Gauss-Newton 
        %   step using direct regularization with fixed alpha 
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x, expected as a function handle
        Jw      = op.Jw; % Jacobian w.r.t. w 
        res     = op.res; % current residual
        yc      = op.yc; % current iterate
        alpha   = op.alpha; % regularization parameter
        S       = op.S;  % regularizer
        xdim    = op.xdim;
        wdim    = op.wdim;
        xc      = yc(1:xdim);
        activex = active(1:xdim);
        activew = active(xdim+1:end);
        Q       = speye(xdim);
        Q       = Q(:,~activex);
        
        % Set up system
        Jw      = Jw(:,not(activew));   % restrict Jw to inactive set
        xc      = xc(not(activex));
        L       = chol(Jw'*Jw,'lower');  % store Cholesky factors for speed 
        S       = S(:,not(activex));    % restrict regularizer

        proj_rhs= [-res + Jw*(L'\(L\(Jw'*res))); -sqrt(alpha)*(S*xc)];
        proj_Jx = @(x,flag) proj_Jx_mf(x, Jx, Jw, L, alpha, S, Q, flag);
        dx      = zeros(xdim,1);
        dw      = zeros(wdim,1);
        
        % Solve for dx iteratively
        [dx(not(activex)),~,~,iters] = lsqr(proj_Jx, proj_rhs, solverTol,solverMaxIter);        
        
        % Substitute directly for dw
        dw(not(activew)) = -real(L'\(L\(Jw'*Jx(dx,'notransp') + Jw'*res)));
        
        % Combine and return results
        dy = [dx(:);dw(:)];    
        
        % Output relevant info
        solveInfo.iters = iters;
    
    case 'mbLAPlsHyBR'
        %
        %   Matrix-based, coupled LAP least-squares solver for Gauss-Newton 
        %   step using hybrid regularization with automatic regularization
        %   parameter selection
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x, expected as a function handle
        Jw      = op.Jw; % Jacobian w.r.t. w 
        res     = op.res; % current residual
        xdim    = op.xdim;
        wdim    = op.wdim;
        activex = active(1:xdim);
        activew = active(xdim+1:end);
        
        % Set up HyBR parameters
        options = IRhybrid_lsqr('defaults');
        options = IRset(options,'IterBar','off');
        options = IRset(options,'resflatTol',solverTol);
        options = IRset(options,'MaxIter', solverMaxIter);
        options = IRset(options,'Reorth','off');
        options = IRset(options,'verbosity','off');
       
        % Set up system
        Jx      = Jx(:,not(activex));   % restrict Jx to inactive set
        Jw      = Jw(:,not(activew));   % restrict Jw to inactive set
        L       = chol(Jw'*Jw,'lower');  % store Cholesky factors for speed 

        proj_rhs= -res + Jw*(L'\(L\(Jw'*res)));
        proj_Jx = Jx - Jw*(L'\(L\(Jw'*Jx)));         
        dx      = zeros(xdim,1);
        dw      = zeros(wdim,1);
        
        % Run HyBR to get regularized solution for dx
        [dx(not(activex)), IterInfo] = IRhybrid_lsqr(proj_Jx, proj_rhs, options, []);
        alpha   = IterInfo.RegP(end);
        iters   = IterInfo.its;
        
        % Retrieve dw, put together results into dy, and compute the gradient dJ
        dw(not(activew)) = -real(L'\(L\(Jw'*(Jx*dx(not(activex))) + Jw'*res)));
        
        % Combine and return results
        dy = [dx(:); dw(:)]; 
        
        % Compute new gradient
        dJ = [res'*op.Jx + alpha*ones(1,xdim), res'*op.Jw];      
        
        % Output relative info
        solveInfo.alpha = alpha;
        solveInfo.iters = iters;
        solveInfo.dJ    = dJ;    
        
    case 'mfLAPlsHyBR'
        %
        %   Matrix-free, coupled LAP least-squares solver for Gauss-Newton 
        %   step using hybrid regularization with automatic regularization
        %   parameter selection
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x, expected as a function handle
        Jw      = op.Jw; % Jacobian w.r.t. w 
        res     = op.res; % current residual
        xdim    = op.xdim;
        wdim    = op.wdim;
        activex = active(1:xdim);
        activew = active(xdim+1:end);
        Q       = speye(xdim);
        Q       = Q(:,~activex);
        
        % Set up HyBR parameters
        options = IRhybrid_lsqr('defaults');
        options = IRset(options,'IterBar','off');
        options = IRset(options,'resflatTol',solverTol);
        options = IRset(options,'MaxIter', solverMaxIter);
        options = IRset(options,'Reorth','off');
        options = IRset(options,'verbosity','off');

        
        % Set up system
        Jw      = Jw(:,not(activew));   % restrict Jw to inactive set
        L       = chol(Jw'*Jw,'lower');  % store Cholesky factors for speed 

        proj_rhs= -res + Jw*(L'\(L\(Jw'*res)));
        proj_Jx = @(x,flag) proj_Jx_mf(x, Jx, Jw, L, [], [], Q, flag);
        dx      = zeros(xdim,1);
        dw      = zeros(wdim,1);
        
        % Run HyBR to get regularized solution for dx
        [dx(not(activex)), IterInfo] = IRhybrid_lsqr(proj_Jx, proj_rhs, options, []);
        alpha   = IterInfo.RegP(end);
        iters   = IterInfo.its;
        
        % Retrieve dw, put together results into dy, and compute the gradient dJ
        dw(not(activew)) = -real(L'\(L\(Jw'*Jx(dx,'notransp') + Jw'*res)));
        
        % Combine and return results
        dy = [dx(:);dw(:)]; 
        
        % Compute new gradient
        dJ = [Jx(res,'transp')' + alpha*ones(1,xdim), res'*op.Jw];      
        
        % Output relative info
        solveInfo.alpha = alpha;
        solveInfo.iters = iters;
        solveInfo.dJ    = dJ;
     
    case 'mbBCDlsdir'
        %
        %   Matrix-based, decoupled BCD least-squares solver for Gauss-Newton 
        %   step on image variables using direct regularization with fixed 
        %   alpha 
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x, expected as a function handle
        res     = op.res; % current residual
        yc      = op.yc; % current iterate
        alpha   = op.alpha; % regularization parameter
        S       = op.S;  % regularizer
        xdim    = op.xdim;
        xc      = yc(1:xdim);
        activex = active(1:xdim);
        Q       = speye(xdim);
        Q       = Q(:,~activex);
        
        % Set up system
        xc      = xc(not(activex));
        S       = S(:,not(activex));    % restrict regularizer
        Jx      = Jx(:,not(activex));
        rhs     = [-res; -sqrt(alpha)*(S*xc)];
        reg_Jx  = [Jx; sqrt(alpha)*S];
        dx      = zeros(xdim,1);
        
        % Solve for dx iteratively
        [dx(not(activex)),~,~,iters] = lsqr(reg_Jx, rhs, solverTol, solverMaxIter);        
        
        % Combine and return results
        dy = dx(:);    
        
        % Output relevant info
        solveInfo.iters = iters;    
        
    case 'mfBCDlsdir'
        %
        %   Matrix-free, decoupled BCD least-squares solver for Gauss-Newton 
        %   step on image variable using direct regularization with fixed 
        %   alpha 
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x, expected as a function handle
        res     = op.res; % current residual
        yc      = op.yc; % current iterate
        alpha   = op.alpha; % regularization parameter
        S       = op.S;  % regularizer
        xdim    = op.xdim;
        xc      = yc(1:xdim);
        activex = active(1:xdim);
        Q       = speye(xdim);
        Q       = Q(:,~activex);
        
        % Set up system
        xc      = xc(not(activex));
        S       = S(:,not(activex));    % restrict regularizer
        rhs     = [-res; -sqrt(alpha)*(S*xc)];
        reg_Jx  = @(x,flag) reg_Jx_mf(x, Jx, alpha, S, Q, length(res), flag);
        dx      = zeros(xdim,1);
        
        % Solve for dx iteratively
        [dx(not(activex)),~,~,iters] = lsqr(reg_Jx, rhs, solverTol, solverMaxIter);        
        
        % Combine and return results
        dy = dx(:);    
        
        % Output relevant info
        solveInfo.iters = iters;
     
     case 'mbBCDlshybr'
        %
        %   Matrix-based, decoupled BCD least-squares solver for Gauss-Newton 
        %   step on image variables using hybrid regularization 
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x, expected as a function handle
        res     = op.res; % current residual
        yc      = op.yc; % current iterate
        xdim    = op.xdim;
        activex = active(1:xdim);
      
        
  
        % Set up HyBR parameters
        options = IRhybrid_lsqr('defaults');
        options = IRset(options,'IterBar','off');
        options = IRset(options,'resflatTol',solverTol);
        options = IRset(options,'MaxIter', solverMaxIter);
        options = IRset(options,'Reorth','off');
        options = IRset(options,'verbosity','off');
        
        % Run HyBR to get regularized solution for dx
        restr_Jx= Jx(:,~activex);
        dx      = zeros(xdim,1);
        [dx(not(activex)), IterInfo] = IRhybrid_lsqr(restr_Jx, -res, options, []);
        alpha   = IterInfo.RegP(end);
        iters   = IterInfo.its;
         
        % Return results
        dy = dx(:);    
        
        % Compute new gradient
        dJ = res'*Jx + alpha*ones(1,xdim);      
        
        % Output relative info
        solveInfo.alpha = alpha;
        solveInfo.iters = iters;
        solveInfo.dJ    = dJ;    
        
     case 'mfBCDlshybr'
        %
        %   Matrix-free, decoupled BCD least-squares solver for Gauss-Newton 
        %   step on image variables using hybrid regularization 
        %
        
        % Load things from Hessian structure
        Jx      = op.Jx; % Jacobian w.r.t. x, expected as a function handle
        res     = op.res; % current residual
        yc      = op.yc; % current iterate
        xdim    = op.xdim;
        activex = active(1:xdim);
        Q       = speye(xdim);
        Q       = Q(:,~activex);
        
  
        % Set up HyBR parameters
        options = IRhybrid_lsqr('defaults');
        options = IRset(options,'IterBar','off');
        options = IRset(options,'resflatTol',solverTol);
        options = IRset(options,'MaxIter', solverMaxIter);
        options = IRset(options,'Reorth','off');
        options = IRset(options,'verbosity','off');
        
        % Run HyBR to get regularized solution for dx
        restr_Jx      = @(x,flag) reg_Jx_mf(x, Jx, 0.0, [], Q, length(res), flag);
        dx      = zeros(xdim,1);
        [dx(not(activex)), IterInfo] = IRhybrid_lsqr(restr_Jx, -res, options, []);
        alpha   = IterInfo.RegP(end);
        iters   = IterInfo.its;
        
        % Return results
        dy = dx(:);    
        
        % Compute new gradient
        dJ = Jx(res,'transp')' + alpha*ones(1,xdim);      
        
        % Output relative info
        solveInfo.alpha = alpha;
        solveInfo.iters = iters;
        solveInfo.dJ    = dJ; 
     
     case 'mbBCDchol'
        %
        %   Matrix-based, decoupled BCD Cholesky solver for Gauss-Newton 
        %   step on motion variables using direct regularization with fixed 
        %   alpha 
        %
        %   Note: Not implemented with constraints, but possible to do so
        %
        
        % Load things from Hessian structure
        Jw      = op.Jw; % Jacobian w.r.t. x, expected as a function handle
        res     = op.res; % current residual
      
        % Solve
        L = chol(Jw'*Jw,'lower');
        rhs = -Jw'*res;    % real for MRI example
        dw = real(L\(L'\(rhs))); % real for MRI example
        
        % Combine and return results
        dy = dw(:);    
        solveInfo = [];   
        
     case 'mbBCDls'
        %
        %   Matrix-based, decoupled BCD least-squares solver for Gauss-Newton 
        %   step on motion variables using direct regularization with fixed 
        %   alpha 
        %
        %   Note: Not implemented with constraints, but possible to do so
        %
        
        % Load things from Hessian structure
        Jw      = op.Jw; % Jacobian w.r.t. x, expected as a function handle
        res     = op.res; % current residual
      
        % Solve
        dw = -real(Jw\res);
        
        % Combine and return results
        dy = dw(:);    
        solveInfo = [];
    
    case 'bispIm'
        %
        %   Operator-based call to PCG on inactive set for imphase and
        %   imphasor objective functions for bispectral imaging
        %
        rhs     = rhs(:);
        Q       = speye(numel(rhs));
        Q       = Q(:,~active);
        proj_op = @(x) Q'*op(Q*x); % Restrict operator to inactive set
        dy      = zeros(numel(rhs),1);
        [dy(not(active)),~,~,iters] = pcg(proj_op,rhs(not(active)),solverTol,solverMaxIter);
        solveInfo.iters = iters;
       
    case 'bispPhFull'    
        %
        %   Matrix-based solver for phase update for bispectral imaging
        %   codes using phase and phasor objective functions 
        %
        rhs     = rhs(:);
        H       = op.matrix;
        dy      = H\rhs;
        solveInfo = [];    
        
    case 'bispPhTrunc'  
        %
        %   Matrix-based solver for phase update for bispectral imaging
        %   codes using phase and phasor objective functions 
        %
        rhs     = rhs(:);
        Hp      = op.matrix; % truncated, permuted Hessian
        p       = op.perm;
        dim     = op.dim;
        dy      = zeros(length(rhs(:)),1);
        rhsp    = rhs(p);
        rhsp    = rhsp(1:dim);
        dyp     = Hp\rhsp;
        dy(p(1:dim)) = dyp;   
        solveInfo = [];    
        
    case 'bispPhIchol'  
        %
        %   Matrix-based solver for phase update for bispectral imaging
        %   codes using phase and phasor objective functions 
        %
        rhs     = rhs(:);
        L       = op.matrix; % (incomplete) Cholesky factor of truncated, permuted Hessian
        p       = op.perm;
        dim     = op.dim;
        dy      = zeros(length(rhs(:)),1);
        rhsp    = rhs(p);
        rhsp    = rhsp(1:dim);
        dyp     = L'\(L\rhsp);
        dy(p(1:dim)) = dyp;   
        solveInfo = [];
        
end

end

function [y] = proj_Jx_mf(x, Jx, Jw, L, alpha, S, Q, flag)
    %
    %   Used to create function handle to projected Jacobian operator for
    %   LAP. Can be used with direct or hybrid regularizers depending on if
    %   the regularizer S and regularization parameter alpha are empty
    %

    switch flag
        case{'notransp'}   
            % Restrict to inactive set
            if ~isempty(Q)
                Qx = Q*x;
            end
            
            % Multiply by Jacobian
            Jxx = Jx(Qx,'notransp');
            
            % Project
            y = Jxx - Jw*(L'\(L\(Jw'*Jxx)));
            
            % Concatenate with regularizer
            if ~isempty(S)
                y = [y; sqrt(alpha)*(S*x)];
            end
        case{'transp'}
            % Split vector into two pieces
            nres = size(Jw,1);
            x1 = x(1:nres);
            x2 = x(nres+1:end);
            
            % Project
            x1 = x1 - Jw*(L'\(L\(Jw'*x1)));
            
            % Multiply by Jacobian transpose
            y = Jx(x1,'transp');
            
            % Restrict to inactive set
            if ~isempty(Q)
                y = Q'*y;
            end
            
            % Add piece from regularizer
            if ~isempty(S)
                y = y + sqrt(alpha)*(S'*x2);
            end         
    end
end

function [y] = reg_Jx_mf(x, Jx, alpha, S, Q, dim, flag)
    %
    %   Used to create function handle to projected Jacobian operator for
    %   LAP. Can be used with direct or hybrid regularizers depending on if
    %   the regularizer S and regularization parameter alpha are empty
    %

    switch flag
        case{'notransp'}   
            % Restrict to inactive set
            if ~isempty(Q)
                Qx = Q*x;
            end
            
            % Multiply by Jacobian
            y = Jx(Qx,'notransp');
            
            % Concatenate with regularizer
            if ~isempty(S)
                y = [y; sqrt(alpha)*(S*x)];
            end
        case{'transp'}
            % Split vector into two pieces
            x1  = x(1:dim);
            x2  = x(dim+1:end);
            
            % Multiply by Jacobian transpose
            y = Jx(x1,'transp');
            
            % Restrict to inactive set
            if ~isempty(Q)
                y = Q'*y;
            end
            
            % Add piece from regularizer
            if ~isempty(S)
                y = y + sqrt(alpha)*(S'*x2);
            end         
    end
end
