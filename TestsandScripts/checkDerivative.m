%==============================================================================
% (c) Jan Modersitzki 2010/12/27, see FAIR.2 and FAIRcopyright.m.
% http://www.mic.uni-luebeck.de/people/jan-modersitzki.html
%
% function [fh,ph,th] = checkDerivative(f0tn,x0);
%
% checks the implementation of a derivative by comparing the function 
% with the Taylor-poly
%
%   \| f(x0 + h ) - TP_p(x0,h) \|   !=   O( h^{p+1} ) 
%
% Input:
%   f0tn    function handle
%  x0      expanding point
%
% Output:
%  fh      figure handle to graphical output
%  ph      plot handle
%  th      text handle
% call checkDerivative for a minimal example.
%==============================================================================

function varargout = checkDerivative(fctn,x0,varargin)

if nargin == 0, % help and minimal example
  help(mfilename); 
  fctn = @xSquare;  x0 = 1;  checkDerivative(fctn,x0);
  return;
end;


fig = [];
for k=1:2:length(varargin),     % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end;

fprintf('%s: derivative test\n',mfilename);
fprintf('T0 = |f0 - ft|, T1 = |f0+h*f0'' - ft|\n');
[f0,df,H] = feval(fctn,x0);  
if ~isnumeric(df),
  try
    [f0,df] = feval(fctn,x0);  
  catch
    if isfield(df,'Q'), df = df.Q;  end;
  end;
end;

h = logspace(-1,-10,10);

v = randn(size(x0));    
    
if isnumeric(df),
  dvf = df'*v;
elseif isa(df,'function_handle'),
  dvf = df(v);
else
  keyboard;
end;

if isnumeric(H),
    vHv = v'*H*v;
elseif isstruct(H),
    if isfield(H,'operator');
        vHv = v'*H.operator(v);
    elseif isfield(H,'matrix');
        vHv = v'*H*v;
    end
end    

for j=1:length(h),
  ft = feval(fctn,x0+h(j)*v);                           % function value
  T0(j) = norm(f0-ft);                                  % TaylorPoly 0
  T1(j) = norm(f0 + h(j)*dvf - ft);                     % TaylorPoly 1
  T2(j) = norm(f0 + h(j)*dvf + 0.5*h(j)*h(j)*vHv - ft); % TaylorPoly 2
  fprintf('h=%12.4e     T0=%12.4e    T1=%12.4e      T2=%12.4e\n',h(j),T0(j),T1(j),T2(j));
end;
if isempty(fig),
  fh = figure;
else
  fh = figure(fig);
end;
ph = loglog(h,[T0;T1;T2]); set(ph(2),'linestyle','--')
th  = title(sprintf('%s: |f-f(h)|,|f+h*dvf -f(h)|,|f+h*dvf+0.5*h*h*vHv -f(h)| vs. h',mfilename));

if nargout>0,
  varargout = {fh,ph,th};
end;
%{ 
  =======================================================================================
  FAIR: Flexible Algorithms for Image Registration, Version 2011
  Copyright (c): Jan Modersitzki
  Maria-Goeppert-Str. 1a, D-23562 Luebeck, Germany
  Email: jan.modersitzki@mic.uni-luebeck.de
  URL:   http://www.mic.uni-luebeck.de/people/jan-modersitzki.html
  =======================================================================================
  No part of this code may be reproduced, stored in a retrieval system,
  translated, transcribed, transmitted, or distributed in any form
  or by any means, means, manual, electric, electronic, electro-magnetic,
  mechanical, chemical, optical, photocopying, recording, or otherwise,
  without the prior explicit written permission of the authors or their
  designated proxies. In no event shall the above copyright notice be
  removed or altered in any way.

  This code is provided "as is", without any warranty of any kind, either
  expressed or implied, including but not limited to, any implied warranty
  of merchantibility or fitness for any purpose. In no event will any party
  who distributed the code be liable for damages or for any claim(s) by
  any other party, including but not limited to, any lost profits, lost
  monies, lost data or data rendered inaccurate, losses sustained by
  third parties, or any other special, incidental or consequential damages
  arrising out of the use or inability to use the program, even if the
  possibility of such damages has been advised against. The entire risk
  as to the quality, the performace, and the fitness of the program for any
  particular purpose lies with the party using the code.
  =======================================================================================
  Any use of this code constitutes acceptance of the terms of the above statements
  =======================================================================================
%}