function [phi_symm, D_over_r0] = GenerateWavefront(N, D2, Cn2, nlayers)
%  function phi = GenerateWavefront(N, D, r0)
%
%  Inputs:
%  N          --- number of grid points in x-direction.
%  D2          --- diameter of the observation aperture [m]
%  Cn2       ---  structure parameter
%  nlayers   ---  the number of turbulent layers
%
%  Outputs:
%  phi_symm        ---  cell array, generated wavefront of "nlayers" layers
%  D2/r0pw   --- D_over_r0

%  last updated Sept 25, 2011

if nargin == 1
    D2 = 0.5;  %  diameter of the observation aperture [m]
    %Cn2 = 1e-16;  %  structure parameter [m^(-2/3)], constant
    Cn2 = 5.5590e-17; % D_over_r0 = 5;
    %Cn2 = 8.1918e-015; %  in this case r0pw = 0.005, and D2/r0pw = 100
    %Cn2 = 5.6031e-016; % r0pw = 0.025, D2/r0pw = 20
    nlayers = 2;
elseif nargin == 2
    Cn2 = 1e-16;
    nlayers = 2;
elseif nargin == 3
    nlayers =2;
end


%  determine geometry
wavelength = 1e-6;  %  optical wavelength [m]
                    % JN:  should be 0.744e-6
k = 2*pi / wavelength;  %  optical wavenumber [rad/m]
Dz = 50e3;  %  propagation distance [m]
            %  JN:  What is this?
            

%  use sinc to model pt source
DROI = 4 * D2; % diam of obs-plane region of interest [m]
D1 = wavelength *Dz /DROI;  %  width of central lobe [m]
R = Dz;  %  wavefront radius of curvature [m]

%  PW coherence diameters [m]
p = linspace(0, Dz , 1e3);
r0pw = (0.423 * k^2 * Cn2  * Dz)^(-3/5);

l0 = 0;  %  inner scale [m]
L0= inf;  %outer scale [m]

%  log-amplitude variance
rytov_pw = 0.563 * k^(7/6)  * Dz^(5/6) * sum(Cn2 * (1-p/Dz).^(5/6) ...
    * (p(2) - p(1)));

if nlayers == 1
%    l0 = 0;  %  inner scale [m]
%    L0= inf;  %outer scale [m]
    
    delta = 10e-3;
   
    %  initialize array for phase screens
    %phi_asymm= zeros(N,N,n);
    phi_symm= cell(nlayers,1);
    
    [phi_symm{1}] = ft_phase_screen(r0pw,N, delta, L0, l0);
    
elseif nlayers > 1
    %  screen properties
    A = zeros(2, nlayers); %  matrix
    alpha = (0:nlayers-1) / (nlayers - 1);
    A(1,:) = ones(1, nlayers);
    A(2,:) = (1-alpha).^(5/6); %.* alpha.^(5/6);
    b = [r0pw.^(-5/3); rytov_pw/1.33*(k/Dz)^(5/6)];
    
    %  two methods solving Fried parameters for sublayers
    %  method I: normal equation
    if(rank(A) ~= 0)
        AAT = A*A';
        y = AAT\b;
        X_normal = A' * y;
    end
    
    %  method II: underdetermined LS problem with constrains
    %  initial guess
    x0 = (nlayers/3*r0pw * ones(nlayers, 1)).^(-5/3);
    %x0 = zeros(3,1);
    %  objective function
    fun = @(X)sum((A*X(:) - b).^2);
    %  constraints
    x1 = zeros(nlayers, 1);
    rmax = 0.1;  %  maximum Rytov number per partial prop
    x2 = rmax/1.33*(k/Dz)^(5/6) ./A(2,:);
    x2(A(2,:) ==0) = 50^(-5/3);
    [X_LS, fval, exitflag, output] ...
        = fmincon(fun, x0, [],[],[],[],x1, x2);
    %  check screen r0s
    r0scrn_normal = X_normal.^(-3/5);
    r0scrn_normal(isinf(r0scrn_normal)) = 1e6;
    
    r0scrn_LS = X_LS.^(-3/5);
    r0scrn_LS(isinf(r0scrn_LS)) = 1e6;
    % % %  check resulting r0pw & rytov_pw
    % bp_normal = A*X_normal(:);
    % [bp_normal(1)^(-3/5) bp_normal(2)*1.33*(Dz/k)^(5/6)];
    % [r0pw rytov_pw];
    %
    % bp_LS = A*X_LS(:);
    % [bp_LS(1)^(-3/5) bp_LS(2)*1.33*(Dz/k)^(5/6)];
    % [r0pw rytov_pw];
    
    n = nlayers;  %  number of planes
    z = (1:n-1) * Dz / (n-1);  % partial prop planes
    zt = [0 z];  % propagation plane locations
    Delta_z = zt(2:n) - zt(1:n-1);  % propagation distances
    
    %  grid spacings
    alpha = zt / zt(n);
    %c =2;
    % D1p = D1 + c*wavelength*Dz/r0pw;
    % D2p = D2 + c*wavelength*Dz/r0pw;
    % deltan = linspace(0, 1.1*wavelength*Dz/D1p, 100);
    % delta1 = linspace(0, 1.1*wavelength*Dz/D2p, 100);
    delta1 = 10e-3;
    deltan = 10e-3;
    delta = (1-alpha) * delta1 + alpha * deltan;
    
    %  initialize array for phase screens
    %phi_asymm= zeros(N,N,n);
    phi_symm= cell(n,1);
    
    
    %  Choose Fried parameters for sublayers from either normal equation
    %  or minimization problem
    for idxscr = 1:n
        %phi_symm(:,:,idxscr) = ft_phase_screen(r0scrn_LS(idxscr), N, delta(idxscr), L0, l0);
        [phi_symm{idxscr}] = ft_phase_screen(r0scrn_normal(idxscr), N, delta(idxscr), L0, l0);
    end
    
end
D_over_r0 = D2/r0pw;


