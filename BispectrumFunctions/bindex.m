function [b,rr] = bindex(N,fourier_rad, second_rad, visualizer)
%
%   This function creates the indexing function used for recursive phase
%   reconstruction and iterative phase reconstruction from the bispectrum
%   as described in Tyler & Schulze (2003)
%
%   [b,rr] = bindex(N,fourier_rad, second_rad, visualizer)
%
%   Input:           N - size of image NxN
%          fourier_rad - larger fourier radius rho
%           second_rad - smaller second radius less than |v|_max=r_0/lamda
%           visualizer - 0 is no, 1 is yes
%           
%
%   Output:     b - indexing structure containing the following fields 
%             b.u - list of u indices
%             b.v - list of v indices
%           b.u_v - list of u+v indices (note: I freqently call u+v = s)
%
%
if nargin <3
    runMinimalExample
end

if nargin < 4
    visualizer = 0;
end

% First, I created a grid of spatial frequencies and also their angles so
% that I can increase my u+v vector radially.
[xx,yy] = meshgrid(-N/2:N/2-1, N/2:-1:-N/2 +1); % center (N/2 +1,N/2 +1)
rr = (xx.*xx + yy.*yy).^0.5;
theta = angle(xx + 1i*yy);
theta(N/2 +2:N,:) = theta(N/2 +2:N,:) + 2*pi;

dc_ind = find(rr == 0);
s_mags = unique(sort(rr(:)));   % sort the u+v values by increasing radii
s_mags = s_mags(s_mags <= fourier_rad); 
v_mags = s_mags(s_mags <= second_rad);

s_inds = [];
v_inds = [];

u_final = [];
v_final = [];
s_final = [];


% First, I collect a small(er) index of all the u+v indices (called s_inds) sorted by increasing order of
% radius and then angle from [0,2pi]. We want all s_inds less than the Fourier radius from DC

for ii = 1:length(s_mags)-1  % |s| from change of variables
    s_temp = find(rr > s_mags(ii) & rr <= s_mags(ii+1));
    [~,P] = sort(theta(s_temp),'descend');
    s_temp = s_temp(P);
    s_inds = [s_inds; s_temp];
end


% Second, I collect the set of v_inds I'm interested in. Again, this is
% sorted by order of increasing radius and then angle. 

for jj = 1:length(v_mags)-1
    v_temp = find(rr > v_mags(jj) & rr <= v_mags(jj+1));
    [~,P] = sort(theta(v_temp),'descend');
    v_temp = v_temp(P);
    v_inds = [v_inds; v_temp];
end


% Thirdly, I need to get the resulting u vectors. Given the u+v and v sets
% above for each potential u+v(j) = s(j), I should be able to displace the 
% v_inds to have center u+v(j) and then calculate the set of u_inds that is
% corresponds to the shifted v_inds and adds up to u+v(j). Lastly, these are
% loaded into the final indices

[x_cent,y_cent] = ind2sub([N,N], dc_ind); % coordinates of dc for calculating displacement
[v1,v2] = ind2sub([N,N],v_inds); % coordinates of the unshifted v and s points
[s1,s2] = ind2sub([N,N],s_inds);

for kk = 1:length(s_inds)
    % First, find the u coordinate by noting u+v = s implies u = s-v, then shift for matrix coordinates 
    u1 = s1(kk) - v1;
    u2 = s2(kk) - v2;
    u1 = u1 + x_cent;
    u2 = u2 + y_cent;


    % Next, find the set of u coordinates that are inside the "matrix" (needed
    % near the boundary.) 
    bd_sub = find((0 < u1 & u1 < N+1) & (0 < u2 & u2 < N+1)); 
    u_temp = sub2ind([N,N], u1(bd_sub),u2(bd_sub));
    
    % Then, find the set of u coordinates that are inside the prescribed fourier
    % radius
    bd_rad = find(rr(u_temp) <= fourier_rad);
    
    % Lastly, update all three final indices to be loaded into the structure.
    % For each loop, u+v = s will add the same index many times to
    % correspond to all the possible u and v such that u+v = s
    u_final = [u_final; u_temp(bd_rad)];
    v_final = [v_final; v_inds(bd_rad)];
    s_final = [s_final; s_inds(kk)*ones(length(u_temp(bd_rad)),1)];
end

% Take out any u_inds corresponding to dc
udc_inds = find(u_final ~= dc_ind);
u_final = u_final(udc_inds);
v_final = v_final(udc_inds);
s_final = s_final(udc_inds);

% Keep only one half of the bispectrum, we know the rest by symmetry
half_inds = find(s_final < dc_ind);
u_final = u_final(half_inds);
v_final = v_final(half_inds);
s_final = s_final(half_inds);

% Output index structure
b.u = u_final;
b.v = v_final;
b.u_v = s_final;



%Visualizer: comment in/out accordingly
if visualizer 
    figure;
    [ux,uy] = ind2sub([N,N],b.u);
    [vx,vy] = ind2sub([N,N],b.v);
    [sx,sy] = ind2sub([N,N],b.u_v);

    four_inds = find(rr < fourier_rad + 0.25 & rr > fourier_rad - 0.25);
    [~,P] = sort(theta(four_inds),'descend');
    four_inds = four_inds(P);
    [c,s] = ind2sub([N,N], four_inds);
    sec_inds = find(rr < second_rad + 0.25 & rr > second_rad - 0.25);
    [~,P] = sort(theta(sec_inds),'descend');
    sec_inds = sec_inds(P);
    [c2,s2] = ind2sub([N,N],sec_inds);
    [dc1,dc2] = ind2sub([N,N], dc_ind);
    for ll = floor(length(b.u)/2):floor(length(b.u)/2+100)
        clf;
        imagesc(rr);
        title('Index Visualizer')
        axis image; hold on;
        plot(c,s,'r','Linewidth',1)
        plot(c2,s2,'g','Linewidth',1)
        plot(dc1,dc2,'mo','MarkerFaceColor','m','MarkerSize',1.5)
        plot(ux(ll),uy(ll),'b+')
        plot(vx(ll),vy(ll),'g+')
        plot(sx(ll),sy(ll),'ro')
        legend('Fourier Rad.','Second Rad.','DC','u','v','u+v','location','eastoutside')
        pause(0.2)
    end
end

function runMinimalExample
    N = 64;
    fourier_rad = 32;
    second_rad = 5;
    visualizer = 1;
    
    b = bindex(N, fourier_rad, second_rad, visualizer);

