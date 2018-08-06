function [full_phase] = phase_foldout(half_phase, visualizer)
%
% Takes a Fourier phase defined on the plane in either phase (double)
% or phasor (complex) form and exploits symmetry to form a full phase.
% Also, given a 3 dimensional array of phases should fold out across the
% 3rd dimension
%
% Inputs:  half_phase - can be either phase (double) or phasor(complex),
%                       assumes the computed phase to exist on the left half
%                       of the Fourier plane (left of DC, quadrants 2,3).
%                       The phase should be truncated by the pupil mask
%                   
%          visualizer - 0 is no, 1 is yes if you want to see the results
%
% Outputs: full_phase - appropriately symmetric full phase or phasor,
%                       corresponding to the input type
%

if nargin<1
    runMinimalExample;
    return;
end

if nargin<2
    visualizer = 0;
end

full_phase = half_phase;
dims = size(half_phase);

if isreal(half_phase)
    full_phase(end:-1:2, end:-1:dims(2)/2+2,:) = -half_phase(2:end, 2:dims(2)/2,:);
    full_phase(end:-1:dims(1)/2+2, dims(2)/2+1,:) = -half_phase(2:dims(1)/2, dims(2)/2+1,:);
else
    full_phase(end:-1:1, end:-1:dims(2)/2+2,:) = conj(half_phase(1:end, 2:dims(2)/2,:));
    full_phase(end:-1:dims(1)/2+2, dims(2)/2+1,:) = conj(half_phase(2:dims(1)/2, dims(2)/2+1,:));
end 
    

if visualizer
    if isreal(half_phase)
        figure;
        subplot(1,2,1); imagesc(half_phase); axis image; title('Half phase');
        subplot(1,2,2); imagesc(full_phase); axis image; title('Symmetric full phase');
    else
        figure;
        subplot(1,2,1); imagesc(angle(half_phase)); axis image; title('Half phase');
        subplot(1,2,2); imagesc(angle(full_phase)); axis image; title('Symmetric full phase');
    end
end

end

function runMinimalExample
    im = TestImage1(64);
    phase = angle(fftshift(fft2(fftshift(im))));
    phase(:,34:end) = 0;
    phase(34:end,33) = 0;
    pupil_mask = MakeMask(2*32,1);
    phase = phase.*pupil_mask;    
    phase2 = phase_foldout(phase,1);

end

