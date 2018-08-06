function ShowSpeckleData(phase_frames, PSF_frames, blurred_image_frames)
%
%  This is just a simple function to show the data created for a
%  multi-frame image blurring problem, assuming PSFs from speckle imaging.
%
n_frames = size(phase_frames, 3);
figure(1), clf
phase_max = max(phase_frames(:));
phase_min = min(phase_frames(:));
warning off
for k = 1:n_frames
    imshow(phase_frames(:,:,k), [phase_min, phase_max])
    colormap(jet)
    pause(0.2)
end
warning on

figure(2), clf
PSF_max = max(PSF_frames(:));
for k = 1:n_frames
    mesh(PSF_frames(:,:,k))
    axis([50,250,50,250,0,PSF_max])
    pause(0.2)
end

figure(3), clf
warning off
for k = 1:n_frames
    imshow(blurred_image_frames(:,:,k), [])
    %colormap(jet)
    pause(0.2)
end
warning on
