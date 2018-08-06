function [data_out, avg_noise_norm] = scale_and_noise(data, K_n, sigma_rn)
%
%   This function takes frames of data and scales them so that the number 
%   of photoevents in each frame is equal to K_n. It also adds zero-mean, 
%   white Gaussian noise with standard deviation sigma_rn. In practice, the
%   data will be either blurred image data or speckle star data.
%
%     Inputs:  
%                   data - 3D array of data frames
%                    K_n - desired number of photevents per frame
%               sigma_rn - desired standard deviation of zero-mean white
%                          Gaussian noise
%                   
%
%     Outputs: 
%               data_out - scaled and noisy data frames with K_n
%                          photoevents per frame and zero-mean white 
%                          Gaussian noise with standard deviation alpha_rn
%     
%             noise_norm - the average of the squared norms of the noise 
%                          added to each data frame. This is useful for 
%                          tests for determining regularization parameters
%

% Generic values if the fields are empty
if isempty(K_n)
    K_n = 3e6;
end

if isempty(sigma_rn)
    sigma_rn = 5;
end


% Scale and add noise to the data
avg_noise_norm = 0;
data_out = zeros(size(data));

for k = 1:size(data,3)
   data_out(:,:,k) = K_n*data(:,:,k)/sum(sum(data(:,:,k))); 
   noise = sigma_rn*randn(size(data,1),size(data,2));
   data_out(:,:,k) = data_out(:,:,k) + noise;
   avg_noise_norm = avg_noise_norm + norm(noise)^2;
end

avg_noise_norm = avg_noise_norm/size(data,3);

end

