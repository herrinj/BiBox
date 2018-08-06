% Script to run the function "GenerateWavefront"
% The input parameter 800 is the size of the generated phase;
% the default number of screens are 3, and we could adjust
% this variable inside the function.
%
% Also, in "GenerateWavefront", we have default parameter
% D and Cn2. These could be overwritten by sending three
% parameters when run the function, eg:
% [phi, rat] = GenerateWavefront(800, 1, 0.05)
%
% The output "phi" is the generated phase, and rat is the ratio
% of the diameter of the observation aperture and r_0.

clear
n = 256;
[phi, rat] = GenerateWavefront(n);


