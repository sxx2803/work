function envelope = adsr(alevel, duration, a_d_s_r, fs)

% adsr:
%
% Matlab function to create an 'attack-delay-sustain-release' window.
%
% Usage:
% envelope = adsr(alevel, duration, fs);
%
% inputs:
%		alevel:   overshoot for attack 
%				    (1 = no overshoot, 1.1 = 10% overshoot, .9 = 10% undershoot)
%     a_d_s_r:  vector definine the percentage of time spent on a, d, s, and r
%				    (must be non-negative and sum to one)
%     duration: duration for envelope
%     fs:       sample frequency
%

if (length(a_d_s_r) ~= 4)
   error('length of a_d_s_r must be equal to 4');
end
if (sum(a_d_s_r) ~= 1)
   %error('a_d_s_r must sum to one');
end
if (sum(a_d_s_r <= 0) ~= 0)
   error('a_d_s_r must have non-negative elements');
end

times = a_d_s_r*duration;
t     = 0:1/fs:duration;
N     = length(t);

N_attack  = round(times(1)*fs);
N_decay   = round(times(2)*fs);
N_sustain = round(times(3)*fs);
N_release = N - N_attack - N_decay - N_sustain;

if alevel > 1
   slevel = 1/alevel;
   alevel  = 1;
else
   s = alevel;
   slevel = 1;
end

envelope = [linspace(0,alevel,N_attack) ...
            linspace(alevel,slevel,N_decay) ...
            linspace(slevel,slevel,N_sustain) ...
            linspace(slevel,0,N_release)];