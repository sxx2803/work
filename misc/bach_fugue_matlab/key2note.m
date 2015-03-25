function xx = key2note(X, keynum, dur)
% KEY2NOTE Produce a sinusoidal waveform corresponding to a
% given piano key number
%
% usage: xx = key2note(X, keynum, dur)
%
% xx = the output sinusoidal waveform
% X = complete amplitude for the sinusoid, X = A*exp(j*phi)
% keynum = the piano keyboard number of the desired note
% dur = the duration (in seconds) of the output note
%
fs = 44100;
tt = 0:(1/fs):dur;
freq = 440 * 2^((keynum-49)/12);
xx = real(X*exp(1i*2*pi*freq*tt));
xx = xx + real(0.3*exp(1i*pi/2)*X*exp(1i*2*pi*2*freq*tt));
xx = xx + real(0.707*exp(1i*3*pi/4)*X*exp(1i*2*pi*3*freq*tt));
xx = xx + real(0.3*exp(1i*5*pi/3)*X*exp(1i*2*pi*4*freq*tt));
xx = xx + real(0.2*exp(1i*5*pi/3)*X*exp(1i*2*pi*5*freq*tt));
xx = xx + real(0.115*exp(1i*5*pi/3)*X*exp(1i*2*pi*6*freq*tt));
xx = xx + real(0.15*exp(1i*5*pi/3)*X*exp(1i*2*pi*7*freq*tt));
xx = xx + real(0.1*exp(1i*5*pi/3)*X*exp(1i*2*pi*8*freq*tt));
xx = xx + real(0.05*exp(1i*5*pi/3)*X*exp(1i*2*pi*9*freq*tt));
xx = xx + real(0.03*exp(1i*5*pi/3)*X*exp(1i*2*pi*10*freq*tt));
adsr_envelope = adsr(4, dur, [0.002, 0.588, 0.01, 0.4], 44100);
xx = xx .* adsr_envelope;