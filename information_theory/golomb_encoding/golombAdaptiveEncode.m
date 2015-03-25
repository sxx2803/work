function [ codeWord, newA, newN ] = golombAdaptiveEncode( inVal, A, N, Nmax)
% golombAdaptive(inVal, A, N, Nmax) - Computes the codeword of the Golomb
% code that the input val 'inVal' would encode to. Also returns the updated
% values of A and N for future use.
%
% Parameters:
%   inVal - the input value to encode as a non-negative integer
%   A - the estimated expected value
%   N - the counter value
%   Nmax - the maximum counter value for renormalization
%
% Output:
%   codeLength - the Golomb code
%   newA - the updated estimated expected value
%   newN - the updated counter value

kEst = max([0 ceil(log2(A/(2*N)))]);
%inBin = dec2bin(inVal);

% Construct the unary code
%nShiftRight = bitsrl(uint32(inVal), kEst);
quotient = floor(inVal/(2^kEst));
unaryCode = [repmat('0', 1, quotient) '1'];

% Construct the remainder code
remainder = mod(inVal, 2^kEst);
remainderCode = dec2bin(remainder, kEst);

codeWord = [unaryCode remainderCode];

% Update A and N
if N == Nmax
    A = floor(A/2);
    N = floor(N/2);
end
newA = A + inVal;
newN = N + 1;

end

