function [ codeLength, newA, newN ] = golombAdaptive( inVal, A, N, Nmax)
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

% Get the number of unary bits
%nShiftRight = bitsrl(uint32(inVal), kEst);
nShiftRight = floor(inVal/(2^kEst));
unaryBits = nShiftRight + 1;

% The length of the remainder bits is just the k-value
codeLength = unaryBits + kEst;

% Update A and N
if N == Nmax
    A = floor(A/2);
    N = floor(N/2);
end
newA = A + inVal;
newN = N + 1;

end

