function [ decoded, newA, newN ] = golombAdaptiveEncode( codeWord, A, N, Nmax)
% golombAdaptive(inVal, A, N, Nmax) - Decodes the codeword of the Golomb
% code from the input codeword 'inVal'. Also returns the updated
% values of A and N for future use.
%
% Parameters:
%   inVal - the input codeword value
%   A - the estimated expected value
%   N - the counter value
%   Nmax - the maximum counter value for renormalization
%
% Output:
%   decoded - the decoded value
%   newA - the updated estimated expected value
%   newN - the updated counter value

kEst = max([0 ceil(log2(A/(2*N)))]);

% Determine the quotient value
split = strsplit(codeWord, '1');
quotient = length(split{1});

% Get the remainder
remainder = codeWord(end-(kEst-1):end);
remainder = bin2dec(remainder);

decoded = (quotient * 2^kEst) + remainder;

% Update A and N
if N == Nmax
    A = floor(A/2);
    N = floor(N/2);
end
newA = A + decoded;
newN = N + 1;

end

