function [codeTable, aveLen] = myhuffman(symbols, probabilities, extension)
% myhuffman(symbols, probabilities, extension) - Computes the Huffman code 
% table for the given symbols and probabilities
% 
% Parameters:
%   symbols - the input symbols (numerical array e.g. [1, 2, 3, 4, 5])
%   probabilities - the probabilities of the input symbols
%   extension - the N-th source extension (e.g. 1, 2, 3, ...)
%
% Output:
%   codeTable - the associated Huffman code table that has the encodings
%

% Print the entropy
entropy = -sum(probabilities.*log2(probabilities))/extension;
%fprintf('Entropy H(S) = %5.4f\n', entropy);

% Sort the probabilities as they may come unsorted
[sortedProb, ind] = sort(probabilities);
sortedSymbols = num2cell(symbols(ind));

% While the number of joined symbols/probabilities is greater than 2, join
% the two least likely probabilities
while length(sortedProb) > 2
    joinedProb = sum(sortedProb(1:2));
    sortedProb(1) = joinedProb;
    sortedProb(2) = [];
    % Combine the symbols
    combinedSym = [sortedSymbols(1:2)];
    sortedSymbols{1} = combinedSym;
    sortedSymbols(2) = [];
    % Resort the probabilities and symbols
    [sortedProb, ind] = sort(sortedProb);
    sortedSymbols = sortedSymbols(ind);
end

% Traverse the tree and assign codes
myCodes = assignCodes(sortedSymbols, '');
myCodes = reshape(myCodes, 2, length(myCodes)/2)';

% Re-sort the code table by symbol number
newSortedSymbols = [myCodes{:,1}]';
[newSortedSymbols, ind] = sort(newSortedSymbols);
newSortedSymbols = num2cell(newSortedSymbols);
newSortedCodes = myCodes(:,2);
newSortedCodes = newSortedCodes(ind);

codeTable = [newSortedSymbols newSortedCodes];

aveLen = 0;
% Calculate the average code length
for ii=1:length(newSortedCodes)
    aveLen = aveLen + length(newSortedCodes{ii})*probabilities(ii);
end

% Account for source extensions
aveLen = aveLen/extension;

%fprintf('Average length of the unconstrained Huffman code = %5.4f bits/symbol\n', aveLen);

%fprintf('Redundancy = L_ave - H(S) = %5.4f\n\n', aveLen - entropy);

%disp(codeTable)


end

