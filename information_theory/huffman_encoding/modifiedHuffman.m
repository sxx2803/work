function [codeTable] = modifiedHuffman( symbols, probabilities, pElse )
% modifiedHuffman(symbols, probabilities) - Computes the Huffman code table
% for the modified Huffman algorithm where the least probable symbols are
% joined until the combined symbol probabilities are greater than or equal
% to pElse.
%
% Parameters:
%   symbols - the input symbols (numerical array e.g. [1, 2, 3, 4, 5])
%   probabilities - the probabilities of the input symbols
%   pElse - the specified minimum probability for the ELSE symbol
%
% Output:
%   codeTable - the relevant Huffman code table in an nx2 cell array
%

% Sort the probabilities as they may come unsorted
[sortedProb, ind] = sort(probabilities);
sortedSymbols = num2cell(symbols(ind));

% Go through the probability list and find the index of the cutoff for the
% cumulative probability of the ELSE symbols
idx = 0;
combinedProb = 0;

while combinedProb < pElse
    idx = idx+1;
    combinedProb = combinedProb + sortedProb(idx);
end

% Make the ELSE symbol and insert it into the symbol and probability array
elseSymbol = sortedSymbols(1:idx);
sortedProb(2:idx) = [];
sortedProb(1) = combinedProb;
sortedSymbols(2:idx) = [];
sortedSymbols{1} = elseSymbol;

% Perform regular Huffman coding with the new ELSE symbol

% Re-sort the array
[sortedProb, ind] = sort(sortedProb);
sortedSymbols = sortedSymbols(ind);

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
myCodes = assignCodesModified(sortedSymbols, '');
myCodes = reshape(myCodes, 2, length(myCodes)/2)';

% Re-sort the code table by symbol number
newSortedSymbols = [myCodes{:,1}]';
[newSortedSymbols, ind] = sort(newSortedSymbols);
newSortedSymbols = num2cell(newSortedSymbols);
newSortedCodes = myCodes(:,2);
newSortedCodes = newSortedCodes(ind);

codeTable = [newSortedSymbols newSortedCodes];

aveLen = 0;
uniqueBits = 0;
elseSymbol = '';
% Calculate the average code length
for ii=1:length(myCodes)
    if isempty(strfind(newSortedCodes{ii}, '+'))
        aveLen = aveLen + length(newSortedCodes{ii})*probabilities(ii);
    else
        actualCode = strsplit(newSortedCodes{ii}, '+');
        aveLen = aveLen + (length(actualCode{1}) + length(actualCode{2})) * probabilities(ii);
        elseSymbol = actualCode{1};
        uniqueBits = length(actualCode{2});
    end
end

if ~isempty(elseSymbol)
    fprintf('The ELSE symbol encoding is %s\n', elseSymbol);
    fprintf('%i additional bits are required for unique decoding of ELSE symbol\n', uniqueBits);
end
fprintf('Average length of the modified Huffman code = %5.4f bits/symbol\n\n', aveLen);

end

