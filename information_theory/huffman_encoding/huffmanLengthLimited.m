function [ codeTable ] = huffmanLengthLimited( symbols, probabilities, limit )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Do the unconstrained Huffman coding first
codeTable = myhuffman(symbols, probabilities, 1);
codeTableOrig = codeTable;

% Get the number of codewords
bits = countLengths(codeTable);
bits = bits(:,2);

% Get the initial index value from the anticipated L_max value
ii = ceil(-log2(min(probabilities)))+1;

% Perform the length limiting algorithm
while true
    % Check if out of bounds
    if ii > length(bits)
        ii = ii - 1;
        continue;
    elseif bits(ii) > 0
        jj = ii - 1;
        jj = jj - 1;
        % If no codeword of that length, keep decrementing
        while bits(jj) == 0
            jj = jj -1;
        end
        bits(ii) = bits(ii)-2;
        bits(ii-1) = bits(ii-1)+1;
        bits(jj+1) = bits(jj+1)+2;
        bits(jj) = bits(jj)-1;
    else
        bits(ii) = [];
        ii = ii - 1;
        if ii == limit
            break;
        end
    end
end

newCodeLengths = [];

for ii=1:length(bits)
    newCodeLengths = [newCodeLengths; repmat(ii, bits(ii), 1)];
end

% To re-construct the code table, the canonical form is used. First, the
% code table is re-sorted.
[unused, ind] = sort(probabilities, 'descend');
codeSymbols = codeTable(:,1);
codeSymbols = codeSymbols(ind);
codeTable = [codeSymbols num2cell(newCodeLengths)];

% The length limited canonical Huffman code table can now be generated in
% the canonical form where each next symbol has a "larger" code word, if the
% codewords were translated from binary to decimal.
code = repmat('0', 1, codeTable{1,2});
for ii=1:length(codeSymbols)-1
    curLength = codeTable{ii,2};
    nextLength = codeTable{ii+1,2};
    codeTable{ii, 2} = code;
    code = bitsll((bin2dec(code)+1), (nextLength - curLength));
    code = dec2bin(code, nextLength);
end
codeTable{ii+1, 2} = code;

% Re-sort the code table by symbol number for display purposes
newSortedSymbols = [codeTable{:,1}]';
[newSortedSymbols, ind] = sort(newSortedSymbols);
newSortedSymbols = num2cell(newSortedSymbols);
newSortedCodes = codeTable(:,2);
newSortedCodes = newSortedCodes(ind);

codeTable = [newSortedSymbols newSortedCodes];

aveLen = 0;
% Calculate the average code length
for ii=1:length(newSortedCodes)
    aveLen = aveLen + length(newSortedCodes{ii})*probabilities(ii);
end

fprintf('Average length of length limited the Huffman code = %5.4f bits/symbol\n', aveLen);

% Print the codeword length counts
bits = countLengths(codeTable);

fprintf('Length\t# of Codewords\n');
disp(bits);

end

