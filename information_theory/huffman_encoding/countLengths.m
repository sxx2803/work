function [ numCodeWords ] = countLengths( codeTable )
% countLengths - Counts the number of codewords with for each codeword
% length from 1 to maximum length
%
% Parameters:
%   codeTable - a nx2 cell where the first column is the symbol designation
%   as an integer and the second column is the symbol encoding as a string.
%
% Outputs:
%   numCodeWords - a nx2 matrix where the first column is the integer
%   length of a code word and the second column is how many code words are
%   that length.
%

codeWords = codeTable(:,2);

% Find the maximum length
maxLen = 0;
for ii=1:length(codeWords)
    if isempty(strfind(codeWords{ii}, '+'))
        wordLen = length(codeWords{ii});
        if wordLen > maxLen
            maxLen = wordLen;
        end
    else
        actualCode = strsplit(codeWords{ii}, '+');
        actualCode = actualCode{1};
        wordLen = length(actualCode);
        if wordLen > maxLen
            maxLen = wordLen;
        end
    end
end

% Initialize the table
numCodeWords = [1:maxLen]';
numCodeWords(:,2) = 0;

countedElseSymbol = false;
% Begin counting
for ii=1:length(codeWords)
    if isempty(strfind(codeWords{ii}, '+'))
        wordLen = length(codeWords{ii});
        numCodeWords(wordLen, 2) = numCodeWords(wordLen, 2) + 1;
    elseif ~countedElseSymbol
        countedElseSymbol = true;
        actualCode = strsplit(codeWords{ii}, '+');
        actualCode = actualCode{1};
        wordLen = length(actualCode);
        numCodeWords(wordLen, 2) = numCodeWords(wordLen, 2) + 1;
    end
end

end

