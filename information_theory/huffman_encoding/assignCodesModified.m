function [codes] = assignCodesModified( curNode, initAssignment )
% assignCodesModified(curNode, initAssignment) - Recursive function that traverses 
% the cell representation of the Huffman tree and assigns binary 
% codes to each of the leaf nodes within the tree. Accounts for the ELSE
% symbol.
%
% Parameters:
%   curNode - the current node in the tree
%   initAssignment - the code assignment so far
%
% Output:
%   codes - a nx2 cell array where the first column is the symbol name and
%           the second column is the code assignment
codes = {};

% Check for the number of children. If it's 2, it's a regular node. If it's
% more than 2, it's the ELSE symbol.
if length(curNode) == 2
    leftCode = strcat(initAssignment, '1');
    rightCode = strcat(initAssignment, '0');
    
    if isa(curNode{1}, 'cell')
    codes = [codes assignCodesModified(curNode{1}, leftCode)];
    else
        codes = [codes curNode{1} leftCode];
    end

    if isa(curNode{2}, 'cell')
        codes = [codes assignCodesModified(curNode{2}, rightCode)];
    else
        codes = [codes curNode{2} rightCode];
    end

else
    numBits = ceil(log2(length(curNode)));
    for ii=(0:length(curNode)-1)
        elseCode = strcat(initAssignment, '+', dec2bin(ii, numBits));
        codes = [codes curNode{ii+1} elseCode];
    end
    
end




