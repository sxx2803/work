function [codes] = assignCodes( curNode, initAssignment )
% assignCodes(curNode, initAssignment) - Recursive function that traverses 
% the cell representation of the Huffman binary tree and assigns binary 
% codes to each of the leaf nodes within the tree
%
% Parameters:
%   curNode - the current node in the tree
%   initAssignment - the code assignment so far
%
% Output:
%   codes - a nx2 cell array where the first column is the symbol name and
%           the second column is the code assignment
codes = {};

leftCode = strcat(initAssignment, '1');
rightCode = strcat(initAssignment, '0');

if isa(curNode{1}, 'cell')
    codes = [codes assignCodes(curNode{1}, leftCode)];
else
    codes = [codes curNode{1} leftCode];
end

if isa(curNode{2}, 'cell')
    codes = [codes assignCodes(curNode{2}, rightCode)];
else
    codes = [codes curNode{2} rightCode];
end



