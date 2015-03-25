clear all;

%% Part A (No Source Extension)

p = [0.3 0.25 0.2 0.1 0.05 0.04 0.04 0.02];
s = 1:length(p);

codeTable = modifiedHuffman(s, p, 0.15);

fprintf('Symbol\tCode Number\n');
disp(codeTable);

lenTable = countLengths(codeTable);

fprintf('Length\t# of Codewords\n');
disp(lenTable);

%% Part B (Huffman Encoding of NYC Image)

img = imread('NYC.tif');
imgHist = imhist(img);
total = sum(sum(imgHist));
imgHistNorm = imgHist./total;

% Only want the non-zero probabilities
imgHistNorm = imgHistNorm(imgHistNorm > 0);

sym = 1:length(imgHistNorm);

codeTable = modifiedHuffman(sym, imgHistNorm, 0.03);

lenTable = countLengths(codeTable);

fprintf('Length\t# of Codewords\n');
disp(lenTable);