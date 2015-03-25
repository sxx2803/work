clear all;

%% Part A (No Source Extension)

p = [0.3 0.25 0.2 0.1 0.05 0.04 0.04 0.02];
s = 1:length(p);

codeTable = myhuffman(s, p, 1);

fprintf('Symbol\tCode Number\n');
disp(codeTable);

%% Part A (2nd Source Extension)

p2 = p'*p;
p2 = reshape(p2, 1, length(p)^2);
s2 = 1:length(p2);

codeTable = myhuffman(s2, p2, 2);

%fprintf('Symbol\tCode Number\n');
%disp(codeTable);

lenTable = countLengths(codeTable);

fprintf('Length\t# of Codewords\n');
disp(lenTable);

%% Part A (3rd Source Extension)

p3 = p2'*p;
p3 = reshape(p3, 1, length(p)^3);
s3 = 1:length(p3);

codeTable = myhuffman(s3,p3,3);

lenTable = countLengths(codeTable);

fprintf('Length\t# of Codewords\n');
disp(lenTable);

%% Part B (Huffman Encoding of Lena Image)

img = imread('Lena_Y.tif');
imgHist = imhist(img);
total = sum(sum(imgHist));
imgHistNorm = imgHist./total;

% Only want the non-zero probabilities
imgHistNorm = imgHistNorm(imgHistNorm > 0);

sym = 1:length(imgHistNorm);

codeTable = myhuffman(sym, imgHistNorm, 1);

lenTable = countLengths(codeTable);

fprintf('Length\t# of Codewords\n');
disp(lenTable);

%% Part C (NYC Image Huffman Encoding);

img = imread('NYC.tif');
imgHist = imhist(img);
total = sum(sum(imgHist));
imgHistNorm = imgHist./total;

% Only want the non-zero probabilities
imgHistNorm = imgHistNorm(imgHistNorm > 0);

sym = 1:length(imgHistNorm);

codeTable = myhuffman(sym, imgHistNorm, 1);

lenTable = countLengths(codeTable);

fprintf('Length\t# of Codewords\n');
disp(lenTable);