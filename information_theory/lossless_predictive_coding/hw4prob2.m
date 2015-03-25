clear all;

%% Part A (Local Huffman)

img = imread('Boy.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the row or column
        if ii == 1 && jj == 1
            prediction = 128;
        % First row
        elseif ii == 1
            prediction = img(ii, jj-1);
        % First column
        elseif jj == 1
            prediction = img(ii-1, jj);
        else
            a = img(ii, jj-1);
            b = img(ii-1, jj);
            c = img(ii-1, jj-1);
            if c >= max([a, b])
                prediction = min([a,b]);
            elseif c <= min([a, b])
                prediction = max([a,b]);
            else
                prediction = a+b-c;
            end
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the histograms

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgUnique = unique(diffImg1d);
diffImgHist = zeros(length(diffImgUnique), 1);

for ii=1:length(diffImgUnique)
    occurences = length(find(diffImg1d==diffImgUnique(ii)));
    diffImgHist(ii) = occurences;
end

diffImgHistNorm = diffImgHist./(numel(diffImg));
entropyImg = -sum(diffImgHistNorm.*log2(diffImgHistNorm));
entropies(9) = entropyImg;
fprintf('Entropy for predictor "JPEG-LS Median Predictor" = %5.4f\n', entropyImg)

% Do a local 2-pass Huffman coding
sym = 1:length(diffImgHistNorm);

[codeTable, aveLen] = myhuffman(sym, diffImgHistNorm, 1);

fprintf('Local Huffman encoding average length: %6.4f\n', aveLen);

%% Part B (Global Huffman)

img = imread('Parrots.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the row or column
        if ii == 1 && jj == 1
            prediction = 128;
        % First row
        elseif ii == 1
            prediction = img(ii, jj-1);
        % First column
        elseif jj == 1
            prediction = img(ii-1, jj);
        else
            a = img(ii, jj-1);
            b = img(ii-1, jj);
            c = img(ii-1, jj-1);
            if c >= max([a, b])
                prediction = min([a,b]);
            elseif c <= min([a, b])
                prediction = max([a,b]);
            else
                prediction = a+b-c;
            end
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgAll = diffImg1d;
diffImgUnique = unique(diffImg1d);

% Repeat for lighthouse image
img = imread('Lighthouse.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the row or column
        if ii == 1 && jj == 1
            prediction = 128;
        % First row
        elseif ii == 1
            prediction = img(ii, jj-1);
        % First column
        elseif jj == 1
            prediction = img(ii-1, jj);
        else
            a = img(ii, jj-1);
            b = img(ii-1, jj);
            c = img(ii-1, jj-1);
            if c >= max([a, b])
                prediction = min([a,b]);
            elseif c <= min([a, b])
                prediction = max([a,b]);
            else
                prediction = a+b-c;
            end
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgAll = [diffImgAll; diffImg1d];
diffImgUnique = [diffImgUnique; unique(diffImg1d)];

% Repeat for Rafting image
img = imread('Rafting.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the row or column
        if ii == 1 && jj == 1
            prediction = 128;
        % First row
        elseif ii == 1
            prediction = img(ii, jj-1);
        % First column
        elseif jj == 1
            prediction = img(ii-1, jj);
        else
            a = img(ii, jj-1);
            b = img(ii-1, jj);
            c = img(ii-1, jj-1);
            if c >= max([a, b])
                prediction = min([a,b]);
            elseif c <= min([a, b])
                prediction = max([a,b]);
            else
                prediction = a+b-c;
            end
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgAll = [diffImgAll; diffImg1d];
diffImgUnique = [diffImgUnique; unique(diffImg1d)];

diffImgUnique = unique(diffImgUnique);

% Need to compute the residual image for the Boy image first as the min and
% max values may be larger than the differential values found in the other
% images
img = imread('Boy.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the row or column
        if ii == 1 && jj == 1
            prediction = 128;
        % First row
        elseif ii == 1
            prediction = img(ii, jj-1);
        % First column
        elseif jj == 1
            prediction = img(ii-1, jj);
        else
            a = img(ii, jj-1);
            b = img(ii-1, jj);
            c = img(ii-1, jj-1);
            if c >= max([a, b])
                prediction = min([a,b]);
            elseif c <= min([a, b])
                prediction = max([a,b]);
            else
                prediction = a+b-c;
            end
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

diffImg1d = reshape(diffImg, numel(diffImg), 1);

minUnique = min(diffImg1d);
maxUnique = max(diffImg1d);

diffImgHist = [minUnique:maxUnique]';

diffImgHist(:,2) = zeros(length(diffImgHist),1);

for ii=1:length(diffImgUnique)
    occurences = length(find(diffImgAll==diffImgUnique(ii)));
    histBins = diffImgHist(:,1);
    idx = find(histBins==diffImgUnique(ii),1);
    diffImgHist(idx, 2) = occurences;
end

% Set minimum histogram value to 1
histValues = diffImgHist(:,2);
histValues(histValues==0) = 1;
diffImgHist = [diffImgHist(:,1), histValues];

%%

% Get the codewords

histValues = diffImgHist(:,2);
histValuesNorm = histValues./sum(histValues);
symbols = diffImgHist(:,1);

[codeTable, aveLen] = myhuffman(symbols, histValuesNorm, 1);

% Use the codetable to encode the Boy image
encodingLength = 0;

for ii=1:length(diffImg1d)
    idx = find(symbols==diffImg1d(ii), 1);
    codeword = codeTable{idx, 2};
    encodingLength = encodingLength + length(codeword);
end

encodingAveLength = encodingLength/numel(diffImg);

fprintf('Global Huffman encoding average length: %6.4f\n', encodingAveLength);
