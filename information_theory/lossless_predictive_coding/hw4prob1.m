clear all;

predictorStrings = {'No prediction', 'A only', 'B only', 'C only', 'A+B-C', 'A+(B-C)/2', 'B+(A-C)/2', '(A+B)/2', 'JPEG-LS Median'};
entropies = zeros(1, length(predictorStrings));

%% No prediction entropy

img = imread('Boy.tif');
imgHist = imhist(img);
total = sum(sum(imgHist));
imgHistNorm = imgHist./total;

% Only want the non-zero probabilities
imgHistNorm = imgHistNorm(imgHistNorm > 0);

imgEntropy = -sum(imgHistNorm.*log2(imgHistNorm));
entropies(1) = imgEntropy;
fprintf('Entropy for no prediction = %5.4f\n', imgEntropy);

%% Using "A" prediction only

img = imread('Boy.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the line
        if jj == 1
            prediction = 128;
        % All other pixels
        else
            prediction = img(ii, jj-1);
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
entropies(2) = entropyImg;
fprintf('Entropy for predictor "A" only = %5.4f\n', entropyImg)

%% Using "B" prediction only

img = imread('Boy.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the row
        if ii == 1 && jj == 1
            prediction = 128;
        elseif ii == 1
            prediction = img(ii, jj-1);
        else
            prediction = img(ii-1, jj);
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
entropies(3) = entropyImg;
fprintf('Entropy for predictor "B" only = %5.4f\n', entropyImg)

%% Using "C" prediction only

img = imread('Boy.tif');
img = double(img);
% Calculate the differential image
diffImg = zeros(size(img,1), size(img, 2));

for ii=1:size(img, 1)
    for jj=1:size(img, 2)
        % First pixel in the row or column
        if ii == 1 && jj == 1
            prediction = 128;
        elseif ii == 1
            prediction = img(ii, jj-1);
        elseif jj == 1
            prediction = img(ii-1, jj);
        else
            prediction = img(ii-1, jj-1);
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgUnique = unique(diffImg1d);
diffImgHist = zeros(length(diffImgUnique), 1);

for ii=1:length(diffImgUnique)
    occurences = length(find(diffImg1d==diffImgUnique(ii)));
    diffImgHist(ii) = occurences;
end

diffImgHistNorm = diffImgHist./(numel(diffImg));
entropyImg = -sum(diffImgHistNorm.*log2(diffImgHistNorm));
entropies(4) = entropyImg;
fprintf('Entropy for predictor "C" only = %5.4f\n', entropyImg)

%% Using "A+B-C" predictor

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
            prediction = img(ii, jj-1) + img(ii-1, jj) - img(ii-1, jj-1);
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgUnique = unique(diffImg1d);
diffImgHist = zeros(length(diffImgUnique), 1);

for ii=1:length(diffImgUnique)
    occurences = length(find(diffImg1d==diffImgUnique(ii)));
    diffImgHist(ii) = occurences;
end

diffImgHistNorm = diffImgHist./(numel(diffImg));
entropyImg = -sum(diffImgHistNorm.*log2(diffImgHistNorm));
entropies(5) = entropyImg;
fprintf('Entropy for predictor "A+B-C" = %5.4f\n', entropyImg)

%% Using "A+(B-C)/2" predictor

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
            prediction = floor(img(ii, jj-1) + (img(ii-1, jj) - img(ii-1, jj-1))/2.0);
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgUnique = unique(diffImg1d);
diffImgHist = zeros(length(diffImgUnique), 1);

for ii=1:length(diffImgUnique)
    occurences = length(find(diffImg1d==diffImgUnique(ii)));
    diffImgHist(ii) = occurences;
end

diffImgHistNorm = diffImgHist./(numel(diffImg));
entropyImg = -sum(diffImgHistNorm.*log2(diffImgHistNorm));
entropies(6) = entropyImg;
fprintf('Entropy for predictor "A+(B-C)/2" = %5.4f\n', entropyImg)

%% Using "B+(A-C)/2" predictor

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
            prediction = floor(img(ii-1, jj) + (img(ii, jj-1) - img(ii-1, jj-1))/2.0);
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgUnique = unique(diffImg1d);
diffImgHist = zeros(length(diffImgUnique), 1);

for ii=1:length(diffImgUnique)
    occurences = length(find(diffImg1d==diffImgUnique(ii)));
    diffImgHist(ii) = occurences;
end

diffImgHistNorm = diffImgHist./(numel(diffImg));
entropyImg = -sum(diffImgHistNorm.*log2(diffImgHistNorm));
entropies(7) = entropyImg;
fprintf('Entropy for predictor "B+(A-C)/2" = %5.4f\n', entropyImg)

%% Using "(A+B)/2" predictor

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
            prediction = floor((img(ii-1, jj) + img(ii, jj-1))/2.0);
        end
        diffImg(ii, jj) = img(ii, jj) - prediction;
    end
end

% Build the normalized histogram

diffImg1d = reshape(diffImg, numel(diffImg), 1);
diffImgUnique = unique(diffImg1d);
diffImgHist = zeros(length(diffImgUnique), 1);

for ii=1:length(diffImgUnique)
    occurences = length(find(diffImg1d==diffImgUnique(ii)));
    diffImgHist(ii) = occurences;
end

diffImgHistNorm = diffImgHist./(numel(diffImg));
entropyImg = -sum(diffImgHistNorm.*log2(diffImgHistNorm));
entropies(8) = entropyImg;
fprintf('Entropy for predictor "(A+B)/2" = %5.4f\n', entropyImg)

%% Using JPEG-LS Median predictor

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

% Build the normalized histogram

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

%% Display output
% Sort the entropies
[entropies, idx] = sort(entropies);
predictorStrings = predictorStrings(idx);

output = [predictorStrings' num2cell(entropies')];

fprintf('Sorted Entropies\n');
disp(output);