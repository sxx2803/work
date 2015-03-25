clear all;

%% Problem 3 Part A

img = imread('Boy.tif');
img = double(img);

magVector = [64 32 16 8 6 4];

for magIdx=1:length(magVector)
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

    % Build the histogram

    maxMag = magVector(magIdx);
    %maxMag = 32;

    diffImgHist = zeros((2*maxMag)+2, 1);

    for ii=-maxMag:maxMag
        occurences = length(find(diffImg1d==ii));
        diffImgHist(ii+(maxMag+1)) = occurences;
    end

    % Number of entries for the ELSE symbol is the rest of the values
    diffImgHist(end) = numel(diffImg1d)-sum(diffImgHist);

    diffImgHistNorm = diffImgHist./(numel(diffImg));

    sym = -maxMag:(maxMag+1);

    % Perform Huffman coding
    codeTable = myhuffman(sym, diffImgHistNorm, 1);
    codeTable{end,1} = 'ELSE';

    % Get the average codeword length

    aveLen = 0;
    diffBitsRequired = floor(log2(abs(min(diffImg1d))))+1;
    pixelBitsRequired = 8;
    for ii=1:length(codeTable(:,1))
        if ii < (2*(maxMag)+2)
            aveLen = aveLen + diffImgHistNorm(ii)*length(codeTable{ii,2});
        else
            aveLen = aveLen + diffImgHistNorm(ii)*(length(codeTable{ii,2})+pixelBitsRequired);
        end
    end

    fprintf('Average codeword length for a maximum magnitude of %i is %5.4f bpp\n', maxMag, aveLen);

end