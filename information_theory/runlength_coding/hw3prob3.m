% Problem 3
clear all;

img = imread('Text-CCITT.tif');

maxRunLength = size(img, 2);
runCountHisto0 = zeros(maxRunLength, 1);
runCountHisto1 = zeros(maxRunLength, 1);
% If the run count is positive, it's a run of white pixels. If it's negative,
% it's a run of black pixels.
runCounts = [];

% Count the number of runs
for ii=1:size(img,1)
    runCount = 1;
    for jj=1:size(img,2)-1
        % Check if run continues
        if img(ii,jj+1) == img(ii,jj)
            runCount = runCount + 1;
        % Current run has ended, save and start a new run
        else
            % A black run
            if img(ii, jj) == 0
                %runCountHisto0(runCount) = runCountHisto0(runCount) + 1;
                %runCounts0 = [runCounts0 runCount];
                runCounts = [runCounts -runCount];
            % A white run
            else
                %runCountHisto1(runCount) = runCountHisto1(runCount) + 1;
                %runCounts1 = [runCounts1 runCount];
                runCounts = [runCounts runCount];
            end
            runCount = 1;
        end
    end
    % End of line, save accordingly
    if img(ii, jj) == 0
        %runCountHisto0(runCount) = runCountHisto0(runCount) + 1;
        %runCounts0 = [runCounts0 runCount];
        runCounts = [runCounts -runCount];
    else
        %runCountHisto1(runCount) = runCountHisto1(runCount) + 1;
        %runCounts1 = [runCounts1 runCount];
        runCounts = [runCounts runCount];
    end
end

% Initialize the coder parameters
Awht= 5;
Ablk = 5;
Nwht = 1;
Nblk = 1;
Nmax = 100;
imgEncoded = cell(length(runCounts), 1);

% Start encoding

for ii=1:length(runCounts)
    % If run of black pixels
    if runCounts(ii) < 0
        runLength = -runCounts(ii);
        [codeWord Ablk Nblk] = golombAdaptiveEncode(runLength, Ablk, Nblk, Nmax);
        imgEncoded{ii} = [codeWord 'b'];
    % If run of white pixels
    else
        runLength = runCounts(ii);
        [codeWord Awht Nwht] = golombAdaptiveEncode(runLength, Awht, Nwht, Nmax);
        imgEncoded{ii} = [codeWord 'w'];
    end
end

% Initialize the decoder parameters
Awht= 5;
Ablk = 5;
Nwht = 1;
Nblk = 1;
Nmax = 100;

decodedRunlengths = zeros(length(runCounts), 1);
decodedImg = zeros(numel(img), 1);
pixelIdx = 0;

% Start decoding

for ii=1:length(imgEncoded)
    codeWord = imgEncoded{ii}(1:end-1);
    bw = imgEncoded{ii}(end);
    % If the encoded run is a white pixel run
    if bw == 'w'
        [decoded Awht Nwht] = golombAdaptiveDecode(codeWord, Awht, Nwht, Nmax);
        decodedImg(pixelIdx+1:pixelIdx+decoded) = 1;
        pixelIdx = pixelIdx + decoded;
        decodedRunlengths(ii) = decoded;
    % Otherwise the encoded run is a black pixel run
    else
        [decoded Ablk Nblk] = golombAdaptiveDecode(codeWord, Ablk, Nblk, Nmax);
        decodedImg(pixelIdx+1:pixelIdx+decoded) = 0;
        pixelIdx = pixelIdx + decoded;
        decodedRunlengths(ii) = -decoded;
    end
end

% Reshape the vector into the image dimensions
shapedImg = reshape(decodedImg, size(img, 2), size(img, 1));
shapedImg = shapedImg';

diff = img - shapedImg;
diff = sum(sum(diff));

fprintf('Number of differences between the original and decoded image: %i\n', diff);