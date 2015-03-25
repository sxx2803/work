% Problem 3 Part B
clear all;

img = imread('Text-CCITT.tif');

maxRunLength = size(img, 2);
runCountHisto0 = zeros(maxRunLength, 1);
runCountHisto1 = zeros(maxRunLength, 1);
runCounts0 = [];
runCounts1 = [];

% Count the number of runs
for ii=1:size(img,1)
    runCount = 1;
    for jj=1:size(img,2)-1
        % Check if run continues
        if img(ii,jj+1) == img(ii,jj)
            runCount = runCount + 1;
        % Current run has ended, save and start a new run
        else
            if img(ii, jj) == 0
                %runCountHisto0(runCount) = runCountHisto0(runCount) + 1;
                runCounts0 = [runCounts0 runCount];
            else
                %runCountHisto1(runCount) = runCountHisto1(runCount) + 1;
                runCounts1 = [runCounts1 runCount];
            end
            runCount = 1;
        end
    end
    % End of line, save accordingly
    if img(ii, jj) == 0
        %runCountHisto0(runCount) = runCountHisto0(runCount) + 1;
        runCounts0 = [runCounts0 runCount];
    else
        %runCountHisto1(runCount) = runCountHisto1(runCount) + 1;
        runCounts1 = [runCounts1 runCount];
    end
end


%% Start doing Golomb coding and count the word lengths

% Init values
A = 5;
Nmax = 100;
N = 1;
totalLength0 = 0;
% Start counting the encoded codelengths for 0-bit runlengths
for ii = 1:length(runCounts0)
    r = runCounts0(ii);
    [codelength, A, N] = golombAdaptive(r-1, A, N, Nmax);
    totalLength0 = totalLength0 + codelength;
end

fprintf('Text-CCITT.tif: For Nmax = %i, runs of 0''s requires %i bits\n', Nmax, totalLength0);

% Init values
A = 5;
Nmax = 100;
N = 1;
totalLength1 = 0;
% Start counting the encoded codelengths for 1-bit runlengths
for ii = 1:length(runCounts1)
    r = runCounts1(ii);
    [codelength, A, N] = golombAdaptive(r-1, A, N, Nmax);
    totalLength1 = totalLength1 + codelength;
end

fprintf('Text-CCITT.tif: For Nmax = 100, runs of 1''s requires %i bits\n', totalLength1);

compressRatio = numel(img)/(totalLength0 + totalLength1);

fprintf('Text-CCITT.tif: Compression ratio is: %6.4f\n', compressRatio);