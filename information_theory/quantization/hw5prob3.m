clear all;

% Generate sample data
rng(1);
testData = normrnd(0, 1, 500000, 1);

% Initialize decision boundaries and reconstruction levels
boundaries = [-Inf, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, Inf];
qLevels = zeros(1,8);

% Initialize iterator count
n = 1;

epsilon = 5e-4;
curDistortionIntervals = zeros(1,8);
curDistortion = 0;
prevDistortion = 0;
doAgain = true;

while(doAgain)
    % Find the new reconstruction thresholds
    for ii=1:length(qLevels)
        topVal = sum(testData(boundaries(ii)<testData & testData<boundaries(ii+1)));
        botVal = numel(testData(boundaries(ii)<testData & testData<boundaries(ii+1)));
        qLevels(ii) = topVal / botVal;
    end
    
    % Find the new decision boundaries
    for ii=2:length(boundaries)-1
        boundaries(ii) = (qLevels(ii-1) + qLevels(ii))/2;
    end
    
    % Calculate the total distortion
    for ii=1:length(qLevels)
        boundedVals = testData(boundaries(ii)<testData & testData<boundaries(ii+1));
        intervalDistortion = sum((boundedVals - qLevels(ii)).^2);
        curDistortionIntervals(ii) = intervalDistortion;
        curDistortion = curDistortion + intervalDistortion;
    end
    % Normalize the distortion to the number of elements
    curDistortion = curDistortion / numel(testData);
    curDistortionIntervals = curDistortionIntervals / numel(testData) / curDistortion;
    
    % Check against previous distortion
    if abs((prevDistortion-curDistortion)/curDistortion) <= epsilon
        doAgain = false;
    else
        n = n+1;
        prevDistortion = curDistortion;
        curDistortion = 0;
    end   
end

% Find the entropy
gaussEntropy = 0;
for ii=1:length(qLevels)
    boundedVals = testData(boundaries(ii)<testData & testData<boundaries(ii+1));
    prob = numel(boundedVals)/numel(testData);
    gaussEntropy = gaussEntropy - (prob*log2(prob));
end

fprintf('The empirical boundaries are: \n'), disp(boundaries');
fprintf('The empirical reconstruction levels are: \n'), disp(qLevels');
for ii=1:length(qLevels)
    fprintf('For empirical boundaries (%5.4f, %5.4f): \n', boundaries(ii), boundaries(ii+1));
    fprintf('The empirical reconstruction level is %5.4f\n', qLevels(ii));
    fprintf('The contribution by the boundary to the MSE is: %3.1f%%\n', curDistortionIntervals(ii)*100);
    fprintf('\n');
end

fprintf('The final distortion is %6.5f\n', curDistortion);
fprintf('The entropy of the quantizer output is %5.4f\n', gaussEntropy);
fprintf('The quantizer needs 3-bits to encode 8 levels, thus R = 3\n');

