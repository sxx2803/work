clear all;

% Generate sample data
rng(1);
testData = normrnd(0, 1, 500000, 1);

% Init some stuff
epsilon = 5e-4;
numLevels = 11;
boundaries = zeros(12,1);
boundaries(1) = -Inf;
boundaries(end) = Inf;
qLevels = zeros(11, 1);
stepSize = 1.0;
learningRate = 0.3;

doAgain = true;
wantDistortion = 0.0345;
curDistortion = 0;

iteration = 1;
maxIteration = 400;

while(doAgain)
    % Compute the decision boundaries from the step size and assuming a
    % forced zero reconstruction level
    maxMin = numLevels * stepSize / 2;
    boundaries = -maxMin : stepSize : maxMin;
    boundaries(1) = -Inf;
    boundaries(end) = Inf;
    
    % Compute the centroids for each interval
    for ii=1:length(qLevels)
        boundedVals = testData(boundaries(ii)<testData & testData<boundaries(ii+1));
        qLevels(ii) = sum(boundedVals) / numel(boundedVals);
        curDistortion = curDistortion + sum((boundedVals - qLevels(ii)).^2);
    end
    curDistortion = curDistortion / numel(testData);
    
    % Compare the distortion against the "theoretical" distortion
    if (abs((wantDistortion-curDistortion)/curDistortion) <= epsilon) || iteration == maxIteration
        doAgain = false;
    else
        % Do iterative gradient descent
        stepSize = stepSize - (curDistortion - wantDistortion)*learningRate;
        curDistortion = 0;
        iteration = iteration + 1;
    end
end;


% Find the entropy
myEntropy = 0;
for ii=1:length(qLevels)
    boundedVals = testData(boundaries(ii)<testData & testData<boundaries(ii+1));
    prob = numel(boundedVals)/numel(testData);
    myEntropy = myEntropy - (prob*log2(prob));
end

fprintf('The ideal step size is Q = %3.2f\n', stepSize);
fprintf('The entropy for UTQ is H = %3.2f bps\n', myEntropy);