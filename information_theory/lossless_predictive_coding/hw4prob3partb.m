clear all;

%% Problem 3 Part B

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

category0 = [0];
category1 = [-1,1];
category2 = [-3:-2 2:3];
category3 = [-7:-4 4:7];
category4 = [-15:-8 8:15];
category5 = [-31:-16 16:31];
category6 = [-63:-32 32:63];
category7 = [-127:-64 64:127];
category8 = [-255:-128 128:255];

categoryHist = zeros(9, 1);

categoryCell = {category0, category1, category2, category3, category4, category5, category6, category7, category8};

for ii=1:length(categoryCell)
    for jj=1:length(categoryCell{ii})
        occurences = length(find(diffImg1d==categoryCell{ii}(jj)));
        categoryHist(ii) = categoryHist(ii) + occurences;
    end
end

categoryHistProb = categoryHist./sum(categoryHist);

sym = 0:8;

% Perform Huffman coding
codeTable = myhuffman(sym, categoryHistProb, 1);

% Compute the average codeword length
aveLen = 0;

for ii=1:length(categoryHistProb)
    prob = categoryHistProb(ii);
    aveLen = aveLen + prob*(length(codeTable{ii,2})+(ii-1));
end

% Display the output

fprintf('%-12s%-24sProbability\n','Category','Codeword + fixed bits');
for ii=1:length(categoryHistProb)
    codewordString = sprintf('%s + %i bits', codeTable{ii,2}, (ii-1));
    fprintf('%-12i%-24s%-7.6f%%\n', (ii-1), codewordString, categoryHistProb(ii)*100);
end

fprintf('Categorized average codeword length is %5.4f bpp\n', aveLen);