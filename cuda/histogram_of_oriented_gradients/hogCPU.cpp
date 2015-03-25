/*
J Nicolas Schrading jxs8172@rit.edu
Sicheng Xu sxx2803@g.rit.edu
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctime> // time(), clock()
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <math.h>
#include "hog.h"

using namespace std;

#define KERNEL_SIZE 3
#define CELL_SIZE 8
#define HISTO_SIZE 9
#define BLOCK_SIZE 2
#define ITERS 10
#define SMALL_ITERS 100

/**
 * Performs convolution in the X direction on the input matrix using the kernel
 * Input:   in      : 2D input matrix
 *          out     : 2D output matrix
 *          kernel  : 1D vector
 *          size    : size of square matrix
 */
void conv1X(int *in, int *kernel, int *out, int size) {

    for(int y = 0; y < size; y++){
        for(int x = 0; x < size; x++){
            out[y*size+x] = 0;
            for(int k = 0; k < KERNEL_SIZE; k++){
                if((x-k) < 0){
                    continue;
                }
                out[y*size+x] += (in[y*size+x-k] * kernel[k]);
            }
        }
    }
}

/**
 * Performs convolution in the Y direction on the input matrix using the kernel
 * Input:   in      : 2D input matrix
 *          out     : 2D output matrix
 *          kernel  : 1D vector
 *          size    : size of square matrix
 */
void conv1Y(int *in, int *kernel, int *out, int size) {

    for(int x = 0; x < size; x++){
        for(int y = 0; y < size; y++){
            out[y*size+x] = 0;
            for(int k = 0; k < KERNEL_SIZE; k++){
                if((y-k) < 0){
                    continue;
                }
                out[y*size+x] += (in[(y-k)*size+x] * kernel[k]);
            }
        }
    }
}

/**
 * Normalizes a matrix between 0 and 255. Modifies in place
 * Input:   in      : 2D input matrix
 *          size    : size of square matrix
 */
void mat2gray(int *in, int size){
    int max = 0;
    int min = 0;
    // find max and min
    for(int i = 0; i < size*size; i++){
        if((in[i]) > max){
            max = (in[i]);
        }
        else if((in[i]) < min){
            min = (in[i]);
        }
    }
    // normalize
    for(int i = 0; i < size*size; i++){
        in[i] = (int) ((((float)in[i] - min) / ((float)max - (float)min)) * 255.0f);
    }

}

/**
 * Reads in a file, returns the size of the file
 * Input:   filename        : name of the file
 *          imgData : the 2D matrix of the data in the file
 */
int loadFile(const char* filename, int **imgData){

    FILE* imgFile = fopen(filename, "rb");
    if(!imgFile){
        // Could not open image file
        return 0;
    }
    // Get size in bytes of file
    fseek(imgFile, 0, SEEK_END);
    int filesize = ftell(imgFile);
    // 1 byte for each pixel, so image size is sqrt(filesize)
    int imgSize = floor(sqrt((double)filesize));
    fseek(imgFile, 0, SEEK_SET);

    int* imgBuffer = new int[imgSize*imgSize];

    unsigned char buf;
    // Go through the file and read elements
    for(int i = 0; i < imgSize*imgSize; i++){
        fread(&buf, 1, 1, imgFile);
        imgBuffer[i] = (int) buf;
    }

    *imgData = imgBuffer;
    return imgSize;
}

/**
 * Calculates the histogram values for a single block of the image
 * Input:   blockX      : vector of values in the block, from the X convolution
 *          blockY      : vector of values in the block, from the Y convolution
 *          hist        : vector of histogram values (9 bins)
 */
void calculateHistoBlock(int* blockX, int* blockY, float* hist) {
    const float PI = 3.141592653589793238463;
    for(int i = 0; i < CELL_SIZE * CELL_SIZE; i++) {
        // calculate magnitude and orientation of the gradient
        float magnitude = sqrt((float)(blockX[i] * blockX[i] + blockY[i] * blockY[i]));
        float orientation = atan2((float)blockY[i], (float)blockX[i]);

        // convert to positive angle if negative
        if(orientation < 0) {
            orientation += PI;
        }

        float bin1 = PI / 9;
        float bin2 = (2 * PI) / 9;
        float bin3 = PI / 3;
        float bin4 = (4 * PI) / 9;
        float bin5 = (5 * PI) / 9;
        float bin6 = (2 * PI) / 3;
        float bin7 = (7 * PI) / 9;
        float bin8 = (8 * PI) / 9;
        float bin9 = PI;

        // place value in appropriate bin based off of orientation
        if(orientation <= bin1) {
            hist[0] += magnitude;
        }
        else if(orientation <= bin2 && orientation > bin1) {
            hist[1] += magnitude;
        }
        else if(orientation <= bin3 && orientation > bin2) {
            hist[2] += magnitude;
        }
        else if(orientation <= bin4 && orientation > bin3) {
            hist[3] += magnitude;
        }
        else if(orientation <= bin5 && orientation > bin4) {
            hist[4] += magnitude;
        }
        else if(orientation <= bin6 && orientation > bin5) {
            hist[5] += magnitude;
        }
        else if(orientation <= bin7 && orientation > bin6) {
            hist[6] += magnitude;
        }
        else if(orientation <= bin8 && orientation > bin7) {
            hist[7] += magnitude;
        }
        else if(orientation <= bin9 && orientation > bin8) {
            hist[8] += magnitude;
        }
    }   
}

/**
 * Normalize overlapping histogram blocks
 * Input:   histograms      : vector of input histograms
 *          size            : size of the image
 *          histogramsOut   : vector of output histograms
 */
vector<float> normalizeBlocks(vector<float*> histograms, int size, vector<float*> histogramsOut) {
    vector<float> featureVector;
    int numCells = size/CELL_SIZE;
    // loop over vertical axis
    for(int i = 0 ; i < (2 * (numCells / BLOCK_SIZE) - 1); i++){
        // loop over horizontal axis
        for(int j = 0; j < (2 * (numCells / BLOCK_SIZE) - 1); j++){
            float* cellArray[BLOCK_SIZE*BLOCK_SIZE] = {};
            float l2norm = 0.0;
            // get cells from block
            for(int k = 0; k < BLOCK_SIZE; k++){
                for(int l = 0; l < BLOCK_SIZE; l++){
                    float* cell = histograms.at((k+i)*BLOCK_SIZE+(l+j));
                    for(int histIdx = 0; histIdx < HISTO_SIZE; histIdx++){
                        // compute l2 norm of current block
                        l2norm += cell[histIdx] * cell[histIdx];
                    }
                }
            }
            l2norm = sqrt(l2norm);
            for(int k = 0; k < BLOCK_SIZE; k++){
                for(int l = 0; l < BLOCK_SIZE; l++){
                    float* cell = histograms.at((k+i)*BLOCK_SIZE+(l+j));
                    float* outCell = histogramsOut.at((k+i)*BLOCK_SIZE+(l+j));
                    for(int histIdx = 0; histIdx < HISTO_SIZE; histIdx++){
                        featureVector.push_back(cell[histIdx] / l2norm);
                        outCell[histIdx] = cell[histIdx] / l2norm;
                    }
                }
            }
        }
    }

    return featureVector;
}

/**
 * Writes the feature vectors from HOG to a file
 * Input:   featureVector       : vector of features
 */
void writeFeatureVector(vector<float> featureVector) {
    FILE* f = fopen("featvec.txt", "w+");

    for(int i = 0; i < featureVector.size(); i++) {
        fprintf(f, "%f\n", featureVector.at(i));
    }
}

/**
 * Writes the feature vectors from HOG to a file
 * Input:   featureVector       : vector of features
 */
void writeHOGFeats(vector<float*> histograms) {
    FILE* f = fopen("histogramsNewv2.txt", "w+");

    for(int i = 0; i < histograms.size(); i++) {
        for(int j = 0; j < HISTO_SIZE; j++) {
            fprintf(f, "%f ", histograms.at(i)[j]);
        }
        fprintf(f, "\n");
    }
}

/**
 * Writes an image to a file
 * Input:   filename        : name of the file to write
 *          imageData       : data of the image
 *          size            : size of the image (square dimension)
 */
int writeImage(const char* filename, int *imageData, int size){

    FILE* imgFile = fopen(filename, "wb");
    if(!imgFile){
        return 0;
    }
    fwrite(imageData, sizeof(int), size*size, imgFile); 
}

/**
 * Writes a vector of histograms to a file
 * Input:   filename        : name of the file to write
 *          histData        : data of the histograms
 *          size            : how many histograms
 */
int writeHisto(const char* filename, float *histData, int size){

    FILE* f = fopen("histo.txt", "w+");

    for(int i = 0; i < 256; i++) {
        for(int j = 0; j < HISTO_SIZE; j++) {
            fprintf(f, "%f ", histData[i * HISTO_SIZE + j]);
        }
        fprintf(f, "\n");
    }
    return 0;
}

/**
 * Calculate the histograms for all blocks in the image, returns vector of them
 * Input:   outX        : the X convolution results
 *          outY        : the Y convolution results
 *          size        : how many histograms
 */
vector<float*> calculateHistos(int* outX, int* outY, int size) {
    int blockX[CELL_SIZE * CELL_SIZE] = {0};
    int blockY[CELL_SIZE * CELL_SIZE] = {0};

    vector<float*> histograms;
    int i = 0;
    for(int xStart = 0; xStart < size; xStart += CELL_SIZE) {
        for(int yStart = 0; yStart < size; yStart += CELL_SIZE) {
            for(int y = yStart; y < yStart + CELL_SIZE; y++) {
                for(int x = xStart; x < xStart + CELL_SIZE; x++) {
                    blockX[i] = outX[y*size + x];
                    blockY[i] = outY[y*size + x];
                    i++;
                }
            }
            float* hist = new float[HISTO_SIZE]();
            calculateHistoBlock(blockX, blockY, hist);
            i = 0;
            histograms.push_back(hist);
        }
    }
    return histograms;
}

/**
 * Compute HOG on the CPU
 * Input:   size        : the size of the image's dimensions
 *          imgData     : the pixels of the image
 *          kernel      : the kernel to perform convolution
 */
void HOG(int size, int* imgData, int* kernel) {
    int* outX = new int[size*size];
    int* outY = new int[size*size];

    // do x and y convolution
    conv1X(imgData, kernel, outX, size);
    conv1Y(imgData, kernel, outY, size);
    // compute histograms from the gradients
    vector<float*> histograms = calculateHistos(outX, outY, size);

    // code for future improvements...
    //vector<float*> histogramsCopy;
    //for(int i = 0; i < histograms.size(); i++){
    //  histogramsCopy.push_back(new float[HISTO_SIZE]());
    //  memcpy(histogramsCopy[i], histograms[i], HISTO_SIZE*sizeof(float));
    //}
    //vector<float> featureVector = normalizeBlocks(histograms, size, histogramsCopy);
    //writeFeatureVector(featureVector);
    //writeHOGFeats(histograms);

    // Free that memory!
    for(int i = 0; i < histograms.size(); i++) {
        float* histo = histograms.at(i);
        delete[] histo;
    }
    delete[] outX;
    delete[] outY;
}

int main() {
    // test CPU and GPU code and time from 128 to 4096 going by powers of 2

    int kernel[KERNEL_SIZE] = {-1, 0, 1};
    int* imgData = 0;
    int size = loadFile("confu128.raw", &imgData);
    std::cout << size << "x" << size << "image: " << std::endl;

    float tcpu, tgpu;
    clock_t start, end;

    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOG(size, imgData, kernel);
    }
    end = clock();
    tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per CPU iteration is: " << tcpu << " ms" << std::endl << std::endl;
    
    // warmup
    HOGGPU(size, imgData, kernel);
    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOGGPU(size, imgData, kernel);
    }
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per GPU iteration is: " << tgpu << " ms" << std::endl << std::endl;
    
    //----------------------------------------------------------------------------------

    size = loadFile("confu256.raw", &imgData);
    std::cout << size << "x" << size << "image: " << std::endl;

    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOG(size, imgData, kernel);
    }
    end = clock();
    tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per CPU iteration is: " << tcpu << " ms" << std::endl << std::endl;
    
    // warmup
    HOGGPU(size, imgData, kernel);
    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOGGPU(size, imgData, kernel);
    }
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per GPU iteration is: " << tgpu << " ms" << std::endl << std::endl;

    //----------------------------------------------------------------------------------

    size = loadFile("confu512.raw", &imgData);
    std::cout << size << "x" << size << "image: " << std::endl;

    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOG(size, imgData, kernel);
    }
    end = clock();
    tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per CPU iteration is: " << tcpu << " ms" << std::endl << std::endl;
    
    // warmup
    HOGGPU(size, imgData, kernel);
    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOGGPU(size, imgData, kernel);
    }
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per GPU iteration is: " << tgpu << " ms" << std::endl << std::endl;

    //----------------------------------------------------------------------------------

    size = loadFile("confu1024.raw", &imgData);
    std::cout << size << "x" << size << "image: " << std::endl;

    start = clock();
    for (int i = 0; i < ITERS; i++) {
        HOG(size, imgData, kernel);
    }
    end = clock();
    tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per CPU iteration is: " << tcpu << " ms" << std::endl << std::endl;
    
    // warmup
    HOGGPU(size, imgData, kernel);
    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOGGPU(size, imgData, kernel);
    }
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per GPU iteration is: " << tgpu << " ms" << std::endl << std::endl;

    //----------------------------------------------------------------------------------

    size = loadFile("confu2048.raw", &imgData);
    std::cout << size << "x" << size << "image: " << std::endl;

    start = clock();
    for (int i = 0; i < ITERS; i++) {
        HOG(size, imgData, kernel);
    }
    end = clock();
    tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per CPU iteration is: " << tcpu << " ms" << std::endl << std::endl;
    
    // warmup
    HOGGPU(size, imgData, kernel);
    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOGGPU(size, imgData, kernel);
    }
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per GPU iteration is: " << tgpu << " ms" << std::endl << std::endl;

    //----------------------------------------------------------------------------------

    size = loadFile("confu4096.raw", &imgData);
    std::cout << size << "x" << size << "image: " << std::endl;

    start = clock();
    for (int i = 0; i < ITERS; i++) {
        HOG(size, imgData, kernel);
    }
    end = clock();
    tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per CPU iteration is: " << tcpu << " ms" << std::endl << std::endl;

    // warmup
    HOGGPU(size, imgData, kernel);
    start = clock();
    for (int i = 0; i < SMALL_ITERS; i++) {
        HOGGPU(size, imgData, kernel);
    }
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    std::cout << "Average time per GPU iteration is: " << tgpu << " ms" << std::endl << std::endl;
    return 0;

}