#ifndef HOG_H
#define HOG_H

int writeImage(const char* filename, int *imageData, int size);

int writeHisto(const char* filename, float *histData, int size);

void HOG(int size, int* imgData, int* kernel);

bool HOGGPU(int size, int* imgData, int* kernel);

#endif