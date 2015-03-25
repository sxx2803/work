/************************************************************************/
// Author: Sicheng Xu
// Date: August 9, 2014
// Course: 0306-724 - High Performance Architectures
//
// File: FilterCPU.cpp
// The purpose of this program is to do median filtering on CPU
/************************************************************************/

#include "MedianFilter.h"

// Length of the square window around the pixel

/**
 * Sorts an array using Bubble Sort.
 * 
 * n -> length of array
 * toSort -> the array to sort
 */
void BubbleSort(char* toSort, int n){
	bool swapped;
	do{
		swapped = false;
		for(int i = 1; i < n; i++){
			// If out of order, swap
			if((unsigned char) toSort[i-1] > (unsigned char) toSort[i]){
				char temp = toSort[i-1];
				toSort[i-1] = toSort[i];
				toSort[i] = temp;
				swapped = true;
			}
		}
	} while(swapped);
}

// median filtering on the CPU
void MedianFilterCPU( Bitmap* image, Bitmap* outputImage ){
	char *window = new char[WINDOW_SIZE * WINDOW_SIZE];
	memcpy(outputImage->image, image->image, image->Width()*image->Height()*sizeof(char));
	// Ignore edge cases (keep original image pixels)
	for(int i = WINDOW_SIZE/2; i < (image->Height() - WINDOW_SIZE/2); i++){
		for(int j = WINDOW_SIZE/2; j < (image->Width() - WINDOW_SIZE/2); j++){
			int windowIdx = 0;
			// Put the pixels into the window array
			for(int k = i - WINDOW_SIZE/2; k <= i + WINDOW_SIZE/2; k++){
				for(int l = j - WINDOW_SIZE/2; l <= j + WINDOW_SIZE/2; l++){
					unsigned char pixValue = image->GetPixel(l, k);
					window[windowIdx++] = pixValue;
				}
			}
			// Sort the array
			BubbleSort(window, WINDOW_SIZE*WINDOW_SIZE);
			// Assign median value to output
			outputImage->SetPixel(j, i, window[(WINDOW_SIZE*WINDOW_SIZE)/2]);
		}
	}
	delete[] window;
}