#include "Bitmap.h"
#include "MedianFilter.h"

/**
 * Compares two bitmaps and returns the number of pixels that differ
 */
int CompareBitmaps( Bitmap* inputA, Bitmap* inputB ){
	int errors = 0;
	// Loop over all pixels
	for(int i = 0; i < inputA->Height(); i++){
		for(int j = 0; j < inputA->Width(); j++){
			// If input image pixel is not same intensity as output image pixel, inc err
			if(inputA->GetPixel(j, i) != inputB->GetPixel(j, i)){
				++errors;
			}
		}
	}

	return errors;
}