#include "MedianFilter.h"
#include <iostream>
#include <ctime>

#define ITERS 1000

int main(){

	clock_t start, end;
	float timeElapsed;
	bool success = false;
	int diff;

	// Do benchmark on Lenna
	Bitmap lennaIn;
	lennaIn.Load("Lenna.bmp");
	Bitmap lennaOutGpuSM(lennaIn.Width(), lennaIn.Height());
	Bitmap lennaOutGpuNSM(lennaIn.Width(), lennaIn.Height());
	Bitmap lennaOutCpu(lennaIn.Width(), lennaIn.Height());

	printf("Operating on a %i x %i image\n", lennaIn.Width(), lennaIn.Height());

	// Start the CPU benchmark
	start = clock();
	for(int i = 0; i < ITERS; i++){
		MedianFilterCPU(&lennaIn, &lennaOutCpu);
	}
	end = clock();
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Average Time per CPU Iteration is: %5.3f ms\n\n", timeElapsed);

	// Start the Non Memory Shared GPU Benchmark with a warmup pass
	success = MedianFilterGPU(&lennaIn, &lennaOutGpuNSM, false);
	if(!success){
		printf("Error running warmup pass for non shared GPU version, exiting\n");
		return 1;
	}
	start = clock();
	for(int i = 0; i < ITERS; i++){
		success = MedianFilterGPU(&lennaIn, &lennaOutGpuNSM, false);
	}
	end = clock();
	if(!success){
		printf("Error running GPU median filter w/o shared memory\n");
		return 1;
	}
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	diff = CompareBitmaps(&lennaOutCpu, &lennaOutGpuNSM);
	printf("Average Time per GPU Iteration with global memory is: %5.3f ms\n", timeElapsed);
	printf("%i differing pixels compared to the CPU algorithm\n\n", diff); 

	// Start the Memory Shared GPU Benchmark with a warmup pass
	success = MedianFilterGPU(&lennaIn, &lennaOutGpuSM, true);
	if(!success){
		printf("Error running warmup pass for shared GPU version, exiting\n");
		return 1;
	}
	start = clock();
	for(int i = 0; i < ITERS; i++){
		success = MedianFilterGPU(&lennaIn, &lennaOutGpuSM, true);
	}
	end = clock();
	if(!success){
		printf("Error running GPU median filter w/ shared memory\n");
		return 1;
	}
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	diff = CompareBitmaps(&lennaOutCpu, &lennaOutGpuSM);
	printf("Average Time per GPU Iteration with shared memory is: %5.3f ms\n", timeElapsed);
	printf("%i differing pixels compared to the CPU algorithm\n---\n", diff); 

	//*************************************//
	//*************************************//
	
	//*************************************//
	//******* NEW BENCHMARK SECTION *******//
	//*************************************//

	// Do benchmark on Milky Way
	Bitmap milkyIn;
	milkyIn.Load("milkyway.bmp");
	Bitmap milkyOutGpuSM(milkyIn.Width(), milkyIn.Height());
	Bitmap milkyOutGpuNSM(milkyIn.Width(), milkyIn.Height());
	Bitmap milkyOutCpu(milkyIn.Width(), milkyIn.Height());

	printf("Operating on a %i x %i image\n", milkyIn.Width(), milkyIn.Height());

	// Start the CPU benchmark
	start = clock();
	for(int i = 0; i < ITERS; i++){
		MedianFilterCPU(&milkyIn, &milkyOutCpu);
	}
	end = clock();
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Average Time per CPU Iteration is: %5.3f ms\n\n", timeElapsed);

	// Start the Non Memory Shared GPU Benchmark with a warmup pass
	success = MedianFilterGPU(&milkyIn, &milkyOutGpuNSM, false);
	if(!success){
		printf("Error running warmup pass for non shared GPU version, exiting\n");
		return 1;
	}
	start = clock();
	for(int i = 0; i < ITERS; i++){
		success = MedianFilterGPU(&milkyIn, &milkyOutGpuNSM, false);
	}
	end = clock();
	if(!success){
		printf("Error running GPU median filter w/o shared memory\n");
		return 1;
	}
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	diff = CompareBitmaps(&milkyOutCpu, &milkyOutGpuNSM);
	printf("Average Time per GPU Iteration with global memory is: %5.3f ms\n", timeElapsed);
	printf("%i differing pixels compared to the CPU algorithm\n\n", diff); 

	// Start the Memory Shared GPU Benchmark with a warmup pass
	success = MedianFilterGPU(&milkyIn, &milkyOutGpuSM, true);
	if(!success){
		printf("Error running warmup pass for shared GPU version, exiting\n");
		return 1;
	}
	start = clock();
	for(int i = 0; i < ITERS; i++){
		success = MedianFilterGPU(&milkyIn, &milkyOutGpuSM, true);
	}
	end = clock();
	if(!success){
		printf("Error running GPU median filter w/ shared memory\n");
		return 1;
	}
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	diff = CompareBitmaps(&milkyOutCpu, &milkyOutGpuSM);
	printf("Average Time per GPU Iteration with shared memory is: %5.3f ms\n", timeElapsed);
	printf("%i differing pixels compared to the CPU algorithm\n---\n", diff); 

	//*************************************//
	//*************************************//
	
	//*************************************//
	//******* NEW BENCHMARK SECTION *******//
	//*************************************//

	// Do benchmark on RIT
	Bitmap ritIn;
	ritIn.Load("RIT.bmp");
	Bitmap ritOutGpuSM(ritIn.Width(), ritIn.Height());
	Bitmap ritOutGpuNSM(ritIn.Width(), ritIn.Height());
	Bitmap ritOutCpu(ritIn.Width(), ritIn.Height());

	printf("Operating on a %i x %i image\n", ritIn.Width(), ritIn.Height());

	// Start the CPU benchmark
	start = clock();
	for(int i = 0; i < ITERS; i++){
		MedianFilterCPU(&ritIn, &ritOutCpu);
	}
	end = clock();
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Average Time per CPU Iteration is: %5.3f ms\n\n", timeElapsed);

	// Start the Non Memory Shared GPU Benchmark with a warmup pass
	success = MedianFilterGPU(&ritIn, &ritOutGpuNSM, false);
	if(!success){
		printf("Error running warmup pass for non shared GPU version, exiting\n");
		return 1;
	}
	start = clock();
	for(int i = 0; i < ITERS; i++){
		success = MedianFilterGPU(&ritIn, &ritOutGpuNSM, false);
	}
	end = clock();
	if(!success){
		printf("Error running GPU median filter w/o shared memory\n");
		return 1;
	}
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	diff = CompareBitmaps(&ritOutCpu, &ritOutGpuNSM);
	printf("Average Time per GPU Iteration with global memory is: %5.3f ms\n", timeElapsed);
	printf("%i differing pixels compared to the CPU algorithm\n\n", diff); 

	// Start the Memory Shared GPU Benchmark with a warmup pass
	success = MedianFilterGPU(&ritIn, &ritOutGpuSM, true);
	if(!success){
		printf("Error running warmup pass for shared GPU version, exiting\n");
		return 1;
	}
	start = clock();
	for(int i = 0; i < ITERS; i++){
		success = MedianFilterGPU(&ritIn, &ritOutGpuSM, true);
	}
	end = clock();
	if(!success){
		printf("Error running GPU median filter w/ shared memory\n");
		return 1;
	}
	timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	diff = CompareBitmaps(&ritOutCpu, &ritOutGpuSM);
	printf("Average Time per GPU Iteration with shared memory is: %5.3f ms\n", timeElapsed);
	printf("%i differing pixels compared to the CPU algorithm\n---\n", diff); 

	
	lennaOutCpu.Save("lennaOutCpu.bmp");
	lennaOutGpuNSM.Save("lennaOutGpuNoShared.bmp");
	lennaOutGpuSM.Save("lennaOutGpuShared.bmp");

	milkyOutCpu.Save("milkyoutCpu.bmp");
	milkyOutGpuNSM.Save("milkyOutGpuNoShared.bmp");
	milkyOutGpuSM.Save("milkyOutGpuShared.bmp");

	ritOutCpu.Save("ritOutCpu.bmp");
	ritOutGpuNSM.Save("ritOutGpuNoShared.bmp");
	ritOutGpuSM.Save("ritOutGpuShared.bmp");

	getchar();


	return 0;
}