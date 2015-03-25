/************************************************************************/
// The purpose of this program is to perform K-Means Clustering on the GPU.
//
// Author: Sicheng Xu
// Date: April 26, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: main.cpp
/************************************************************************/

#include "KMeans.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include <cmath>

#define TILE_SIZE 16
#define K 3

__constant__ Vector2 clustersIn[K];

bool checkForError( cudaError_t error)
{
	// Check if the status is an error message
	if(error != cudaSuccess)
	{
		// Print out the error message
		printf("CUDA Error: %s\n", cudaGetErrorString(error));
		return true;
	}
	// Otherwise, no error occured
	else
	{
		return false;
	}
}

__global__ void KMeansKernel(Datapoint* data, long n, int k){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	// Outside of domain
	if(threadId >= n){
		return;
	}
	int nearestCluster = data[threadId].cluster;
	// Check all clusters if they're nearer
	data[threadId].altered = false;
	for(int i = 0; i < k; i++){
		if(data[threadId].p.distSq(clustersIn[i]) < data[threadId].p.distSq(clustersIn[nearestCluster])){
				nearestCluster = i;
				data[threadId].altered = true;
				
		}
	}
	data[threadId].cluster = nearestCluster;
}

bool KMeansGPU(Datapoint* data, long n, Vector2* clusters, int k){
	cudaError_t status;

	// Arrays for the data
	Datapoint* dataIn;
	status = cudaMalloc((void**)&dataIn, n*sizeof(Datapoint));
	if(checkForError(status)){
		printf("Error allocating for data points\n");
		return false;
	}
	// Copy data in
	status = cudaMemcpy(dataIn, data, n*sizeof(Datapoint), cudaMemcpyHostToDevice);
	if(checkForError(status)){
		printf("Error copying data points\n");
		return false;
	}
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil(sqrtf((float)n)/(TILE_SIZE)), (int)ceil(sqrtf((float)n)/(TILE_SIZE)));

	bool changedClusters = true;
	while(changedClusters){
		status = cudaMemcpyToSymbol(clustersIn, clusters, k*sizeof(Vector2), 0, cudaMemcpyHostToDevice);
		if(checkForError(status)){
			printf("Error updating constant memory \n");
			return false;
		}
		changedClusters = false;
		// Assign the data points to clusters
		KMeansKernel<<<dimGrid, dimBlock>>>(dataIn, n, k);
		cudaDeviceSynchronize();
		status = cudaGetLastError();
		if(checkForError(status)){
			printf("Error in k-means kernel\n");
			return false;
		}
		status = cudaMemcpy(data, dataIn, n*sizeof(Datapoint), cudaMemcpyDeviceToHost);
		if(checkForError(status)){
			printf("Error updating host data for data points\n");
			return false;
		}
	
		// Update the centers
		for(int i = 0; i < k; i++){
			float clusterXSum = 0.0f;
			float clusterYSum = 0.0f;
			int numPoints = 0;
			for(int j = 0; j < n; j++){
				if(data[j].cluster == i){
					clusterXSum += data[j].p.x;
					clusterYSum += data[j].p.y;
					numPoints++;
				}
				if(data[j].altered){
					changedClusters = true;
				}
			}
			float xMean = clusterXSum / (float) numPoints;
			float yMean = clusterYSum / (float) numPoints;
			clusters[i].x = xMean;
			clusters[i].y = yMean;
		}
	}

	status = cudaFree(dataIn);
	if(checkForError(status)){
		printf("Error freeing cuda global memory for data points\n");
		return false;
	}
	return true;

}