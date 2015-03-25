/************************************************************************/
// The purpose of this program is to perform K-Means Clustering on the CPU.
//
// Author: Sicheng Xu
// Date: April 26, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: main.cpp
/************************************************************************/

#include "KMeans.h"


void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k){
	
	bool changedClusters = true;
	while(changedClusters){
		changedClusters = false;
		// Assign data points to clusters
		for(int i = 0; i < n; i++){
			int nearestCluster = data[i].cluster;
			for(int j = 0; j < k; j++){
				if(data[i].p.distSq(clusters[j]) < data[i].p.distSq(clusters[nearestCluster])){
					nearestCluster = j;
					changedClusters = true;

				}
			}
			data[i].cluster = nearestCluster;
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
			}
			float xMean = clusterXSum / (float) numPoints;
			float yMean = clusterYSum / (float) numPoints;
			clusters[i].x = xMean;
			clusters[i].y = yMean;
		}
	}
}