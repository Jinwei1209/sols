#pragma once

#include <cmath>
#include <numeric>
#include <algorithm>
#include <queue>
#include <list>
#include <tuple>
#include <vector>
#include <bits/stdc++.h>
#include "utils.hpp"

#ifdef ParallelProvider_IntelTBB
#include <tbb/parallel_for.h>
#else
#include <omp.h>
#endif


auto SNN_slow(int k, int n, int d, int nc, float* data, int dataset) {
	const int unassigned = -1;
	const float infinity = std::numeric_limits<float>::infinity();
	std::cout << "Processing Processing dataset " << dataset << "..." << std::endl;

	// Compute neighbor with Euclidean distance
	// --------------------------------------------------------------------------------

	const auto indexNeighbor = new int[n * k];
	const auto distance_cur = new float[n];
	const auto indexDistanceAsc_cur = new float[n];
	// #pragma omp parallel for (does not work in O(n) memory allocation)
	for (int i = 0; i < n; i++) {
		std::fill(distance_cur, distance_cur + n, 0);
		for (int j = 0; j < n; j++) {
			distance_cur[j] = pow(data[i * d + 0] - data[j * d + 0], 2) + pow(data[i * d + 1] - data[j * d + 1], 2);
		}
		std::iota(indexDistanceAsc_cur, indexDistanceAsc_cur + n, 0);
		std::sort(indexDistanceAsc_cur, indexDistanceAsc_cur + n, [&](int a, int b) { return distance_cur[a] < distance_cur[b]; });
		std::copy(indexDistanceAsc_cur, indexDistanceAsc_cur + k, indexNeighbor + i * k);
		std::sort(indexNeighbor + i * k, indexNeighbor + (i + 1) * k); // For set_intersection()
	}

	delete[] distance_cur;
	delete[] indexDistanceAsc_cur;

	// Compute shared neighbor
	// --------------------------------------------------------------------------------

	const auto numSharedNeighbor = new int[n * k];  // each element in numSharedNeighbor corresponds to indexNeighbor in order
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			int b = indexNeighbor[i * k + j];
			numSharedNeighbor[i * k + j] = intersection_size(indexNeighbor + i * k, indexNeighbor + (i + 1) * k, 
															 indexNeighbor + b * k, indexNeighbor + (b + 1) * k);
		}
	}

	// Compute centroid
	// --------------------------------------------------------------------------------

	const auto indexAssignment = new int[n];
	const auto indexCentroid = new int[nc];
	std::fill(indexAssignment, indexAssignment + n, unassigned);

	// finding the closest data points to the prior centroids (all datasets)
	std::vector<std::vector<float>> centroids {
		{0.17, 0.45}, 
		{0.08, 0.4}, 
		{0.32, 0.3}, 
		{0.8, 0.8}
	};

	const auto distance2centroid1 = new float[n];
	const auto distance2centroid2 = new float[n];
	const auto distance2centroid3 = new float[n];
	const auto distance2centroid4 = new float[n];

	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		distance2centroid1[i] = sqrt(pow(data[i * d + 0] - centroids[0][0], 2) + pow(data[i * d + 1] - centroids[0][1], 2));
		distance2centroid2[i] = sqrt(pow(data[i * d + 0] - centroids[1][0], 2) + pow(data[i * d + 1] - centroids[1][1], 2));
		distance2centroid3[i] = sqrt(pow(data[i * d + 0] - centroids[2][0], 2) + pow(data[i * d + 1] - centroids[2][1], 2));
		distance2centroid4[i] = sqrt(pow(data[i * d + 0] - centroids[3][0], 2) + pow(data[i * d + 1] - centroids[3][1], 2));
	}

	float min1 = *(std::min_element(distance2centroid1, distance2centroid1+n));
	int index1 = std::find(distance2centroid1, distance2centroid1+n, min1)-distance2centroid1;
	indexCentroid[0] = index1;

	float min2 = *(std::min_element(distance2centroid2, distance2centroid2+n));
	int index2 = std::find(distance2centroid2, distance2centroid2+n, min2)-distance2centroid2;
	indexCentroid[1] = index2;

	float min3 = *(std::min_element(distance2centroid3, distance2centroid3+n));
	int index3 = std::find(distance2centroid3, distance2centroid3+n, min3)-distance2centroid3;
	indexCentroid[2] = index3;

	float min4 = *(std::min_element(distance2centroid4, distance2centroid4+n));
	int index4 = std::find(distance2centroid4, distance2centroid4+n, min4)-distance2centroid4;
	indexCentroid[3] = index4;

	// std::sort(indexCentroid, indexCentroid + nc);
	for (int i = 0; i < nc; i++)
		indexAssignment[indexCentroid[i]] = i;

	delete[] distance2centroid1;
	delete[] distance2centroid2;
	delete[] distance2centroid3;
	delete[] distance2centroid4;

	// Assign non centroid step 1
	// --------------------------------------------------------------------------------

	float factor = 1.27;  // magic number: 1.27, only 2/3 cases with 2 clusters as one
	std::queue<int> queue;
	const auto numAssigned = new int[nc];
	std::fill(numAssigned, numAssigned + nc, 1);

	for (int i = 0; i < nc; i++)
		queue.push(indexCentroid[i]);
	while (!queue.empty()) {
		int a = queue.front();
		queue.pop();
		for (int i = 0; i < k; i++) {
			int b = indexNeighbor[a * k + i];  // in the KNN list of 'a' count 'b' (indexNeighbor(a*k+b)) s.t. numSharedNeighbor(a, b) (sim(a, b)) >= k/factor 
			if (indexAssignment[b] == unassigned && numSharedNeighbor[a * k + i] * factor >= k) { 
				indexAssignment[b] = indexAssignment[a];
				numAssigned[indexAssignment[b]]++;
				queue.push(b);
			}
		}
	}

	std::cout << "Number of core points before augmentation: ";
	for (int i = 0; i < nc; i++) {
		std::cout << numAssigned[i] << " ";
	}
	std::cout << std::endl;

	int idx = 0;
	for (int i = 0; i < nc; i++) {
		if (numAssigned[i] < k) {
			for (int j = 0; j < k; j++) {
				int a = indexCentroid[i];
				indexAssignment[indexNeighbor[a * k + j]] = indexAssignment[a];
				numAssigned[indexAssignment[indexNeighbor[a * k + j]]]++;
			}
		}
	}

	std::cout << "Number of core points after augmentation:  ";
	for (int i = 0; i < nc; i++) {
		std::cout << numAssigned[i] << " ";
	}
	std::cout << std::endl;

	delete[] numSharedNeighbor;

	// Assign non centroid step 2
	// --------------------------------------------------------------------------------

	std::list<int> indexUnassigned;
	for (int i = 0; i < n; i++)
		if (indexAssignment[i] == unassigned)
			indexUnassigned.push_back(i);

	int numUnassigned = indexUnassigned.size();
	const auto numNeighborAssignment = new int[numUnassigned * nc];
	while (numUnassigned) {
		std::fill(numNeighborAssignment, numNeighborAssignment + numUnassigned * nc, 0);
		int i = 0;
		for (const auto& a : indexUnassigned) {
			for (int j = 0; j < k; j++) {
				int b = indexNeighbor[a * k + j];
				if (indexAssignment[b] != unassigned)
					++numNeighborAssignment[i * nc + indexAssignment[b]];
			}
			i++;
		}
		if (int most = *std::max_element(numNeighborAssignment, numNeighborAssignment + numUnassigned * nc)) {
			auto it = indexUnassigned.begin();
			for (int j = 0; j < numUnassigned; j++) {
				const auto first = numNeighborAssignment + j * nc;
				const auto last = numNeighborAssignment + (j + 1) * nc;
				const auto current = std::find(first, last, most); // In MATLAB, if multiple hits, the last will be used
				if (current == last) ++it;
				else {
					indexAssignment[*it] = current - first;  // calculate std::find position (counting from zero)
					it = indexUnassigned.erase(it);
				}
			}
			numUnassigned = indexUnassigned.size();
		} 
	}
	delete[] indexNeighbor;
	delete[] numNeighborAssignment;

	// Return
	// --------------------------------------------------------------------------------

	return std::tuple{indexCentroid, indexAssignment};
}