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


auto SNN_refine(int k, int n, int d, int nc, float* data, int* labels, int dataset) {
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
		if (labels[i] == -1) {
			for (int j = 0; j < n; j++) {
				distance_cur[j] = pow(data[i * d + 0] - data[j * d + 0], 2) + pow(data[i * d + 1] - data[j * d + 1], 2);
			}
			std::iota(indexDistanceAsc_cur, indexDistanceAsc_cur + n, 0);
			std::sort(indexDistanceAsc_cur, indexDistanceAsc_cur + n, [&](int a, int b) { return distance_cur[a] < distance_cur[b]; });
			std::copy(indexDistanceAsc_cur, indexDistanceAsc_cur + k, indexNeighbor + i * k);
			std::sort(indexNeighbor + i * k, indexNeighbor + (i + 1) * k); // For set_intersection()
		}
	}

	delete[] distance_cur;
	delete[] indexDistanceAsc_cur;

	// Compute shared neighbor
	// --------------------------------------------------------------------------------

	const auto numSharedNeighbor = new int[n * k];  // each element in numSharedNeighbor corresponds to indexNeighbor in order
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		if (labels[i] == -1) {
			for (int j = 0; j < k; j++) {
				int b = indexNeighbor[i * k + j];
				numSharedNeighbor[i * k + j] = intersection_size(indexNeighbor + i * k, indexNeighbor + (i + 1) * k, 
																indexNeighbor + b * k, indexNeighbor + (b + 1) * k);
			}
		}
	}

	// Assign labels from SNN clustering
	// --------------------------------------------------------------------------------
	const auto indexAssignment = new int[n];
	for (int i = 0; i < n; i++) {
		indexAssignment[i] = labels[i];
	}
	
	// Assign outliers
	// --------------------------------------------------------------------------------
	float factor = 2.0;  // to put less weights on cluster #1
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
				if (indexAssignment[b] != unassigned) {
					if (indexAssignment[b] != 1) numNeighborAssignment[i * nc + indexAssignment[b]] += (int)(factor*10);
					else numNeighborAssignment[i * nc + indexAssignment[b]] += 10;
				}
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

	return indexAssignment;
}