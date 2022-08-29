#include <chrono>

#include "SNNwithCentroids_slow.hpp"
#include "SNNwithCentroids_fast.hpp"
#include "SNNrefine.hpp"
#include "utils.hpp"

using namespace std::chrono;

int main(int argc, char* argv[]) {

	int dataset_size[6] = {34993, 19495, 25192, 26494, 23194, 17195};

	for (int dataset_idx = 1; dataset_idx <= 6; dataset_idx++) {
		// Parameter
		// --------------------------------------------------------------------------------
		char dataset[10];
		sprintf(dataset, "%d", dataset_idx);
		const auto pathData1 = SOLUTION_DIR"data/Dataset_";
		const auto suffix = ".tsv";
		char pathData[100];   // array to hold the result.
		strcpy(pathData, pathData1);
		strcat(pathData, dataset);
		strcat(pathData, suffix);

		const int k = 35;
		const int n = dataset_size[dataset_idx-1];
		const int d = 2; // Dimension
		const int nc = 4; // Number of centroids
		char command[1024];

		// Create dataset
		// --------------------------------------------------------------------------------
		// sprintf(command, "python %s %d", SOLUTION_DIR"convert_to_csv.py", dataset[0] - '0');
		// system(command);

		// Read dataset
		// --------------------------------------------------------------------------------
		int label[n];
		float data[n * d];
		const auto fileData = fopen(pathData, "r");
		for (int i = 0; i < n; i++)
			fscanf(fileData, "%f %f %d\n", &data[i * d], &data[i * d + 1], &label[i]);
		fclose(fileData);

		// Run Clustering Algorithm
		// --------------------------------------------------------------------------------
		auto time = high_resolution_clock::now();
		if (argv[1][0] == '1') {
			printf("\nRunning fast algorithm with O(n^2) memory consumption. \n");
			const auto [centroid, assignment] = SNN_fast(k, n, d, nc, data, dataset_idx);
			export_results(centroid, assignment, nc, n);
			delete[] centroid;
			delete[] assignment;
		}
		if (argv[1][0] == '2') {
			printf("\nRunning slow algorithm with O(n) memory consumption. \n");
			const auto [centroid, assignment] = SNN_slow(k, n, d, nc, data, dataset_idx);
			export_results(centroid, assignment, nc, n);
			delete[] centroid;
			delete[] assignment;
		}
		printf("Time Cost = %lldms\n", duration_cast<milliseconds>(high_resolution_clock::now() - time).count());

		// For Dataset 3 and 6, correct with GMM model for cluster #2
		// --------------------------------------------------------------------------------
		// if (dataset_idx == 3 || dataset_idx == 6) {
		sprintf(command, "python %s %d", SOLUTION_DIR"GMM_for_refinement.py", dataset[0] - '0');
		system(command);
		// }

		// run SNNrefine after GMM on cluster #1
		// --------------------------------------------------------------------------------
		char* pathLabel = SOLUTION_DIR"temp/Assignment_wo_outliers.txt";
		const auto fileLabel = fopen(pathLabel, "r");
		for (int i = 0; i < n; i++)
			fscanf(fileLabel, "%d\n", &label[i]);
		fclose(fileLabel);
		printf("\nRunning SNNrefine after GMM. \n");
		const auto assignment = SNN_refine(k, n, d, nc, data, label, dataset_idx);
		export_results_refine(assignment, n);
		delete[] assignment;

		// Save results after refinement
		// --------------------------------------------------------------------------------
		sprintf(command, "python %s %d %d", SOLUTION_DIR"plot_Q2_results.py", dataset[0] - '0', 1);
		system(command);

		// Save results
		// --------------------------------------------------------------------------------
		sprintf(command, "python %s %d %d", SOLUTION_DIR"plot_Q2_results.py", dataset[0] - '0', 0);
		system(command);

	}
}
