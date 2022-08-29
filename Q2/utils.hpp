#pragma once

#include <cmath>
#include <numeric>
#include <algorithm>
#include <queue>
#include <list>
#include <tuple>
#include <vector>
#include <bits/stdc++.h>

struct Counter
{
  struct value_type { template<typename T> value_type(const T&) { } };
  void push_back(const value_type&) { ++count; }
  size_t count = 0;
};

template<typename T1, typename T2>
size_t intersection_size(const T1& s1_begin, const T1& s1_end, const T2& s2_begin, const T2& s2_end)
{
  Counter c;
  std::set_intersection(s1_begin, s1_end, s2_begin, s2_end, std::back_inserter(c));
  return c.count;
}

void export_results(int* centroid, int* assignment, int nc, int n) {
  // Export centroid
  const auto fileCentroid = fopen(SOLUTION_DIR"temp/Centroid.txt", "w");
  for (int i = 0; i < nc; i++)
    fprintf(fileCentroid, "%d\n", centroid[i]);
  fclose(fileCentroid);

  // Export assignment
  const auto pathAssignment = SOLUTION_DIR"temp/Assignment.txt";
  const auto fileAssignment = fopen(pathAssignment, "w");
  for (int i = 0; i < n; i++)
    fprintf(fileAssignment, "%d\n", assignment[i]);
  fclose(fileAssignment);
}

void export_results_refine(int* assignment, int n) {
  // Export assignment
  const auto pathAssignment = SOLUTION_DIR"temp/Assignment_refine.txt";
  const auto fileAssignment = fopen(pathAssignment, "w");
  for (int i = 0; i < n; i++)
    fprintf(fileAssignment, "%d\n", assignment[i]);
  fclose(fileAssignment);
}