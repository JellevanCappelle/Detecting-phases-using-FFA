#pragma once

#include <vector>
#include <cstdint>

using namespace std;

typedef uint8_t xy_t;
vector<xy_t> generate_samples(int L, float T, int n_samples);

/*
SOURCES:
Introduction to Cluster Monte Carlo Algorithms, Erik Luijten
Collective Monte Carlo Updating for Spin Systems, Ulli Wolff
*/