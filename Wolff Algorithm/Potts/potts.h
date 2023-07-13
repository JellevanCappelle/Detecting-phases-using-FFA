#pragma once

#include <vector>
#include <cstdint>

using namespace std;

typedef int8_t potts_t;
vector<potts_t> generate_samples(int q, int L, float T, int n_samples);

/*
SOURCES:
Introduction to Cluster Monte Carlo Algorithms, Erik Luijten
Collective Monte Carlo Updating for Spin Systems, Ulli Wolff
*/