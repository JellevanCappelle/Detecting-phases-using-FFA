#pragma once

#include <vector>
#include <cstdint>

using namespace std;

typedef int8_t ising_t;
vector<ising_t> generate_sample(int L, float T, int n_samples);