#define _USE_MATH_DEFINES // this only works in MSVC if it's all the way at the top... >:(
#include <cmath>

#include "xy.h"

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std;

const int equilibration_steps = 100;

struct Cell
{
	float angle;
	uint32_t flipped;
	uint32_t left_considered;
	uint32_t down_considered;
};

struct Model
{
	Cell* cells;
	uint32_t* to_consider; // index into cells
	uint32_t* to_flip; // index into cells // TODO: remove
	uint32_t n_to_consider;
	uint32_t n_to_flip; // TODO: remove
	int n_flipped;
	int L;
	float T;
	float cluster_probability; // TODO: remove

public:
	Model(int L, float T)
	{
		this->L = L;
		set_T(T);
		n_flipped = 0;
		cells = (Cell*)malloc(L * L * sizeof(Cell));
		memset(cells, 0, L * L * sizeof(Cell));
		to_consider = (uint32_t*)malloc(L * L * sizeof(uint32_t));
		to_flip = (uint32_t*)malloc(L * L * sizeof(uint32_t));

		// randomly initialize angles
		for (int i = 0; i < L * L; i++)
			cells[i].angle = (rand() / static_cast<float>(RAND_MAX)) * 2 * M_PI; // random in [0, 2pi]
	}

	~Model()
	{
		free(cells);
		free(to_consider);
		free(to_flip);
	}

	void set_T(float T)
	{
		this->T = T;
		cluster_probability = 1. - exp(-1. / T); // TODO: this is useless for the XY model
		
		// strictly speaking, the probability of adding to the cluster should be equal to 1 - exp(-J/(k_B*T)), for a hamiltonian H = -J * # number of aligned neigbor-pairs
		// k_B = 1 is chosen, and for emulating the Ising model as a Potts model, J = 2, but everyone else uses J = 1 for Potts models, so that value is chosen
	}

	bool check_probability(uint32_t i, uint32_t j, float r)
	{
		// implements equation (5) in the paper by Wolff, i.e. PhysRevLett.62.361, assumes cell[i] has been flipped already
		// probability depends on temperature and the cosine of the angle between the spin directions and the reflection plane
		float cos_i = cos(cells[i].angle - r);
		float cos_j = cos(cells[j].angle - r);
		float cluster_probability = 1. - exp(min(0., 2. * cos_i * cos_j / T));
		float random = rand() / static_cast<float>(RAND_MAX); // random number in [0, 1]
		return random < cluster_probability;
	}

	// flip site if not already flipped in this step
	void flip(uint32_t i, float r, uint32_t step_id)
	{
		if (cells[i].flipped == step_id)
			return;
		cells[i].angle = fmod(2 * M_PI + r + M_PI - (cells[i].angle - r), 2. * M_PI); // limit to range [0, 2pi)
		n_flipped++;
	}

	// perform one step of simulation
	void step(uint32_t step_id)
	{
		// crash if step_id above limit
		if (step_id >= ((unsigned)1 << 31) - 1)
		{
			cout << "EXCEEDED MAXIMUM NUMBER OF STEPS" << endl;
			exit(1);
		}

		// start at random index with random reflection plane, always flip the first one
		float r = (rand() / static_cast<float>(RAND_MAX)) * 2 * M_PI; // random in [0, 2pi]
		to_consider[0] = rand() % (L * L);
		flip(to_consider[0], r, step_id);
		n_to_consider = 1;

		// loop while there are cells to consider
		while (n_to_consider > 0)
		{
			// pop a site from the stack
			uint32_t site_idx = to_consider[n_to_consider - 1];
			int x = site_idx % L;
			int y = site_idx / L;
			n_to_consider--;

			// compute neighbor indices
			uint32_t left_idx = (x < L - 1 ? x + 1 : 0) + y * L;
			uint32_t right_idx = (x > 0 ? x - 1 : L - 1) + y * L;
			uint32_t down_idx = x + (y < L - 1 ? y + 1 : 0) * L;
			uint32_t up_idx = x + (y > 0 ? y - 1 : L - 1) * L;
			
			// consider left bond
			if (cells[site_idx].left_considered != step_id)
			{
				if (check_probability(site_idx, left_idx, r))
				{
					to_consider[n_to_consider++] = left_idx;
					flip(left_idx, r, step_id);
				}
				cells[site_idx].left_considered = step_id;
			}

			// consider right bond
			if (cells[right_idx].left_considered != step_id)
			{
				if (check_probability(site_idx, right_idx, r))
				{
					to_consider[n_to_consider++] = right_idx;
					flip(right_idx, r, step_id);
				}
				cells[right_idx].left_considered = step_id;
			}

			// consider downward bond
			if (cells[site_idx].down_considered != step_id)
			{
				if (check_probability(site_idx, down_idx, r))
				{
					to_consider[n_to_consider++] = down_idx;
					flip(down_idx, r, step_id);
				}
				cells[site_idx].down_considered = step_id;
			}

			// consider upward bond
			if (cells[up_idx].down_considered != step_id)
			{
				if (check_probability(site_idx, up_idx, r))
				{
					to_consider[n_to_consider++] = up_idx;
					flip(up_idx, r, step_id);
				}
				cells[up_idx].down_considered = step_id;
			}
		}
	}

	vector<xy_t> get_states()
	{
		vector<xy_t> result = vector<xy_t>();
		result.reserve(L * L);
		for (int i = 0; i < L * L; i++)
			result.push_back((uint8_t) (cells[i].angle / M_PI * 128.)); // compress angles into the 8-bit [0, 255] range
		return result;
	}
};

vector<xy_t> generate_samples(int L, float T, int n_samples)
{
	Model* model = new Model(L, T);
	
	vector<xy_t> batch;
	batch.reserve(L * L * n_samples);
	uint32_t step_id = 1;
	for (int s = 0; s < equilibration_steps + n_samples; s++)
	{
		// run the model until L * L states are flipped
		model->n_flipped = 0;
		for (; model->n_flipped < L * L; step_id++)
			model->step(step_id);

		if (s >= equilibration_steps)
		{
			vector<xy_t> sample = model->get_states();
			batch.insert(batch.end(), sample.begin(), sample.end());
		}
	}
	
	delete model;
	return batch;
}