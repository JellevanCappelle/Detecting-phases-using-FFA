#include "potts.h"

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace std;

const int equilibration_steps = 100;

struct Cell
{
	uint32_t state;
	uint32_t flipped;
	uint32_t left_considered;
	uint32_t down_considered;
};

struct Model
{
	Cell* cells;
	uint32_t* to_consider; // index into cells
	uint32_t* to_flip; // index into cells
	uint32_t n_to_consider;
	uint32_t n_to_flip;
	int n_flipped;
	int q;
	int L;
	float T;
	float cluster_probability;

public:
	Model(int q, int L, float T)
	{
		this->q = q;
		this->L = L;
		set_T(T);
		n_flipped = 0;
		cells = (Cell*)malloc(L * L * sizeof(Cell));
		memset(cells, 0, L * L * sizeof(Cell));
		to_consider = (uint32_t*)malloc(L * L * sizeof(uint32_t));
		to_flip = (uint32_t*)malloc(L * L * sizeof(uint32_t));

		// randomly initialize states
		for (int i = 0; i < L * L; i++)
			cells[i].state = rand() % q;
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
		cluster_probability = 1. - exp(-1. / T);
		
		// strictly speaking, the probability of adding to the cluster should be equal to 1 - exp(-J/(k_B*T)), for a hamiltonian H = -J * # number of aligned neigbor-pairs
		// k_B = 1 is chosen, and for emulating the Ising model as a Potts model, J = 2, but everyone else uses J = 1 for Potts models, so that value is chosen
	}

	bool check_probability(uint32_t i, uint32_t j)
	{
		if (cells[i].state != cells[j].state)
			return false; // zero probability for non-aligned states

		float random = rand() / static_cast<float>(RAND_MAX); // random number in [0, 1]
		return random < cluster_probability;
	}

	// flip site if not already flipped in this step
	void flip(uint32_t i, uint32_t step_id)
	{
		if (cells[i].flipped == step_id)
			return;
		to_flip[n_to_flip++] = i;
		cells[i].flipped = step_id;
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

		// start at random index, always flip the first one
		n_to_flip = 0;
		flip(to_consider[0] = rand() % (L * L), step_id);
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
				if (check_probability(site_idx, left_idx))
				{
					to_consider[n_to_consider++] = left_idx;
					flip(left_idx, step_id);
				}
				cells[site_idx].left_considered = step_id;
			}

			// consider right bond
			if (cells[right_idx].left_considered != step_id)
			{
				if (check_probability(site_idx, right_idx))
				{
					to_consider[n_to_consider++] = right_idx;
					flip(right_idx, step_id);
				}
				cells[right_idx].left_considered = step_id;
			}

			// consider downward bond
			if (cells[site_idx].down_considered != step_id)
			{
				if (check_probability(site_idx, down_idx))
				{
					to_consider[n_to_consider++] = down_idx;
					flip(down_idx, step_id);
				}
				cells[site_idx].down_considered = step_id;
			}

			// consider upward bond
			if (cells[up_idx].down_considered != step_id)
			{
				if (check_probability(site_idx, up_idx))
				{
					to_consider[n_to_consider++] = up_idx;
					flip(up_idx, step_id);
				}
				cells[up_idx].down_considered = step_id;
			}
		}

		// flip states
		uint32_t new_state = (cells[to_flip[0]].state + rand() % (q - 1)) % q; // random, but different state
		for (uint32_t i = 0; i < n_to_flip; i++)
			cells[to_flip[i]].state = new_state; // all cells in the cluster had the same state to begin with, and will be flipped to the same new state
		n_flipped += n_to_flip;
	}

	vector<potts_t> get_states()
	{
		vector<potts_t> result = vector<potts_t>();
		for (int i = 0; i < L * L; i++)
			result.push_back(cells[i].state);
		return result;
	}
};

vector<potts_t> generate_samples(int q, int L, float T, int n_samples)
{
	Model* model = new Model(q, L, T);
	
	vector<potts_t> batch;
	uint32_t step_id = 1;
	for (int s = 0; s < equilibration_steps + n_samples; s++)
	{
		// run the model until L * L states are flipped
		model->n_flipped = 0;
		for (; model->n_flipped < L * L; step_id++)
			model->step(step_id);

		if (s >= equilibration_steps)
		{
			vector<potts_t> sample = model->get_states();
			batch.insert(batch.end(), sample.begin(), sample.end());
		}
	}
	
	delete model;
	return batch;
}