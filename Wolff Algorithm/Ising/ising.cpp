#include "ising.h"

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace std;

const int equilibration_steps = 100;

struct Cell // TODO: make sure this is packed tightly
{
	uint32_t state : 1;
	uint32_t flipped : 31;
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
	int L;
	float T;

public:
	Model(int L, float T)
	{
		this->L = L;
		this->T = T;
		n_flipped = 0;
		cells = (Cell*)malloc(L * L * sizeof(Cell));
		memset(cells, 0, L * L * sizeof(Cell));
		to_consider = (uint32_t*)malloc(L * L * sizeof(uint32_t));
		to_flip = (uint32_t*)malloc(L * L * sizeof(uint32_t));

		// randomly initialize states
		for (int i = 0; i < L * L; i++)
			cells[i].state = rand() & 1;
	}

	~Model()
	{
		free(cells);
		free(to_consider);
		free(to_flip);
	}

	bool check_probability(uint32_t i, uint32_t j)
	{
		float random = rand() / static_cast<float>(RAND_MAX); // random number in [0, 1]
		return random < 1. - exp(-(2. / T) * (cells[i].state == cells[j].state ? 1. : 0.));
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

		// start at random index
		to_consider[0] = rand() % (L * L);
		n_to_consider = 1;
		n_to_flip = 0;

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
					flip(site_idx, step_id);
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
					flip(site_idx, step_id);
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
					flip(site_idx, step_id);
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
					flip(site_idx, step_id);
					flip(up_idx, step_id);
				}
				cells[up_idx].down_considered = step_id;
			}
		}

		// flip states
		for (uint32_t i = 0; i < n_to_flip; i++)
			cells[to_flip[i]].state ^= 1; // flip by XOR-ing with 1
		n_flipped += n_to_flip;
	}

	vector<ising_t> get_states()
	{
		vector<ising_t> result = vector<ising_t>();
		for (int i = 0; i < L * L; i++)
			result.push_back(cells[i].state == 1 ? 1 : -1);
		return result;
	}
};

vector<ising_t> generate_sample(int L, float T, int n_samples)
{
	Model* model = new Model(L, T);
	
	vector<ising_t> batch;
	uint32_t step_id = 1;
	for (int s = 0; s < equilibration_steps + n_samples; s++)
	{
		// run the model until L * L states are flipped
		model->n_flipped = 0;
		for (; model->n_flipped < L * L; step_id++)
			model->step(step_id);

		if (s >= equilibration_steps)
		{
			vector<ising_t> sample = model->get_states();
			batch.insert(batch.end(), sample.begin(), sample.end());
		}
	}
	
	delete model;
	return batch;
}