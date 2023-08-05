import numpy as np
import math
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from typing import Optional
import os
import re

r = 1001
n = 100
k = 20
max_W = 10
correlation_features = False
permutations = min(math.factorial(k), 10) if k is not None else None

FOLDER = "mbl_dataset"

if k is not None:
    SUBFOLDER = f"{FOLDER}/{r=}, {n=}, {k=}, {max_W=}"
else:
    SUBFOLDER = f"{FOLDER}/{r=}, {n=}, {max_W=} (wavefunction)"

def load_data(L):
    subfolder = f"{SUBFOLDER}/{L=}"
    dataset = {float(m[1]): np.load(f"{subfolder}/{filename}")
               for filename in os.listdir(subfolder)
               if (m := re.search(r"W=(\d+\.\d+).npy", filename)) is not None}
    
    if permutations is not None and k is not None:
        for P in dataset.keys():
            new_rows = []
            for row in dataset[P]:
                if correlation_features:
                    # calculate correlation features
                    row = np.reshape(row, (k, L)).sum(axis = 0)
                    new_rows.append(((2 * row / k - 1) ** 2).flatten())
                else:
                    # generate random permutations to augment the dataset
                    for _ in range(permutations):
                        p = np.random.permutation(k)
                        permuted_row = np.reshape(row, (k, L))[p]
                        new_rows.append(permuted_row.flatten())
            dataset[P] = np.array(new_rows)
    return dataset

def binary_conv(x, L):
	'''
	convert base-10 integer to binary and adds zeros to match length
	returns: Binary
	'''
	b = bin(x).split('b')[1]
	while len(b) < L:
		b = '0'+b
	return b

def count_ones(bitstring):
    return np.sum([1 for a in bitstring if a == '1'])

def energy_diag(bitString, V, U):
	E = 0
	for index, i in enumerate(bitString):
		if i =='1':
			E += V[index]
			try:
				if bitString[index+1] == '1':
					E += U[index]

			except IndexError:
				continue
	return E

def construct_basis(L, n = 0):
    s2i = {} # State_to_index
    i2s = {} # index_to_State

    index = 0
    for i in range(int(2**L)): # We could insert a minimum
        binary = binary_conv(i, L)

        if n != 0:
            ones = count_ones(binary)
            if ones == n:
                s2i[binary] = index
                i2s[i] = binary
                index +=1
        else:
            s2i[binary] = index
            i2s[i] = binary
            index += 1

    return s2i, i2s

def construct_hamiltonian(onsite_coeff, nn_coeff, hopping_coeff):
    L = len(onsite_coeff)
    s2i, i2s = construct_basis(L, L // 2)
    num_states = len(s2i)

    H = sparse.lil_matrix((num_states, num_states))

    for key in s2i.keys():
        E = energy_diag(key, onsite_coeff, nn_coeff)
        H[s2i[key], s2i[key]] = E

        for site in range(L):
            try:
                if key[site] == '1' and key[site+1] == '0':
                    new_state = key[:site] + '0' + '1' + key[site+2:]
                    H[s2i[key], s2i[new_state]] = hopping_coeff[site]
                    H[s2i[new_state], s2i[key]] = np.conjugate(hopping_coeff[site])
            except IndexError: # periodic boundary conditions
                continue
    return H, s2i

def sample(W: float, L: int, n: int, k: Optional[int],
           J: float = 1,
           t: float = 2,
           ground_probs: bool = False):
    samples = []
    for i in range(n):
        h = np.random.uniform(-W, W, L)
        H, s2i = construct_hamiltonian(h, J * np.ones(L), -t * np.ones(L))
        i2s = {value: key for key, value in s2i.items()}
        states = [i2s[i] for i in sorted(i2s.keys())]

        evals, evecs = linalg.eigsh(H, k = H.shape[0] // 2, which='SA')
        if ground_probs:
            state = evecs[:, 0] # ground state
        else:
            state = evecs[:, -1] # halfway the spectrum
        probabilities = state**2
        if k is not None:
            samples.append("".join(np.random.choice(states, k, p = probabilities)))
        else:
            samples.append(probabilities)
    return np.array([[1 if s == '1' else 0 for s in state] for state in samples]) if k is not None else np.vstack(samples)

if __name__ == "__main__":
    # generate the dataset if it doens't exist yet
    for L in [8]:
        subfolder = f"{SUBFOLDER}/{L=}"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        for W in np.linspace(0, max_W, r):
            file = f"{subfolder}/{W=}.npy"
            if not os.path.exists(file): 
                print(file)
                np.save(file, sample(W, L, n, k))