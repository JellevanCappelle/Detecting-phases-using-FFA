import numpy as np

FOLDER = "bh_dataset"

def load_dataset(L: int):
    subfolder = f"{FOLDER}/{L=}"

    # load everything
    strings = np.loadtxt(f"{subfolder}/strings.txt")
    n_strings = strings.shape[0]
    probs = np.array([np.loadtxt(f"{subfolder}/probs_W={i}.txt") for i in range(201)])
    probs /= np.sum(probs, axis = 1, keepdims = True) # normalise to keep numpy happy

    # sample strings according to probabilities
    n_per_W = 1000
    dataset = {i / 10: strings[np.random.choice(n_strings, n_per_W, p = probs[i])] for i in range(201)}
    return dataset

