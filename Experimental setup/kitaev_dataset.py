import numpy as np

FOLDER = "kitaev_dataset"
def load_data():
    spectrum = np.loadtxt(f"{FOLDER}/spectrum.txt", dtype = np.float32)
    mu = np.loadtxt(f"{FOLDER}/mu.txt", dtype = np.float32)
    return {m: np.array([s]) for m, s in zip(mu, spectrum)}
