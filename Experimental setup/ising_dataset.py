import numpy as np
import re
import os
import subprocess

FOLDER  = "ising_dataset"
SIMULATOR_PATH = "Ising Simulator.exe"
threads = 8
batch_size = 100

Tc = 2.27

def compress(file: str):
    # ising samples only need one bit per site
    binary = np.load(file) == 1
    np.save(file, np.packbits(binary, axis = 0))

def load(file: str):
    data = np.load(file)
    data = data.reshape((data.shape[0], -1))
    if data.dtype == np.uint8:
        return np.unpackbits(data, axis = 0).astype(np.int8) * 2 - 1
    return

def generate_dataset(L: int, N_temperatures: int, N_samples: int, path: str):
    # create folder
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return
    
    # call the simulator for each value of T
    for T in np.linspace(1.95, 0.05, N_temperatures) * Tc:
        T_str = f"{T:.4f}"
        file = f"{path}/T = {T_str}.npy"
        subprocess.call([SIMULATOR_PATH, str(N_samples), str(L), T_str, str(threads), str(batch_size), file])
        compress(file)

def load_dataset(L: int):
    path = f"{FOLDER}/L = {L}"
    dataset = {float(m[1]): load(f"{path}/{filename}")
               for filename in os.listdir(path)
               if (m := re.search(r"T = (\d+\.\d+).npy", filename)) is not None}
    return dataset

if __name__ == "__main__":
    # generate dataset for different system sizes
    for L in [10, 20, 30, 40]:
        path = f"{FOLDER}/L = {L}"
        generate_dataset(L, 1000, 1000 if L <= 50 else 200 if L <= 100 else 100, path)