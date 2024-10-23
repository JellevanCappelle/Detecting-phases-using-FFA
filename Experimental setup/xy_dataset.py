import numpy as np
import re
import os
import subprocess

FOLDER  = "xy_dataset"
SIMULATOR_PATH = "XY Model Simulator.exe"
threads = 8
batch_size = 100

T_min = 0
T_max = 2.5

def load(file: str):
    data = np.load(file)
    data = data.reshape((data.shape[0], -1)) # flatten the samples
    data = data * np.pi / 128 # convert from [0, 255] to radians
    data = np.hstack([np.sin(data), np.cos(data)]) # convert angles to XY
    return data

def generate_dataset(L: int, N_temperatures: int, N_samples: int, path: str):
    # create folder
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return
    
    # call the simulator for each value of T
    for T in np.linspace(T_min, T_max, N_temperatures):
        T_str = f"{T:.4f}"
        file = f"{path}/T = {T_str}.npy"
        subprocess.call([SIMULATOR_PATH, str(N_samples), str(L), T_str, str(threads), str(batch_size), file])

def load_dataset(L: int):
    path = f"{FOLDER}/L = {L}"
    dataset = {float(m[1]): load(f"{path}/{filename}")
               for filename in os.listdir(path)
               if (m := re.search(r"T = (\d+\.\d+).npy", filename)) is not None}
    return dataset

if __name__ == "__main__":
    # generate dataset for different system sizes
    for L in [10, 20, 40, 80]:
        path = f"{FOLDER}/L = {L}"
        generate_dataset(L, 1000, 500 if L <= 20 else 200 if L <= 40 else 40, path)