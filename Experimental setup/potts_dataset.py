import numpy as np
import re
import os
import subprocess

FOLDER = "potts_dataset"
simulator_path = "Potts Simulator.exe"
threads = 8
batch_size = 100

def load(q: int, file: str):
    data = np.load(file)
    data.reshape((data.shape[0], -1))
    
    # convert to one-hot vector per pixel, by indexing the identity matrix with the pixel label
    identity = np.eye(q, dtype = np.uint8) # use uint8 and cast during training to save memory
    return identity[data]

def generate_dataset(q: int, L: int, N_temperatures: int, N_samples: int, path: str):
    # create folder
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return
    
    # call the simulator for each value of T
    Tc = 1 / np.log(1 + np.sqrt(q)) # the simulator uses k_B = J = 1
    for T in np.linspace(0, 2, N_temperatures) * Tc:
        T_str = f"{T:.4f}"
        file = f"{path}/T = {T_str}.npy"
        subprocess.call([simulator_path, str(N_samples), str(q), str(L), T_str, str(threads), str(batch_size), file])

def load_dataset(q: int, L: int):
    path = f"{FOLDER}/q = {q}/L = {L}"
    dataset = {float(m[1]): load(q, f"{path}/{filename}")
               for filename in os.listdir(path)
               if (m := re.search(r"T = (\d+\.\d+).npy", filename)) is not None}
    return dataset

if __name__ == "__main__":
    # generate dataset for different combinations of q and L
    for q in [3, 4, 5]:
        for L in [10, 20, 30, 40]:
            path = f"{FOLDER}/q = {q}/L = {L}"
            generate_dataset(q, L, 1000, 500 if L <= 40 else 200 if L <= 60 else 100, path)