import numpy as np

FOLDER = "igt_dataset"

def load_dataset(L: int):
    subfolder = f"{FOLDER}/{L=}"

    # load everything
    T = np.load(f"{subfolder}/T.npy")
    configs = np.load(f"{subfolder}/configs.npy")

    # augment by flipping
    configs = np.hstack([configs, -configs])
    
    # create dictionary
    n_per_T = 200 # 100 * 2 due to augmentation
    dataset = {T[i]: configs[i].reshape((n_per_T, -1)) for i in range(T.shape[0])}
    return dataset

if __name__ == "__main__":
    L = 4
    S = (2 * L - 1, 2 * L - 1)
    dataset = load_dataset(L)
    keys = list(dataset.keys())
    print(keys)
    print(keys[len(keys) // 2])
    #print(dataset[keys[0]][0].reshape(S))
    #print(dataset[keys[0]][-1].reshape(S))
    #print(dataset[keys[len(keys) // 2]][0].reshape(S))
    #print(dataset[keys[len(keys) // 2]][-1].reshape(S))
    #print(dataset[keys[-1]][0].reshape(S))
    #print(dataset[keys[-1]][-1].reshape(S))
    print(dataset[keys[0]][0].reshape(S) - dataset[keys[0]][-1].reshape(S))
    print(dataset[keys[len(keys) // 2]][0].reshape(S) - dataset[keys[len(keys) // 2]][-1].reshape(S))
    print(dataset[keys[-1]][0].reshape(S) - dataset[keys[-1]][-1].reshape(S))