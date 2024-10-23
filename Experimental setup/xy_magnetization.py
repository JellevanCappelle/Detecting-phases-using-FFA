from xy_dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt

Ls = [10, 20, 40, 80]

for L in Ls:
    N = L ** 2
    data = load_dataset(L)
    T = list(data.keys())
    M = [np.mean(data[t][:,:N].mean(axis = -1) ** 2 + data[t][:,N:].mean(axis = -1) ** 2) for t in T]
    plt.plot(T, M, label = f"{L = }")
plt.title("Magnetization")
plt.xlabel("T")
plt.ylabel("M")
plt.legend()
plt.show()

for L in Ls:
    N = L ** 2
    data = load_dataset(L)
    fig, ax = plt.subplots(nrows = 2, ncols = 3)
    Ts = list(data.keys())
    Ts = Ts[::(len(Ts) - 1) // 5]
    for n, T in enumerate(Ts):
        i, j = n // 3, n % 3
        samples = data[T]
        sample = samples[np.random.randint(len(samples))]
        angle = np.arctan2(sample[:N], sample[N:]).reshape((L, L))
        angle -= angle.mean()
        angle = (3 * np.pi + angle) % (2 * np.pi) - np.pi
        img = ax[i, j].imshow(angle, cmap = "twilight", vmin = -np.pi, vmax = np.pi)
        fig.colorbar(img)
        ax[i, j].set_title(f"{T = }")
    plt.show()