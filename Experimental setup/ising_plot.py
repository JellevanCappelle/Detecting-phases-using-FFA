import numpy as np
import matplotlib.pyplot as plt

from experiment_classes import IsingExperiment

threshold = 10

def plot_ising_spectrum(exp: IsingExperiment):
    units = exp.output_dim
    Ps = sorted(exp.data.keys())
    y_by_M = [[] for _ in range(3)]
    y_total = []
    for P in Ps:
        samples = exp.valid_x[(exp.valid_y == P).flatten()]
        leftover = np.full(samples.shape[0], True)
        y_P = np.zeros(units)
        for m in [-1, 1]:
            magnetization = m * samples.mean(axis = 1)
            selection = magnetization.flatten() > 0.75
            leftover &= ~selection
            magnetized = samples[selection]
            y = exp.model(magnetized).numpy()
            y_P += y.sum(axis = 0)
            if selection.sum() > threshold:
                y_by_M[m + 1].append(y.mean(axis = 0))
            else:
                y_by_M[m + 1].append(np.zeros(units))
        demagnetized = samples[leftover]
        y = exp.model(demagnetized).numpy()
        y_P += y.sum(axis = 0)
        if leftover.sum() > threshold:
            y_by_M[1].append(y.mean(axis = 0))
        else:
            y_by_M[1].append(np.zeros(units))
        y_total.append(y_P)
    y_by_M.append(y_total)
    y_by_M = np.array(y_by_M)
    
    # plot separate images
    _, ax = plt.subplots(2, 4, gridspec_kw = {"hspace": 0.1, "wspace": 0.05})
    aspect = (Ps[-1] - Ps[0]) / units
    for m, y in enumerate(y_by_M):
        y = y.T
        ax[0, m].imshow(y[:exp.layers[0]], aspect = aspect, interpolation = "nearest", extent = (Ps[0], Ps[-1], 0, units))
        ax[1, m].imshow(y[exp.layers[0]:], aspect = aspect, interpolation = "nearest", extent = (Ps[0], Ps[-1], 0, units))
        ax[0, m].set_title(["spin down", "demagnetized", "spin up", "total activation"][m])
        ax[0, m].get_xaxis().set_visible(False)
        ax[1, m].set_xlabel("$T$")
        if m == 0:
            ax[0, m].set_ylabel("unit")
            ax[1, m].set_ylabel("unit")
        else:
            ax[0, m].get_yaxis().set_visible(False)
            ax[1, m].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
