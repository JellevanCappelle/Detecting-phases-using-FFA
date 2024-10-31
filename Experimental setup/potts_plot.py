import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import scipy.special

from experiment_classes import PottsExperiment

# palette = ['black', 'red', 'lime', 'yellow', 'cyan', 'magenta']
palette = ['white'] + list(colors.TABLEAU_COLORS.keys()) # type: ignore
palette = [colors.to_rgb(c) for c in palette]

threshold = 10

def plot_potts_spectrum(exp: PottsExperiment):
    units = exp.output_dim
    Ps = sorted(exp.data.keys())
    y_by_q = [[] for _ in range(exp.q + 1)]
    y_total = []
    for P in Ps:
        samples = exp.valid_x[(exp.valid_y == P).flatten()]
        leftover = np.full(samples.shape[0], True)
        y_P = np.zeros(units)
        for q in range(exp.q):
            test = np.tile(np.eye(exp.q)[q], exp.L ** 2)
            magnetization = samples.dot(test) / exp.L ** 2
            selection = magnetization.flatten() > 0.5
            leftover &= ~selection
            magnetized = samples[selection]
            y = exp.model(magnetized).numpy()
            y_P += y.sum(axis = 0)
            if selection.sum() > threshold:
                y_by_q[q + 1].append(y.mean(axis = 0))
            else:
                y_by_q[q + 1].append(np.zeros(units))
        demagnetized = samples[leftover]
        y = exp.model(demagnetized).numpy()
        y_P += y.sum(axis = 0)
        if leftover.sum() > threshold:
            y_by_q[0].append(y.mean(axis = 0))
        else:
            y_by_q[0].append(np.zeros(units))
        y_total.append(y_P)
    y_by_q = np.array(y_by_q)
    y_total = np.array(y_total)
    
    # plot separate images
    _, ax = plt.subplots(2, 3, gridspec_kw = {"hspace": 0.2, "wspace": 0.05})
    aspect = (Ps[-1] - Ps[0]) / units
    for q, y in enumerate(y_by_q):
        y = y.T
        i, j = q // 3, q % 3
        ax[i, j].imshow(y, aspect = aspect, interpolation = "nearest", extent = (Ps[0], Ps[-1], 0, units))
        if q == 0:
            ax[i, j].set_title("demagnetized")
        else:
            ax[i, j].set_title(f"state {q}")
        if i == 1:
            ax[i, j].set_xlabel("$T$")
        else:
            ax[i, j].get_xaxis().set_visible(False)
        if j == 0:
            ax[i, j].set_ylabel("unit")
        else:
            ax[i, j].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

    # plot combined image
    plt.imshow(y_total.T, aspect = aspect, interpolation = "nearest", extent = (Ps[0], Ps[-1], 0, units))
    plt.xlabel("$T$")
    plt.ylabel("unit")
    plt.title("total activation")
    plt.tight_layout()
    plt.show()

    # most-active diagram
    max_pref = y_by_q.max(axis = 0, keepdims = True)
    preference = np.where((y_by_q == max_pref) & (max_pref > 0), 1, 0)
    preference[0] += (max_pref == 0)[0]
    img = np.zeros((units, len(Ps), 3))
    for pref, clr in zip(preference, palette):
        y_img = np.tile(pref.T.reshape((units, len(Ps), 1)), (1, 1, 3)) * np.array(clr)
        img += y_img
    handles = []
    for q in range(exp.q + 1):
        if q == 0:
            handles.append(patches.Patch(color = palette[0], label = "demagnetized"))
        else:
            handles.append(patches.Patch(color = palette[q], label = f"state {q}"))
    plt.legend(handles = handles, loc = "upper right")
    plt.imshow(img, aspect = "auto", interpolation = "nearest", extent = (Ps[0], Ps[-1], 0, units))
    plt.xlabel("$T$")
    plt.ylabel("unit")
    plt.show()

def plot_potts_acitivity_by_class(exp: PottsExperiment):
    # collect andsplit validation data by state
    units = exp.output_dim
    y_by_q = [[] for _ in range(exp.q + 1)]
    samples = exp.valid_x
    leftover = np.full(samples.shape[0], True)
    for q in range(exp.q):
        test = np.tile(np.eye(exp.q)[q], exp.L ** 2)
        magnetization = samples.dot(test) / exp.L ** 2
        selection = magnetization.flatten() > 0.5
        leftover &= ~selection
        magnetized = samples[selection]
        y = exp.model(magnetized).numpy()
        y_by_q[q + 1] = y.mean(axis = 0)
    demagnetized = samples[leftover]
    y = exp.model(demagnetized).numpy()
    y_by_q[0] = y.mean(axis = 0)

    # make plot
    y_by_q = np.array(y_by_q)
    y_by_q = y_by_q > y_by_q.mean(axis = 0, keepdims = True)
    fig, ax = plt.subplots()
    ax.imshow(y_by_q, aspect = "auto", interpolation = "nearest", extent = (0, units, 0, exp.q + 1))
    ax.set_yticks(np.arange(0.5, exp.q + 1))
    states = ["demagnetized"] + [f"state {q + 1}" for q in range(exp.q)]
    ax.set_yticklabels(states[::-1])
    ax.set_xlabel("unit")
    plt.tight_layout()
    plt.show()