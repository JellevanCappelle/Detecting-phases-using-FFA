import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import PCA

from experiment_classes import Experiment

def plot_matrix(exp: Experiment, results: dict, plot_P_c: bool = True):
    Ps = results["Ps"]
    min_P, max_P = Ps.min(), Ps.max()
    plt.imshow(results["matrix"], extent = (min_P, max_P, max_P, min_P), cmap="cividis")
    cbar = plt.colorbar()
    cbar.set_label("similarity")
    plt.ylabel(f"${exp.param_name}_1$")
    plt.xlabel(f"${exp.param_name}_2$")
    if plot_P_c:
        class_P_c = results["P_c_class"]
        sim_P_c = results["P_c_sim"]
        plt.plot([class_P_c, class_P_c], [min_P, max_P], color = "green", linestyle = "dashed", label = f"${exp.param_name}_c$ (class)")
        plt.plot([min_P, max_P], [class_P_c, class_P_c], color = "green", linestyle = "dashed")
        plt.plot([sim_P_c, sim_P_c], [min_P, max_P], color = "red", linestyle = "dashed", label = f"${exp.param_name}_c$ (similarity)")
        plt.plot([min_P, max_P], [sim_P_c, sim_P_c], color = "red", linestyle = "dashed")
        plt.legend()
    plt.show()

def smoothe(curve, n):
    x, y = curve
    x = np.convolve(x, np.ones(n) / n, mode = "valid")
    y = np.convolve(y, np.ones(n) / n, mode = "valid")
    return (x, y)

def derivative(curve):
    x, y = curve
    dy = y[1:] - y[:-1]
    x = (x[1:] + x[:-1]) / 2
    return (x, dy)

def plot_order(exp: Experiment, results: dict, n_smooth: int = 10, plot_P_c: bool = True):
    # obtain smoothed order curve and its (smoothed) derivative
    Ps = results["Ps"]
    order_curve = (Ps, results["order"])
    n_smooth = 10 if not exp.dataset_name == "mbl" else 50
    order_curve = smoothe(order_curve, n_smooth)
    d1 = smoothe(derivative(order_curve), n_smooth)

    # critical point
    order_P_c = d1[0][np.argmax(np.abs(d1[1]))]
    print(f"{order_P_c=}")
    plot_derivative(order_curve, d1, f"${exp.param_name}$", "learned order parameter", order_P_c if plot_P_c else None)

def plot_derivative(curve, d1, xlabel: str, ylabel: str, P_c = None):
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    l1 = ax1.plot(curve[0], curve[1], label = ylabel, c = "black")
    l2 = ax2.plot(d1[0], d1[1], label = "1st derivative", c = "red", alpha = 0.6)
    ls = l1 + l2
    ax1.grid(visible = True, which = "major")
    ax1.set_xlabel(xlabel)
    ax2.spines["right"].set_color("red")
    ax2.tick_params(axis = 'y', which = "both", colors = "red")
    ax2.yaxis.label.set_color("red")
    plt.legend(ls, [l.get_label() for l in ls])
    if P_c is not None:
        ax1.plot([P_c, P_c], [curve[1].min(), curve[1].max()], color = "black", linestyle = "dotted")
    plt.tight_layout()
    plt.show()

def plot_pca(exp: Experiment, n = 3000, n_group = 1, midpoint = None, raw = False):
    # generate `n` outputs, selected randomly from the validationset
    sample_idx = np.random.permutation(exp.valid_x.shape[0])[:n]
    v = exp.valid_x[sample_idx]
    if not raw:
        v = exp.model(v).numpy()

    # fit PCA with data
    pca = PCA(n_components = 2)
    pca.fit(v)

    # data to plot
    x = pca.transform(v).T
    y = exp.valid_y[sample_idx]
    
    # group by y and take average
    if n_group > 1:
        sort_idx = np.argsort(y.flatten())
        x = x[:, sort_idx]
        y = y[sort_idx, :]
        x = np.mean(x.reshape(2, -1, n_group), axis = -1)
        y = np.mean(y.reshape(-1, 1, n_group), axis = -1)
        scramble_idx = np.random.permutation(y.size) # scramble data again to avoid visual bias
        x = x[:, scramble_idx] 
        y = y[scramble_idx, :]

    # plot the first two components
    norm = None if midpoint is None else colors.TwoSlopeNorm(midpoint)
    plt.scatter(x[0] * pca.explained_variance_[0], x[1] * pca.explained_variance_[1], c = y, cmap = "seismic", norm = norm, alpha = 0.5) # type: ignore
    plt.title(("Samples" if raw else "Network outputs") + " scattered by leading principal components")
    plt.xlabel("first principal component")
    plt.ylabel("second principal component")
    colorbar = plt.colorbar()
    colorbar.set_label(f"${exp.param_name}$", rotation = "horizontal")
    colorbar.solids.set_alpha(1) # type: ignore
    # for older versions of matplotlib:
    # colorbar.set_alpha(1)
    # colorbar.draw_all()
    plt.show()

def plot_pca_deriv(exp: Experiment, n_smooth: int = 10, plot_P_c: bool = True):
    # generate outputs from the validationset
    Ps = exp.data.keys()
    v = exp.model(exp.valid_x).numpy()
    Es = [v[exp.valid_y.flatten() == p, :].mean(axis = 0) for p in Ps] # mean embedding for each parameter
    Ps = np.array(list(Ps))
    Es = np.array(Es)

    # fit PCA with data
    pca = PCA(n_components = 1)
    pca.fit(Es)

    # data to plot
    curve = (Ps, pca.transform(Es).flatten())
    if curve[1][0] < curve[1][-1]:
        curve = (curve[0], -curve[1])
    curve = smoothe(curve, n_smooth)
    d1 = smoothe(derivative(curve), n_smooth)

    # critical point
    PCA_P_c = d1[0][np.argmax(np.abs(d1[1]))]
    print(f"{PCA_P_c=}")
    plot_derivative(curve, d1, f"${exp.param_name}$", "first principal component", PCA_P_c if plot_P_c else None)

def plot_L2(exp: Experiment, n = 3000):
    # generate `n` outputs, selected randomly from the validationset
    sample_idx = np.random.permutation(exp.valid_x.shape[0])[:n]
    sample_idx = np.sort(sample_idx)
    v = exp.model(exp.valid_x[sample_idx]).numpy() # apply trained model to selected samples

    # fit PCA with data
    pca = PCA(n_components = 2)
    pca.fit(v)

    # data to plot
    x = pca.transform(v)
    y = exp.valid_y[sample_idx].flatten()

    # plot the l2-norm
    norm = np.linalg.norm(x, axis = -1)
    curve = smoothe((y, norm), 30)
    plt.plot(curve[0], curve[1])
    plt.show()

def plot_spectrum(exp: Experiment):
    Ps = sorted(exp.data.keys())
    x = np.vstack([exp.data[P][0] for P in Ps])
    y = exp.model(x).numpy()

    plt.imshow(y.T, aspect = "auto", interpolation = "nearest", extent = (Ps[0], Ps[-1], 0, y.shape[1]))
    plt.xlabel(f"${exp.param_name}$")
    plt.ylabel("unit")
    plt.show()