import numpy as np
import matplotlib.pyplot as plt
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

    # plot everything
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    l1 = ax1.plot(order_curve[0], order_curve[1], label = "learned order parameter", c = "black")
    l2 = ax2.plot(d1[0], d1[1], label = "1st derivative", c = "red", alpha = 0.6)
    ls = l1 + l2
    ax1.grid(visible = True, which = "major")
    ax1.set_ylabel("learned order parameter")
    ax1.set_xlabel(f"${exp.param_name}$")
    ax2.set_ylabel(f"derivative of learned order parameter")
    plt.legend(ls, [l.get_label() for l in ls])
    if plot_P_c:
        ax1.plot([order_P_c, order_P_c], [order_curve[1].min(), order_curve[1].max()], color = "black", linestyle = "dotted")
    plt.show()

def plot_pca(exp: Experiment, n = 3000):
    # generate `n` outputs, selected randomly from the validationset
    sample_idx = np.random.permutation(exp.valid_x.shape[0])[:n]
    v = exp.model(exp.valid_x[sample_idx]).numpy() # apply trained model to selected samples

    # fit PCA with data
    pca = PCA(n_components = 2)
    pca.fit(v)

    # data to plot
    x = pca.transform(v).T
    y = exp.valid_y[sample_idx]

    # plot the first two components
    plt.scatter(x[0] * pca.explained_variance_[0], x[1] * pca.explained_variance_[1], c = y, cmap = "seismic", alpha = 0.5)
    plt.title("Network ouputs scattered by leading principal components")
    plt.xlabel("first principal component")
    plt.ylabel("second principal component")
    colorbar = plt.colorbar()
    colorbar.set_label(f"${exp.param_name}$", rotation = "horizontal")
    colorbar.set_alpha(1)
    colorbar.draw_all()
    plt.show()