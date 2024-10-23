from experiment_classes import IsingExperiment, PottsExperiment, XYExperiment, TFIMExperiment, KitaevExperiment, MBLExperiment, BHExperiment, IGTExperiment
from plotting import plot_matrix, plot_order, plot_pca, plot_L2, plot_spectrum, plot_pca_deriv
from potts_plot import plot_potts_spectrum
from ising_plot import plot_ising_spectrum
import numpy as np

# prepare dataset
exp = IsingExperiment(30) # IsingExperiment(30) # PottsExperiment(5, 30) # XYExperiment(40) # TFIMExperiment(40) # KitaevExperiment() # MBLExperiment(8) # BHExperiment(8) # IGTExperiment(4)

# model-specific hyperparameters
lr = 0.01
if exp.dataset_name in ["ising", "potts"]:
    lr *=  10 / exp.L
elif exp.dataset_name == "igt":
    lr *= (4 / exp.L) ** 2
min_offset = 0 # {"ising": 0.1, "potts": 0.05, "tfim": 0.05, "kitaev": 0.05, "mbl": 0.2, "bose-hubbard": 0.2}[exp.dataset_name]
n_epochs = {"ising": 4, "potts": 4, "xy": 4, "tfim": 4, "kitaev": 4, "mbl": 6, "bose-hubbard": 6, "igt": 6}[exp.dataset_name]
if exp.dataset_name == "xy":
    n_epochs *= 1 if exp.L <= 20 else 2 if exp.L <= 40 else 4
batch_size = 50 if not exp.dataset_name in ["ising", "potts", "xy", "igt"] else 200

# initialize and train network
dim = 5
match exp.dataset_name:
    case "xy":
        model = [exp.L ** 2 * 4 // 10, dim] #[exp.L ** 2 * 3 // 10, dim]
    case "igt":
        model = [10 * 16 * exp.L ** 2, dim]
    case _:
        model = [30, dim]

exp.init_model(model, len(model) - 1, lr = lr, min_offset = min_offset)
# exp.init_model([30, 2], 0, lr = lr, min_offset = min_offset)
exp.train(n_epochs, batch_size)

# run experiment
results = exp.run(True)
class_P_c = results["P_c_class"]
sim_P_c = results["P_c_sim"]
print(f"class-based {exp.param_name}_c: {class_P_c}")
print(f"similarity-based {exp.param_name}_c: {sim_P_c}")

# plot results
# if isinstance(exp, PottsExperiment):
#     plot_potts_spectrum(exp)
# if isinstance(exp, IsingExperiment):
#     plot_ising_spectrum(exp)
plot_matrix(exp, results, False)
if "order" in results.keys():
    plot_order(exp, results, 100 if exp.dataset_name == "igt" else 50 if exp.dataset_name == "mbl" else 10)
midpoint = 5 if exp.dataset_name == "bose-hubbard" else 3 if exp.dataset_name == "mbl" else None # adjust color scale midpoint for bose-hubbard and mbl
#if exp.dataset_name == "igt": # TODO: clean up
#    midpoint = 1 / np.log(2 * exp.L ** 2)#3 / np.log(2 * exp.L ** 2)
n_group = 10 if exp.dataset_name in ["mbl", "bose-hubbard", "igt"] else 1
plot_pca(exp, midpoint = midpoint, n_group = n_group, raw = True)
plot_pca(exp, midpoint = midpoint, n_group = n_group)
plot_pca_deriv(exp)
plot_spectrum(exp)
plot_L2(exp)