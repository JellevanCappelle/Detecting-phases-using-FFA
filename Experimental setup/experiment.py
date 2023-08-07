from experiment_classes import IsingExperiment, PottsExperiment, TFIMExperiment, KitaevExperiment, MBLExperiment
from plotting import plot_matrix, plot_order, plot_pca

# prepare dataset
exp = IsingExperiment(30) # PottsExperiment(4, 30) # TFIMExperiment(40) # KitaevExperiment() # MBLExperiment(8)

# model-specific hyperparameters
lr = 0.01 if not exp.dataset_name in ["ising", "potts"] else 0.1 / exp.L
min_offset = {"ising": 0.1, "potts": 0.05, "tfim": 0.05, "kitaev": 0.05, "mbl": 0.2}[exp.dataset_name]
n_epochs = {"ising": 10, "potts": 10, "tfim": 4, "kitaev": 4, "mbl": 6}[exp.dataset_name]
batch_size = 50 if not exp.dataset_name in ["ising", "potts"] else 200

# initialize and train network
exp.init_model([100, 100, 5], 2, lr = lr, min_offset = min_offset)
exp.train(n_epochs, batch_size)

# run experiment
results = exp.run(True)
class_P_c = results["P_c_class"]
sim_P_c = results["P_c_sim"]
print(f"class-based {exp.param_name}_c: {class_P_c}")
print(f"similarity-based {exp.param_name}_c: {sim_P_c}")

# plot results
plot_matrix(exp, results)
if "order" in results.keys():
    plot_order(exp, results, 10 if exp.dataset_name != "mbl" else 50)
plot_pca(exp)