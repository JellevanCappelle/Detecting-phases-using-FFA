import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import numpy as np
import os

from FFA_new import FFModel
from FFScalarDiscriminator import FFScalarDiscriminator
from ising_dataset import load_dataset as load_ising
from potts_dataset import load_dataset as load_potts
from xy_dataset import load_dataset as load_xy
from tfim_dataset import load_data as load_tfim
from mbl_dataset import load_data as load_mbl
from kitaev_dataset import load_data as load_kitaev
from bh_dataset import load_dataset as load_bh
from igt_dataset import load_dataset as load_igt


class Experiment:
    def __init__(self):
        # tuning parameter name to be defined by subclass
        self.param_name = ""

        # examples of high order and low order inputs are to be defined by subclasses
        # can also be left at None, in which case no order parameter results will be returned
        self.high_O_inputs = self.low_O_inputs = None

        # shape must be set for 2D experiments
        self.shape = None
        self.convolutional = False

    def prepare_dataset(self, name: str, data: dict, split_per_P: bool = True):
        self.dataset_name = name
        self.data = data
        
        # split dataset
        if split_per_P:
            Ps = np.array(sorted(self.data.keys()))
            train_x, train_y, valid_x, valid_y = [], [], [], []
            for P in Ps:
                n = self.data[P].shape[0]
                samples = self.data[P]
                np.random.shuffle(samples)
                labels = np.full((n, 1), P)
                
                n_valid = n // 5
                train_x.append(samples[n_valid:])
                train_y.append(labels[n_valid:])
                valid_x.append(samples[:n_valid])
                valid_y.append(labels[:n_valid])

            train_x = np.vstack(train_x)
            train_y = np.vstack(train_y)
            valid_x = np.vstack(valid_x)
            valid_y = np.vstack(valid_y)
        else:
            Ps = np.array(list(self.data.keys()))
            np.random.shuffle(Ps)
            n_valid = Ps.shape[0] // 5
            train_x = np.vstack([self.data[P] for P in Ps[n_valid:]])
            train_y = np.vstack(Ps[n_valid:]) # type: ignore
            valid_x = np.vstack([self.data[P] for P in Ps[:n_valid]])
            valid_y = np.vstack(Ps[:n_valid]) # type: ignore
            Ps = np.sort(valid_y.flatten())
        
        self.Ps = Ps
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y

    def init_model(self, layers: list, min_output: int, min_offset: float,
                   n_buckets: int = 10,
                   lr: float = 0.01,
                   momentum: float = 0.9,
                   peer_norm: float = 0.03,
                   peer_norm_delay: float = 0.9,):
        
        # sanity checks for convolutional layers
        if isinstance(layers[0], tuple):
            if self.shape is None:
                raise Exception("This experiment is not suited for convolutional layers.")
            self.convolutional = True
            # reshape dataset
            self.reshape = lambda X: X.reshape((X.shape[0], self.shape[0], self.shape[1], -1)) # type: ignore
            self.train_x = self.reshape(self.train_x)
            self.valid_x = self.reshape(self.valid_x)

        adjusted_lr = lr * (1 - momentum)

        ff_opt = optimizers.SGD(adjusted_lr, momentum)
        self.ffmodel = FFModel(layers, min_output_layer = min_output, mean_weight = peer_norm, mean_delay = peer_norm_delay)
        self.ffmodel.compile(optimizer = ff_opt)
        self.model = FFScalarDiscriminator(self.ffmodel, self.Ps[0], self.Ps[-1], n_buckets, min_offset)
        self.model.compile()
        
        # call model once to intialize all weights
        self.model((tf.constant([self.train_x[0]]), tf.constant([self.train_y[0]])))
        self.layers = layers
        self.output_dim = sum(layers[min_output:])

    def train(self, n_epochs: int, batch_size: int):
        class lr_callback(callbacks.Callback):
            def __init__(self, models, max_epoch):
                self.models = models
                self.max_epoch = max_epoch
            
            def on_epoch_begin(self, epoch, logs = None):
                def gain(epoch):
                    epoch = epoch + 1
                    if epoch <= self.max_epoch / 2:
                        return 1.
                    return (1 + 2 * (self.max_epoch - epoch)) / self.max_epoch
                for model in self.models:
                    lr = model.optimizer.learning_rate
                    lr.assign(lr * gain(epoch) / gain(epoch - 1))
                    print(f"model: {model}, learning rate: {lr.numpy():.6}")
        
        schedule = lr_callback([self.ffmodel], n_epochs)
        self.model.fit(self.train_x, self.train_y, batch_size = batch_size, epochs = n_epochs, validation_data = [self.valid_x, self.valid_y], callbacks = [schedule], verbose = 2)
    
    def save_weights(self, path: str):
        os.makedirs(path)
        self.model.save_weights(path + "/weights")

    def load_weights(self, path: str):
        self.model.load_weights(path + "/weights")

    def run(self, neutral_label: bool):
        # measure derivative & correlation between Ts
        outputs = {}
        unnorm_Vs = {}
        Vs = {}
        for P in self.Ps:
            idx = (self.valid_y == P).flatten()
            outputs[P] = self.model(self.valid_x[idx] if neutral_label else (self.valid_x[idx], self.valid_y[idx])).numpy()
            v = outputs[P].mean(axis = 0)
            unnorm_Vs[P] = v
            Vs[P] = v / np.linalg.norm(v)

        # correlation matrix method
        matrix = np.array([[unnorm_Vs[T1].dot(unnorm_Vs[T2]) for T2 in self.Ps] for T1 in self.Ps])
        norm_matrix = np.array([[Vs[T1].dot(Vs[T2]) for T2 in self.Ps] for T1 in self.Ps])

        # std based
        score_std = lambda i: - np.hstack([norm_matrix[i,:i], norm_matrix[i, i+1:]]).std()
        score_unnorm = lambda i: - np.hstack([matrix[i,:i], matrix[i, i+1:]]).std()
        curve_std = np.array([score_std(i) for i in range(norm_matrix.shape[0])])
        idx_c = np.nanargmax(curve_std[10:-10]) + 10
        P_c_sim_norm = self.Ps[idx_c]
        curve_unnorm = np.array([score_unnorm(i) for i in range(norm_matrix.shape[0])])
        idx_c = np.nanargmax(curve_unnorm[10:-10]) + 10
        P_c_sim = self.Ps[idx_c]

        # classification based
        score = lambda i: 0.5 * norm_matrix[:i, :i].mean() + 0.5 * norm_matrix[i+1:, i+1:].mean() - norm_matrix[:i, i+1:].mean()
        score_unnorm = lambda i: 0.5 * matrix[:i, :i].mean() + 0.5 * matrix[i+1:, i+1:].mean() - matrix[:i, i+1:].mean()
        curve_class = np.array([score(i) for i in range(norm_matrix.shape[0])])
        curve_class_u = np.array([score_unnorm(i) for i in range(matrix.shape[0])])
        idx_c = np.nanargmax(curve_class[10:-10]) + 10
        P_c_class_norm = self.Ps[idx_c]
        idx_c = np.nanargmax(curve_class_u[10:-10]) + 10
        P_c_class = self.Ps[idx_c]

        # operator for retrieving 'order' from activities directly
        if self.high_O_inputs is not None and self.low_O_inputs is not None:
            if self.convolutional:
                self.high_O_inputs = self.reshape(self.high_O_inputs)
                self.low_O_inputs = self.reshape(self.low_O_inputs)
            if neutral_label:
                high_O = self.model(self.high_O_inputs).numpy().mean(axis = 0)
                low_O = self.model(self.low_O_inputs).numpy().mean(axis = 0)
            else:
                # NOTE: this assumes Ps[0] is in the ordered phase and P[-1] in the disordered phase
                high_O = self.model((self.high_O_inputs, np.vstack([self.Ps[0]] * self.high_O_inputs.shape[0]))).numpy().mean(axis = 0)
                low_O = self.model((self.low_O_inputs, np.vstack([self.Ps[-1]] * self.low_O_inputs.shape[0]))).numpy().mean(axis = 0)
            
            orthogonal_O = high_O - low_O.dot(high_O) * low_O / np.linalg.norm(low_O)**2
            op_O = orthogonal_O / np.linalg.norm(orthogonal_O)
            op_O /= op_O.dot(high_O)
            order = np.array([outputs[P].dot(op_O).mean() for P in self.Ps])
        else:
            op_O = order = None

        # return experiment results
        tf.keras.backend.clear_session()
        return {
            "Ps": self.Ps.copy(),
            "matrix": matrix,
            "norm_matrix": norm_matrix,
            "P_c_sim": P_c_sim,
            "P_c_sim_norm": P_c_sim_norm,
            "P_c_class": P_c_class,
            "P_c_class_norm": P_c_class_norm,
        } | ({
            "op_O": op_O,
            "order": order,
        } if op_O is not None else {})

class IsingExperiment(Experiment):
    def __init__(self, L: int, n_low_order: int = 1000):
        super().__init__()
        data = load_ising(L)
        self.prepare_dataset("ising", data)
        self.param_name = "T"
        self.L = L
        self.shape = (L, L)

        # initialize examples of high- and low order inputs
        self.high_O_inputs = np.vstack([np.full((L*L,), 1), np.full((L*L,), -1)])
        self.low_O_inputs = np.random.choice([1, -1], (n_low_order, L*L))

class PottsExperiment(Experiment):
    def __init__(self, q: int, L: int, n_low_order: int = 1000):
        super().__init__()
        data = load_potts(q, L)
        self.prepare_dataset("potts", data)
        self.param_name = "T"
        self.q = q
        self.L = L
        self.shape = (L, L)

        # initialize examples of high- and low order inputs
        self.high_O_inputs = np.tile(np.eye(q), L * L)
        self.low_O_inputs = np.eye(q)[np.random.choice(q, size = (n_low_order, L * L))].reshape(n_low_order, L * L * q)

class XYExperiment(Experiment):
    def __init__(self, L: int):
        super().__init__()
        data = load_xy(L)
        self.prepare_dataset("xy", data) # TODO: add ordered/unordered examples?
        self.param_name = "T"
        self.L = L
        self.shape = (L, L)

class TFIMExperiment(Experiment):
    def __init__(self, N: int, n_low_order: int = 1000):
        super().__init__()
        data = load_tfim(N)
        self.prepare_dataset("tfim", data)
        self.param_name = "g"
        self.N = N

        # initialize examples of high- and low order inputs
        self.high_O_inputs = np.vstack([np.full((N,), 1), np.full((N,), -1)])
        self.low_O_inputs = np.random.choice([1, -1], (n_low_order, N))

class KitaevExperiment(Experiment):
    def __init__(self):
        super().__init__()
        data = load_kitaev()
        self.prepare_dataset("kitaev", data, False)
        self.param_name = "\\mu"

class MBLExperiment(Experiment):
    def __init__(self, L: int, descending_order: bool = True):
        super().__init__()
        data = load_mbl(L)
        self.prepare_dataset("mbl", data)
        self.param_name = "W"
        self.L = L

        # initialize examples of high- and low order inputs
        if descending_order:
            self.high_O_inputs = self.data[self.Ps[0]]
            self.low_O_inputs = self.data[self.Ps[-1]]
        else:
            self.high_O_inputs = self.data[self.Ps[-1]]
            self.low_O_inputs = self.data[self.Ps[0]]

class BHExperiment(Experiment):
    def __init__(self, L: int, descending_order: bool = True):
        super().__init__()
        data = load_bh(L)
        self.prepare_dataset("bose-hubbard", data)
        self.param_name = "W"
        self.L = L

        # initialize examples of high- and low order inputs
        if descending_order:
            self.high_O_inputs = self.data[self.Ps[0]]
            self.low_O_inputs = self.data[self.Ps[-1]]
        else:
            self.high_O_inputs = self.data[self.Ps[-1]]
            self.low_O_inputs = self.data[self.Ps[0]]

class IGTExperiment(Experiment):
    def __init__(self, L: int):
        super().__init__()
        data = load_igt(L)
        self.prepare_dataset("igt", data)
        self.param_name = "T"
        self.L = L
        self.shape = (2 * L - 1, 2 * L - 1)