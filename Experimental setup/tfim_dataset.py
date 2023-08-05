import numpy as np
import os
import re
import sys

FOLDER = "tfim_dataset"
def load_data(N):
    dataset = {float(m[1]): np.load(f"{FOLDER}/{N=}/{filename}")
               for filename in os.listdir(f"{FOLDER}/{N=}")
               if (m := re.search(r"g=(\d+\.\d+).npy", filename)) is not None}
    return dataset

# AWFUL AWFUL HACK, all because Jax doesn't run on Windows
if sys.platform != "win32":
    import netket
    from netket.hilbert import Spin
    from netket.operator import Ising
    from netket.operator.spin import sigmax,sigmaz 
    from netket.graph import Chain
    from netket.sampler import MetropolisLocal
    from netket.vqs import MCState
    from netket.optimizer import Sgd
    from netket.driver import VMC
    from netket.models import RBM
    import jax.numpy as jnp
    import netket.nn as nknn
    import flax.linen as nn

    def chain_hamiltonian(hi, N: int, transverse_field: float, J: float):
        g = transverse_field

        H = sum([-g * sigmax(hi, i) for i in range(N)])
        H += sum([J * sigmaz(hi, i) * sigmaz(hi, (i + 1) % N) for i in range(N)])
        return H

    def chain(N: int, transverse_field: float, n_samples: int):
        # define the TFIM
        J = -1
        graph = Chain(length = N, pbc = True)
        hi = Spin(s = 0.5, N = graph.n_nodes)
        H = Ising(hi, graph, transverse_field, J)
        # H = chain_hamiltonian(hi, N, transverse_field, J)

        # from: https://netket.readthedocs.io/en/latest/tutorials/gs-ising.html
        class SymmModel(nn.Module):
            alpha: int

            @nn.compact
            def __call__(self, x):
                # add an extra dimension with size 1, because DenseSymm requires rank-3 tensors as inputs.
                # the shape will now be (batches, 1, Nsites)
                x = x.reshape((-1, 1, x.shape[-1]))
                
                x = nknn.DenseSymm(symmetries = graph.translation_group(),
                                features = self.alpha,
                                kernel_init=nn.initializers.normal(stddev = 0.01))(x)
                x = nn.relu(x)

                # sum the output
                return jnp.sum(x,axis = (-1,-2))

        # sample TFIM
        sampler = MetropolisLocal(hi, reset_chains = True)
        model = SymmModel(alpha = 4)
        # model = RBM(alpha =  4, param_dtype = complex)
        vstate = MCState(sampler, model, n_samples = n_samples)
        optimizer = Sgd(learning_rate = 0.01)
        driver = VMC(H, optimizer, variational_state = vstate, preconditioner = netket.optimizer.SR(diag_shift = 0.1))
        log = netket.logging.RuntimeLog()
        driver.run(n_iter = 600, out = log)
        
        divisor = 50
        def sample_states():
            vstate.reset()
            n = n_samples // divisor
            samples = vstate.sample(n_samples = n)
            return np.array(samples.reshape((-1, samples.shape[2]))[:n])
        return np.vstack([sample_states() for _ in range(divisor)])

    if __name__ == "__main__":
        for N in [20, 40, 80]:
            subfolder = f"{FOLDER}/{N=}"
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            gs = np.linspace(0, 2, 101)
            for g in gs:
                file = f"{subfolder}/{g=}.npy"
                if not os.path.exists(file):
                    print(f"\n\n{g=}")
                    samples = chain(N, g, 1000)
                    np.random.shuffle(samples)
                    print(f"{samples.shape=}")
                    for i in range(5):
                        print(samples[i])
                    np.save(file, samples)
