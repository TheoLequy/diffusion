import sys
import os
import threading
from queue import Queue
sys.path.append(os.path.abspath(".."))
from xxhash import xxh64_hexdigest
import jax
import os
import shutil
import jax.numpy as jnp
import numpy as np
import diffrax as dfx
import equinox as eqx
import optimistix as optx
from timeit import timeit
from time import time
from jax import jit, grad, vmap
import jax.random as jr
from numba import njit
from jax.typing import ArrayLike
from jaxtyping import Float, Array, Int, Complex, PyTree

from dask.distributed import Client, LocalCluster
from dask.diagnostics.progress import ProgressBar
import dask.dataframe as dd
import dask.array as da
from dask.delayed import delayed
import pandas as pd
import warnings 
import scipy.sparse
from matplotlib.axes import Axes
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, RadioButtons

from typing import Tuple, Collection, Any, Callable, cast
import importlib
import inspect
from functools import partial
from tqdm.notebook import tqdm, tnrange
import zarr
from zarr.codecs import BloscCodec
try:
    ts = importlib.import_module("tensorstore")
except ModuleNotFoundError:
    ts = None
from mpl_toolkits.mplot3d.axes3d import Axes3D
from helpers.plotting_fields import COLORS
from helpers.plotting_graphs import (
    _get_vertex_position,
    _parse_vertex_label,
    _lighten_color,
    _darken_color,
    _round_sig,
    plot_graph_model_eigenmodes,
    plot_inferred_graph_on_field,
    plot_graph as plot_laplacian_graph,
)
from helpers.plotting_simulation import (
    plot_simulation_sample,
    plot_simulation_propagator,
    plot_simulation_mfpt,
    plot_simulation_disc_stationary_distribution,
    plot_simulation_final_magnetization,
    plot_simulation_model_mri_response,
    plot_cell_residence_times,
    plot_effective_diffusion_matrix,
    plot_simulator_comparison,
)
from helpers.general_helpers import update_transition_array, create_lookup, get_chunks_and_shards, update_propagator_aggregators, update_mfpt_aggregators, update_aggregators
from helpers.cell_subgraph_inference import fit_model_one_center, fit_graph_to_field, p_model_one_center, transition_rate_model_one_center
from helpers.fullgraph_inference import (
    construct_mises_kernel,
    fit_best_model,
    propagator_from_laplacian,
    dehermitianize,
    detailed_balance_error,
    mmd_losses,
    mmd_prop,
    mmd_prop_std,
)
from fields import Field
from lineax import DiagonalLinearOperator
import pickle
import networkx as nx
from matplotlib.lines import Line2D

zarr.config.set({
    'async.concurrency': 6,      # Limit concurrent async operations
    'threading.max_workers': 6,  # Limit Zarr's internal thread pool
})


def get_dask_client(verbose=True, n_cpu_max: int = 16) -> Client:
    # Calculate available resources
    try:
        client = Client.current()
    except ValueError:
        n_workers = min(os.cpu_count() or n_cpu_max, n_cpu_max)

        worker_env = {
            "JAX_PLATFORMS": "cpu",          # Force JAX to CPU on workers
            "CUDA_VISIBLE_DEVICES": "",      # Hide GPU from workers
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
        }
        cluster = LocalCluster(
            env=worker_env,  # Disable GPU usage for Dask workers
            n_workers=n_workers,
            threads_per_worker=1,      # Best for GIL-bound NumPy/Pandas operations
            memory_limit='16GB',         # Adjust based on your total System RAM
            dashboard_address=':8787'   # Monitor progress at http://localhost:8787
        )

        client = Client(cluster)
    
    if verbose:
        print(f"Dask Dashboard available at: {client.dashboard_link}")
    return client

SOLVERS = {"EulerHeun": dfx.EulerHeun(), "ShARK": dfx.ShARK()}

class Simulator(eqx.Module):
    field: Field
    D: Float
    beta: Float
    gamma: Float
    tau_diff: Float
    tau_drift: Float
    n_particles: Int
    dt: Float
    dt_save: Float
    ts: Float[Array, "n_snapshots"]
    t_end: Float
    storage_dir: str
    zarr_root: zarr.Group
    name: str
    seed: Int
    solver: str
    Bfield: Callable[[Float, Float, Float], Float] | None
    Bfield_tag: str | None

    def _zarr_root_path(self) -> str:
        return os.path.join(self.storage_dir, "root.zarr")

    def _open_tensorstore_array(self, name: str):
        """Open an existing Zarr array via TensorStore for read/write access."""
        if ts is None:
            raise ImportError("TensorStore is required for saving arrays. Install the 'tensorstore' package.")

        base_spec = {
            "kvstore": {"driver": "file", "path": self._zarr_root_path()},
            "path": name,
        }
        last_error: Exception | None = None
        for driver in ("zarr3", "zarr"):
            try:
                return ts.open(
                    {**base_spec, "driver": driver},
                    read=True,
                    write=True,
                    open=True,
                ).result()
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Failed to open tensorstore array '{name}' at {self._zarr_root_path()}") from last_error

    def _ts_write(self, store, data: Any, index: Any | None = None) -> None:
        """Write data to a TensorStore array and wait for completion."""
        target = store if index is None else store[index]
        target.write(data).result()

    def _bfield_signature(self) -> str | None:
        if self.Bfield is None:
            return None
        if self.Bfield_tag is not None:
            return f"tag:{self.Bfield_tag}"

        bfield = self.Bfield
        module = getattr(bfield, "__module__", "")
        qualname = getattr(bfield, "__qualname__", getattr(bfield, "__name__", type(bfield).__name__))
        signature_parts = [str(module), str(qualname)]

        try:
            signature_parts.append(inspect.getsource(bfield))
        except (OSError, TypeError):
            code = getattr(bfield, "__code__", None)
            if code is not None:
                signature_parts.append(code.co_code.hex())
                signature_parts.append(str(code.co_consts))

        return xxh64_hexdigest("|".join(signature_parts))

    def __hash__(self) -> Int:
        return hash((hash(self.field), self.D, self.beta, self.n_particles, self.dt, self.t_end, int(self.ts.shape[0]), self.seed, self.solver, self._bfield_signature()))
    
    def static_hash(self) -> str:
        return xxh64_hexdigest(str((self.field.static_hash(), float(self.D), float(self.beta), int(self.n_particles), float(self.dt), float(self.t_end), int(self.ts.shape[0]), int(self.seed), self.solver, self._bfield_signature())))
    
    def __init__(self, field: Field, D: Float = 1., beta: Float | None = 1., gamma: Float | None = None, n_particles: Int = 2**16, storage_dir: str | None = None, 
                 dt: Float | None = None, t_end: Float = 1.0, n_timesteps: Int | None = None, delta_t_save: Float | None = None, 
                 seed: Int = 42, solver: str = "ShARK", name: str | None = None, Bfield: Callable[[Float, Float, Float], Float] | None = None, Bfield_tag: str | None = None, verbose: bool = True):
        self.field = field
        self.n_particles = n_particles
        self.seed = seed
        self.Bfield = Bfield
        self.Bfield_tag = Bfield_tag
        assert (D is None) + (beta is None) + (gamma is None) == 1, "Only two of D, beta and gamma can be specified"
        if D is None:
            assert beta is not None and gamma is not None
            D = 1./(beta * gamma)
        if beta is None:
            assert D is not None and gamma is not None
            beta = 1./(D * gamma)
        if gamma is None:
            assert D is not None and beta is not None
            gamma = 1./(D * beta)
        self.D = D  # diffusion coefficient
        self.beta = beta # inverse temperature
        self.gamma = gamma
        self.tau_diff = field.lambda0 ** 2 / D
        rms_amplitude = jnp.sqrt(jnp.sum(field.EA ** 2)) if field.EA is not None else jnp.sum(jnp.abs(field.A) ** 2)
        self.tau_drift = gamma * field.lambda0 ** 2 / rms_amplitude if rms_amplitude > 0 else jnp.inf
        self.dt = min(1e-2 * self.tau_diff, 2e-3 * self.tau_drift) if dt is None else dt
        if n_timesteps is not None:
            t_end = n_timesteps * self.dt
            if verbose:
                print(f"Overriding t_end to {t_end:.2f} based on n_timesteps={n_timesteps} and dt={self.dt:.2e}")
        self.t_end = t_end
        if delta_t_save is None:
            delta_t_save = float(self.dt)
        self.dt_save = delta_t_save
        self.ts = jnp.arange(0., t_end - delta_t_save/10, delta_t_save)
        assert solver in SOLVERS.keys(), f"only the following solvers are supported: {SOLVERS.keys()}"
        self.solver = solver

        if storage_dir is None:
            if name is not None:
                storage_dir = f"data/{name}_{self.static_hash()}"
            else:
                storage_dir = f"data/Simulator_{self.static_hash()}"

        if name is None:
            name = f"Simulator_{self.static_hash()}"
        self.name = name

        root_dir = storage_dir + "/root.zarr"
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)
            self.zarr_root = zarr.open_group(root_dir, mode="w")
            self.zarr_root.update_attributes({
                "Field_type": type(self.field).__name__,
                "Field_hash": self.field.static_hash(),
                "Field_pickle_hex": self.field.to_pickle_hex(),
                "D": float(self.D), 
                "n_particles": int(self.n_particles),
                "dt": float(self.dt),
                "t_end": float(t_end),
                "delta_t_save": delta_t_save,
                "seed": int(self.seed),
                "name": self.name,
                "has_Bfield": self.Bfield is not None,
                "Bfield_signature": self._bfield_signature(),
                "Bfield_tag": self.Bfield_tag
            })
            if verbose:
                print("created new zarr store")
        else:
            self.zarr_root = zarr.open_group(root_dir)
            if verbose:
                print("opened existing zarr store")

            attrs = {
                "Field_type": type(self.field).__name__,
                "Field_hash": self.field.static_hash(),
                "Field_pickle_hex": self.field.to_pickle_hex(),
                "D": float(self.D),
                "n_particles": int(self.n_particles),
                "dt": float(self.dt),
                "t_end": float(t_end),
                "delta_t_save": delta_t_save,
                "seed": int(self.seed),
                "name": self.name,
                "has_Bfield": self.Bfield is not None,
                "Bfield_signature": self._bfield_signature(),
                "Bfield_tag": self.Bfield_tag
            }
            if self.zarr_root.attrs.asdict() != attrs:
                print(f"Warning: Zarr store attributes do not match Simulator parameters. " +
                       f"This may indicate a mismatch between the stored data and the current Simulator configuration: " +
                       f"previous_values {self.zarr_root.attrs.asdict()} \nThey have been overwritten with the current Simulator parameters.")
                self.zarr_root.attrs.update(attrs)    
    
    @classmethod
    def from_file(cls, storage_dir: str, Bfield: Callable[[Float, Float, Float], Float] | None = None, Bfield_tag: str | None = None, verbose=True):
        assert os.path.exists(storage_dir), f"storage directory {storage_dir} does not exist"
        zarr_root = zarr.open_group(storage_dir + "/root.zarr")
        field_type = zarr_root.attrs["Field_type"]
        D = float(zarr_root.attrs["D"]) # type: ignore
        n_particles = int(zarr_root.attrs["n_particles"]) # type: ignore
        dt = float(zarr_root.attrs["dt"]) # type: ignore
        t_end = float(zarr_root.attrs["t_end"]) # type: ignore
        delta_t_save = float(zarr_root.attrs["delta_t_save"]) # type: ignore
        seed = int(zarr_root.attrs["seed"]) # type: ignore
        name = str(zarr_root.attrs["name"])
        field_pickle_hex = str(zarr_root.attrs["Field_pickle_hex"])
        field = pickle.loads(bytes.fromhex(field_pickle_hex))
        
        if verbose:
            print(
                f"loaded Simulator from {storage_dir} with parameters: Field type: {field_type}, \n"
                f"D: {D:.2g}, n_particles: 2^{int(jnp.log2(n_particles))}, seed: {seed}, name: {name} \n"
                f"dt: {dt:.1g}, t_end: {t_end:.2g}, delta_t_save: {delta_t_save:.2g}, \n"
            )

        return cls(field=field, D=D, n_particles=n_particles, dt=dt, t_end=t_end, storage_dir=storage_dir,
               delta_t_save=delta_t_save, seed=seed, name=name, Bfield=Bfield, Bfield_tag=Bfield_tag)
    
    def run(self, save_continuous: bool = False, save_discretized: bool = False, save_final: bool = False, save_propagator: bool = True, save_mfpts: bool = True, save_transitions: bool = False, batch_size: Int = 2**15, dt_prop_time: float = 0.128, t_max_prop_time: float = 16.384, use_interp=False, recompute=False, verbose=True, t_max_trans: Int = 2**12, save_every_n_batches: Int = 2**4, max_queue_size: Int = 2):
        """Simulate trajectories in batches and optionally stream-save to a Zarr store.

        Returns the Zarr array (opened) when saved, otherwise returns an in-memory numpy array
        with shape (n_particles, n_snapshots, dim_sim), where dim_sim = dim + 1 when Bfield is provided.
        """
        """Simulate trajectories in batches if not already present in the Zarr store."""
        if dt_prop_time <= 0:
            raise ValueError("dt_prop must be positive.")
        if t_max_prop_time <= 0:
            raise ValueError("t_max_prop must be positive.")

        dt_prop_steps = int(dt_prop_time / self.dt)
        t_max_prop_steps = int(t_max_prop_time / self.dt)
        if dt_prop_steps <= 0:
            raise ValueError("dt_prop is too small relative to self.dt; increase dt_prop or decrease self.dt.")
        if t_max_prop_steps <= 0:
            raise ValueError("t_max_prop is too small relative to self.dt; increase t_max_prop or decrease self.dt.")
        if t_max_prop_steps < dt_prop_steps:
            raise ValueError("t_max_prop must be >= dt_prop.")
        
        traj_disc, traj_cont, traj_final, transition_array, prop_array, mfpt_array, stationary_array = None, None, None, None, None, None, None
        if "traj_cont" in self.zarr_root.array_keys() and not (recompute and save_continuous):
            traj_cont = self.zarr_root["traj_cont"]
            save_continuous = False
        if "traj_final" in self.zarr_root.array_keys() and not (recompute and save_final):
            traj_final = self.zarr_root["traj_final"]
            save_final = False
        if "traj_disc" in self.zarr_root.array_keys() and not (recompute and save_discretized):
            traj_disc = self.zarr_root["traj_disc"]
            save_discretized = False
        if "transitions" in self.zarr_root.array_keys() and not (recompute and save_transitions):
            save_transitions = False
        if "propagator" in self.zarr_root.array_keys() and "stationary" in self.zarr_root.array_keys() and not (recompute and save_propagator):
            save_propagator = False 
        if "mfpts" in self.zarr_root.array_keys() and not (recompute and save_mfpts):
            save_mfpts = False
        
        save_statistics = save_propagator or save_mfpts or save_transitions
        sim_dim = int(self.field.dim) + int(self.Bfield is not None)
        traj_cont_has_expected_dim = isinstance(traj_cont, zarr.Array) and traj_cont.shape[-1] == sim_dim

        if not save_continuous and not save_discretized and not save_final and not save_statistics:
            if verbose:
                print("data already exists, not resimulated")
            return self.zarr_root

        if ts is None:
            raise ImportError("TensorStore is required for saving arrays. Install the 'tensorstore' package.")
    
        np_dtype = np.float64 if jax.config.read('jax_enable_x64') else np.float32

        needs_resimulation = (
            save_continuous
            or (save_discretized and not isinstance(traj_cont, zarr.Array))
            or (save_final and not (isinstance(traj_cont, zarr.Array) and traj_cont_has_expected_dim))
            or (save_statistics and not (isinstance(traj_disc, zarr.Array) or isinstance(traj_cont, zarr.Array)))
        )
        if verbose and not needs_resimulation:
            print("trajectories already exist, will not be resimulated")

        ### set up simulator ###
        if needs_resimulation:
            t0, t1 = 0., jnp.max(self.ts)
            if use_interp:
                assert self.field.force_interp is not None, "No force interpolator available in the field."
                drift_pos = jit(lambda t, y, args: self.field.force_interp(y)) # type: ignore
            else:
                drift_pos = jit(lambda t, y, args: self.field.force(y))

            if self.Bfield is None:
                drift = drift_pos
                diffusion = jit(lambda t, y, args: jnp.sqrt(2 * self.D) * DiagonalLinearOperator(jnp.ones(self.field.dim)))
                x_inits_transform = lambda x: x
            else:
                bfield = self.Bfield
                assert bfield is not None and callable(bfield), "Bfield must be callable with signature Bfield(t, x, y)."

                def drift_with_phase(t, y, args):
                    pos = y[:-1]
                    x = pos[0]
                    y_coord = pos[1] if pos.shape[0] > 1 else jnp.asarray(0.0, dtype=pos.dtype)
                    phi_dot = jnp.asarray(bfield(t, x, y_coord), dtype=pos.dtype)
                    return jnp.concatenate([drift_pos(t, pos, args), jnp.array([phi_dot], dtype=pos.dtype)])

                drift = jit(drift_with_phase)

                def diffusion_with_phase(t, y, args):
                    diag = jnp.concatenate([jnp.ones(self.field.dim), jnp.zeros((1,))])
                    return jnp.sqrt(2 * self.D) * DiagonalLinearOperator(diag)

                diffusion = jit(diffusion_with_phase)
                x_inits_transform = lambda x: jnp.concatenate([x, jnp.zeros((x.shape[0], 1), dtype=x.dtype)], axis=1)

            save_continuous_internal = save_continuous or save_final

            solver = SOLVERS[self.solver]
            
            if save_continuous_internal:
                saveat = dfx.SaveAt(ts=self.ts)
            else:
                fn = lambda t, y, args: self.field.discretize(jnp.reshape(y[:self.field.dim], (1, -1)))[0]
                saveat = dfx.SaveAt(ts=self.ts, fn=fn)

            ms = int(self.t_end // self.dt + 1)

            @jit
            @partial(vmap, in_axes=(0, 0))
            def sim_batch(x0, key):
                if verbose:
                    print(f"compiling batch simulation function with JAX")
                brownian_motion = dfx.VirtualBrownianTree(t0, t1, tol=self.dt/2, shape=(sim_dim,), key=key, levy_area=dfx.SpaceTimeLevyArea)
                terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, brownian_motion))
                sol = dfx.diffeqsolve(terms, solver, t0, t1, dt0=self.dt, y0=x0, saveat=saveat, throw=False, max_steps=ms)
                cont, disc, final = None, None, None
                if save_continuous_internal:
                    cont_arr = jnp.asarray(sol.ys)
                    cont = cont_arr
                    final = cont_arr[-1]
                    if save_discretized or save_statistics:
                        disc = self.field.discretize(cont_arr[:, :self.field.dim]) # vmap over times and particles
                else:
                    disc = jnp.asarray(sol.ys) #, dtype=self.field.num_vert.dtype)

                return cont, disc, final, sol.result == dfx.RESULTS.successful 

            ## initialization
        
            init_key, bm_key = jr.split(jr.key(self.seed))
            if verbose:
                print(f"sampling {self.n_particles} initial positions from the stationary distribution")
            x_inits = self.field.sample_stationary_distribution(key=init_key, n_samples=int(self.n_particles), beta=self.beta, max_it=5, verbose=verbose)
            if verbose:
                print(f"initial positions sampled, shape {x_inits.shape}, transforming to simulation space")
            x_inits = x_inits_transform(x_inits)
            bm_keys = jr.split(bm_key, self.n_particles)

        ### zarr setup ###
        n_snap = len(self.ts)
        dim = sim_dim
        num_cells = int(self.field.num_vert)
        batch_size = int(min(batch_size, self.n_particles)) # type: ignore

        if save_continuous:
            shape=(int(self.n_particles), n_snap, dim)
            chunks, shards = get_chunks_and_shards(shape, chunking_axis=0, itemsize=np_dtype().itemsize) # type: ignore
            if verbose:
                print(f"chose chunk size {chunks[0]}, shard_size {shards[0]} for traj_cont, would give {jnp.ceil(self.n_particles / chunks[0]).astype(int)} chunks")
            traj_cont = self.zarr_root.create_array(name="traj_cont", shape=shape, chunks=chunks, shards=shards, dtype=np_dtype, overwrite=recompute, # type: ignore
                                                    compressors=BloscCodec(cname="zstd", clevel=5, typesize=np_dtype().itemsize)) # type: ignore

        if save_final:
            shape = (int(self.n_particles), dim)
            chunks, shards = get_chunks_and_shards(shape, chunking_axis=0, itemsize=np_dtype().itemsize) # type: ignore
            if verbose:
                print(f"chose chunk size {chunks[0]}, shard_size {shards[0]} for traj_final, would give {jnp.ceil(self.n_particles / chunks[0]).astype(int)} chunks")
            traj_final = self.zarr_root.create_array(name="traj_final", shape=shape, chunks=chunks, shards=shards, dtype=np_dtype, overwrite=recompute, # type: ignore
                                                     compressors=BloscCodec(cname="zstd", clevel=5, typesize=np_dtype().itemsize)) # type: ignore
        
        if save_discretized:
            assert self.field.tesselation is not None, "Field tesselation must be defined to save discretized trajectories."
            shape=(int(self.n_particles), n_snap)
            chunks, shards = get_chunks_and_shards(shape, chunking_axis=0, itemsize=self.field.num_vert.dtype.itemsize)
            if verbose:
                print(f"chose chunk size {chunks[0]}, shard_size {shards[0]} for traj_disc, would give {jnp.ceil(self.n_particles / chunks[0]).astype(int)} chunks")
            traj_disc = self.zarr_root.create_array(name="traj_disc", shape=shape, chunks=chunks, shards=shards, dtype=self.field.num_vert.dtype,
                                compressors=BloscCodec(cname="zstd", clevel=9, typesize=self.field.num_vert.dtype.itemsize), overwrite=recompute)
        
        lookup, transition_array_local_copy, prop_aggs, mfpt_aggs, stationary_aggs = None, None, None, None, None
        if save_statistics:
            assert self.field.tesselation is not None, "Field tesselation must be defined to save statistics."

            max_neighbors = int(self.field.tesselation.neighbors.shape[1]) # type: ignore
            if save_transitions:
                shape=(num_cells, max_neighbors + 1, max_neighbors + 1, t_max_trans)
                count_dtype = np.uint32
                chunks, shards = get_chunks_and_shards(shape, chunking_axis=0, itemsize=count_dtype().itemsize)
                if verbose and ("transitions" in self.zarr_root.array_keys() or "propagator" in self.zarr_root.array_keys() or "mfpts" in self.zarr_root.array_keys() or "stationary" in self.zarr_root.array_keys()) and not recompute:
                    print(f"Overwriting existing arrays {list(filter(lambda x: x in ['transitions', 'propagator', 'mfpts', 'stationary'], self.zarr_root.array_keys()))} in zarr root.")

                if verbose:
                    print(f"chose chunk size {chunks[0]}, shard_size {shards[0]} for transitions, would give {jnp.ceil(num_cells / chunks[0]).astype(int)} chunks")
                transition_array = self.zarr_root.create_array(name="transitions", fill_value=0, shape=shape, chunks=chunks, shards=shards, dtype=count_dtype, overwrite=recompute or "transitions" in self.zarr_root.array_keys(),
                                                        compressors=BloscCodec(cname="zstd", clevel=5, typesize=count_dtype().itemsize))
                transition_array_local_copy = np.zeros(shape, dtype=count_dtype)
                lookup = create_lookup(np.asarray(self.field.tesselation.neighbors))

            if save_propagator:
                n_prop_snapshots = t_max_prop_steps // dt_prop_steps + 1
                prop_shape = (n_prop_snapshots, num_cells, num_cells, 2)
                prop_dtype = np.float32
                chunks, shards = get_chunks_and_shards(prop_shape, chunking_axis=0, itemsize=prop_dtype().itemsize)
                prop_array = self.zarr_root.create_array(name="propagator", fill_value=0, shape=prop_shape, chunks=chunks, shards=shards,    dtype=prop_dtype, overwrite=recompute or "propagator" in self.zarr_root.array_keys(),
                                                        compressors=BloscCodec(cname="zstd", clevel=5, typesize=prop_dtype().itemsize))
                
                if verbose:
                    print(f"chose chunk size {chunks[0]}, shard_size {shards[0]} for propagator, would give {jnp.ceil(num_cells / chunks[0]).astype(int)} chunks")

                stationary_shape = (num_cells, 2)
                stationary_array = self.zarr_root.create_array(name="stationary", fill_value=0, shape=stationary_shape, dtype=np.float32, overwrite=recompute or "stationary" in self.zarr_root.array_keys())
            
                prop_shape = (n_prop_snapshots, num_cells, num_cells)
                prop_aggs = (np.zeros(prop_shape, dtype=np.float32), np.zeros(prop_shape, dtype=np.float32))
                stationary_aggs = (np.zeros((num_cells,), dtype=np.float32), np.zeros((num_cells,), dtype=np.float32))

            if save_mfpts:
                mfpt_array = self.zarr_root.create_array(name="mfpts", fill_value=0, shape=(num_cells, num_cells, 3), chunks=(num_cells, num_cells, 2), dtype=np.float32, overwrite=recompute or "mfpt" in self.zarr_root.array_keys(), 
                                                        compressors=BloscCodec(cname="zstd", clevel=5, typesize=np.float32().itemsize))
                mfpt_aggs = (np.zeros((num_cells, num_cells), dtype=np.float32), np.zeros((num_cells, num_cells), dtype=np.float32), np.zeros((num_cells, num_cells), dtype=np.uint32))

        traj_cont_ts = self._open_tensorstore_array("traj_cont") if save_continuous else None
        traj_final_ts = self._open_tensorstore_array("traj_final") if save_final else None
        traj_disc_ts = self._open_tensorstore_array("traj_disc") if save_discretized else None
        transition_array_ts = self._open_tensorstore_array("transitions") if save_transitions else None
        prop_array_ts = self._open_tensorstore_array("propagator") if save_propagator else None
        stationary_array_ts = self._open_tensorstore_array("stationary") if save_propagator else None
        mfpt_array_ts = self._open_tensorstore_array("mfpts") if save_mfpts else None
            
    

        ### batch_processing loop ###
        n_batches = (int(self.n_particles) + batch_size - 1) // batch_size
        save_queue: Queue[tuple] = Queue(maxsize=int(max_queue_size))
        pbar = tqdm(total=n_batches, desc="saving batches", disable=not verbose)
        times = []
        time0 = 0

        use_worker = save_continuous or save_final or save_discretized or save_transitions
        if use_worker:
            def save_worker():
                transition_local = transition_array_local_copy
                while True:
                    item = save_queue.get()
                    if item is None:
                        save_queue.task_done()
                        break
                    start, end, cont, disc, final, success, batch_index, is_last = item
                    if success is not None:
                        success_np = np.asarray(success)
                        if not np.all(success_np):
                            print(f"Warning: {np.sum(~success_np)} trajectories did not finish successfully.")
                    disc_np = np.asarray(disc) if disc is not None else None
                    if save_continuous:
                        self._ts_write(traj_cont_ts, cont, np.s_[start:end, :, :])
                    if save_final:
                        self._ts_write(traj_final_ts, final, np.s_[start:end, :])
                    if save_discretized:
                        self._ts_write(traj_disc_ts, disc, np.s_[start:end, :])
                    if save_transitions:
                        assert disc_np is not None
                        transition_local = update_transition_array(disc_np, lookup=lookup, transition_array=transition_local)
                        if batch_index % save_every_n_batches == 0 or is_last:
                            self._ts_write(transition_array_ts, transition_local)
                    pbar.update(1)
                    save_queue.task_done()

                if save_transitions:
                    self._ts_write(transition_array_ts, transition_local)
            
            worker = threading.Thread(target=save_worker, daemon=True)
            worker.start()

        for batch_index, start in enumerate(range(0, int(self.n_particles), batch_size)):
            end = min(start + batch_size, int(self.n_particles))
            bs = end - start
            time_start = time()
            if needs_resimulation:
                with warnings.catch_warnings():
                    if not save_continuous:
                        warnings.filterwarnings("ignore", message="invalid value encountered in cast")
                    x_init = x_inits[start:end]  # type: ignore
                    keys = bm_keys[start:end]  # type: ignore
                    cont, disc, final, success = sim_batch(x_init, keys)  # type: ignore
            elif isinstance(traj_cont, zarr.Array) and (save_discretized or save_statistics):
                cont_np = np.asarray(traj_cont[start:end, :, :])
                cont = cont_np
                disc = jax.jit(jax.vmap(self.field.discretize))(jnp.asarray(cont_np[:, :, :self.field.dim]))
                final = cont_np[:, -1, :] if save_final else None
                success = jnp.ones(bs, dtype=bool)
            elif isinstance(traj_cont, zarr.Array) and save_final:
                cont = None
                disc = None
                final = np.asarray(traj_cont[start:end, -1, :])
                success = jnp.ones(bs, dtype=bool)
            elif save_statistics and not save_discretized and isinstance(traj_disc, zarr.Array):
                cont = None
                disc = traj_disc[start:end, :]
                final = None
                success = jnp.ones(bs, dtype=bool)
            else:
                raise ValueError("Something I did not think about happened")

            is_last = end == self.n_particles
            if use_worker:
                save_queue.put((start, end, cont, disc, final, success, batch_index, is_last))
            else:
                pbar.update(1)
            
            if save_propagator or save_mfpts:
                assert stationary_aggs is not None
                prop_aggs, mfpt_aggs, stationary_aggs = update_aggregators(
                    jnp.asarray(disc),
                    prop_aggs,
                    mfpt_aggs,
                    stationary_aggs,
                    tmax=t_max_prop_steps,
                    delta_t=dt_prop_steps
                )

                if (batch_index + 1) % save_every_n_batches == 0 or is_last:
                    if save_propagator:
                        prop = prop_aggs[0]/(batch_index + 1)
                        prop_err = jnp.sqrt(prop_aggs[1]/(batch_index + 1) - prop**2) / jnp.sqrt(batch_index) # bessel's correction for std error
                        stationary = stationary_aggs[0]/(batch_index + 1)
                        stationary_err = jnp.sqrt(jnp.maximum(stationary_aggs[1]/(batch_index + 1) - stationary**2, 0.0)) / jnp.sqrt(batch_index + 1)

                        self._ts_write(prop_array_ts, jnp.stack([prop, prop_err], axis=-1))
                        self._ts_write(stationary_array_ts, jnp.stack([stationary, stationary_err], axis=-1))
                    if save_mfpts:
                        mfpt = mfpt_aggs[0]/(batch_index + 1) * self.dt
                        mfpt_err = jnp.sqrt(mfpt_aggs[1]/(batch_index + 1) - mfpt**2) / jnp.sqrt(batch_index) * self.dt
                        mfpt_term_frac = mfpt_aggs[2]/(batch_index + 1)

                        self._ts_write(mfpt_array_ts, jnp.stack([mfpt, mfpt_err, mfpt_term_frac], axis=-1))


            time_end = time()
            if start != 0 and bs == batch_size:
                times.append(time_end - time_start)
            if start == 0 and verbose:
                time0 = time_end - time_start

        if use_worker:
            save_queue.put(None) # type: ignore
            worker.join() # type: ignore
            pbar.close()

        
        times = jnp.array(times)
        attrs = {"mean_compute_time": float(jnp.mean(times)), "std_rel_compute_time": float(jnp.std(times, ddof=1)/jnp.mean(times) if len(times) > 1 else 0.0), 
                 "batch_size": int(batch_size), "compile_time": float(time0 - jnp.mean(times) if len(times) > 0 else time0)}


        if save_continuous:
            traj_cont.attrs.update(attrs) # type: ignore
        if save_final:
            traj_final.attrs.update(attrs) # type: ignore
        if save_discretized:
            traj_disc.attrs.update(attrs) # type: ignore
        if save_statistics:
            if save_transitions:
                transition_array.attrs.update(attrs) # type: ignore
            if save_propagator:
                prop_array.attrs.update(attrs) # type: ignore
                prop_array.attrs.update({"t_max": int(t_max_prop_steps), "delta_t": int(dt_prop_steps)}) # type: ignore
                stationary_array.attrs.update(attrs) # type: ignore
            if save_mfpts:
                mfpt_array.attrs.update(attrs) # type: ignore
            
        
        if verbose:
            print(f"Simulation completed in {attrs['mean_compute_time']:.2f} ± {attrs['std_rel_compute_time']:.2f} seconds per batch (batch size: {attrs['batch_size']}), compile time: {attrs['compile_time']:.2f} s.")
        
        return self.zarr_root

    # --- plotting ---

    def plot_sample(self, ax: Axes | None = None, ax_pot: Axes | None = None, n_sample: Int = 3, ids: Int[Array, "n_sample"] | Collection[Int] | None = None, discrete: bool | None = None, seed: Int = 42, colors: list[str] = COLORS, line_styles: list[str] = ['-', '--', '-.', ':'], plot_args: dict = {"zorder": 10}, plot_field: bool | None = None, field_args: dict = {}, res: Int | None = None, linewidth: float = 1.0, decay_constant: float = 0.2, darken: float = 0.2, arrow_darken: float = 0.4, video_dir: str | None = None):
        return plot_simulation_sample(
            self,
            ax=ax,
            ax_pot=ax_pot,
            n_sample=n_sample,
            ids=ids,
            discrete=discrete,
            seed=seed,
            colors=colors,
            line_styles=line_styles,
            plot_args=plot_args,
            plot_field_flag=plot_field,
            field_args=field_args,
            res=res,
            linewidth=linewidth,
            decay_constant=decay_constant,
            darken=darken,
            arrow_darken=arrow_darken,
            video_dir=video_dir,
        )

    def plot_propagator(self, vert0_id: Int = 0, n_subset_particles: Int | None = None, n_snapshots: Int = 2**10, ax: Axes | None = None, ax_slider: Axes | None = None, ax_check: Axes | None = None, t_index_init: int = 0, show_data: bool = True, show_model: bool = False, show_cell_ids: bool = True, posterior_init: bool = False, verbose: bool = True):
        return plot_simulation_propagator(
            self,
            vert0_id=vert0_id,
            n_subset_particles=n_subset_particles,
            n_snapshots=n_snapshots,
            ax=ax,
            ax_slider=ax_slider,
            ax_check=ax_check,
            t_index_init=t_index_init,
            show_data=show_data,
            show_model=show_model,
            show_cell_ids=show_cell_ids,
            posterior_init=posterior_init,
            verbose=verbose,
        )

    def plot_mfpt(self, in_cell_id: Int, ax: Axes | None = None, ax_check: Axes | None = None, show_cell_ids: bool = True):
        return plot_simulation_mfpt(
            self,
            in_cell_id=in_cell_id,
            ax=ax,
            ax_check=ax_check,
            show_cell_ids=show_cell_ids
        )

    def plot_disc_stationary_distribution(self, ax: Axes | None = None, ax_radio: Axes | None = None, show_cell_ids: bool = True, res: float = 50.0, source_init: str = "data"):
        return plot_simulation_disc_stationary_distribution(
            self,
            ax=ax,
            ax_radio=ax_radio,
            show_cell_ids=show_cell_ids,
            res=res,
            source_init=source_init
        )

    def show_final_magnetization(self, ax: Axes | None = None, cell_averages: bool = False, show_cell_ids: bool = False, marker_size: float = 8.0, alpha: float = 0.6, field_args: dict | None = None):
        return plot_simulation_final_magnetization(
            self,
            ax=ax,
            cell_averages=cell_averages,
            show_cell_ids=show_cell_ids,
            marker_size=marker_size,
            alpha=alpha,
            field_args=field_args,
        )

    def model_mri_response(self, Bfield: Callable[[Float, Float, Float], Float], t_end: float, plot: bool = True, sparse: bool = False, show_complex: bool = True, show_field: bool = False, debug1: bool = False, debug2: bool = False, final_only: bool = False, atol: float = 1e-6, rtol: float = 1e-6, video_dir: str | None = None, verbose: bool = False, num_frames: int | None = None):
        if self.field.dim != 2:
            raise NotImplementedError("model_mri_response is only implemented for 2D fields.")
        if self.field.tesselation is None:
            raise ValueError("Field tesselation is required for model_mri_response.")
        if "edge_weights" not in self.zarr_root.array_keys():
            raise ValueError("No edge_weights found in zarr store. Run fit_graph_model first.")
        if t_end <= 0:
            raise ValueError("t_end must be positive.")
        if atol <= 0:
            raise ValueError("atol must be positive.")
        if rtol <= 0:
            raise ValueError("rtol must be positive.")
        if video_dir is not None and not plot:
            raise ValueError("video_dir requires plot=True.")
        if num_frames is not None and num_frames < 2:
            raise ValueError("num_frames must be >= 2 when provided.")

        lap_h_full = jnp.asarray(np.asarray(self.zarr_root["edge_weights"], dtype=np.float32))
        if "stationary" in self.zarr_root.array_keys():
            stationary_array = self.zarr_root["stationary"]
            assert isinstance(stationary_array, zarr.Array)
            pi_stat = jnp.asarray(np.asarray(stationary_array[:, 0], dtype=np.float32))
        else:
            pi_stat = jnp.asarray(self.field.get_discrete_stationary_distribution(), dtype=jnp.float32)

        pi_stat = jnp.clip(pi_stat, a_min=1e-12)
        pi_stat = pi_stat / jnp.sum(pi_stat)
        pi_sqrt = jnp.sqrt(pi_stat)
        lap_h = jnp.zeros_like(lap_h_full) if debug1 else lap_h_full

        if final_only:
            times = jnp.asarray([float(t_end)], dtype=jnp.float32)
        else:
            if num_frames is not None:
                times = jnp.linspace(0.0, float(t_end), int(num_frames), dtype=jnp.float32)
            else:
                n_save = int(np.floor(float(t_end) / float(self.dt_save))) + 1
                times = jnp.arange(n_save, dtype=jnp.float32) * jnp.float32(self.dt_save)
                if float(times[-1]) < float(t_end):
                    times = jnp.concatenate([times, jnp.array([float(t_end)], dtype=jnp.float32)])
                times = jnp.asarray(jnp.clip(times, a_min=0.0, a_max=float(t_end)), dtype=jnp.float32)
        centers = jnp.asarray(self.field.tesselation.centers, dtype=jnp.float32)
        y0 = jnp.asarray(pi_stat, dtype=jnp.complex64)

        try:
            _ = Bfield(
                jnp.float32(0.0),
                jnp.asarray(centers[0, 0], dtype=jnp.float32),
                jnp.asarray(centers[0, 1], dtype=jnp.float32),
            )
        except NameError as exc:
            raise ValueError(
                "Bfield callable references an undefined external variable. "
                "Define Bfield using only its arguments (t, x, y) or close over values that exist in scope."
            ) from exc

        def eval_bfield(t_val: Float) -> Array:
            if debug2:
                return jnp.zeros((centers.shape[0],), dtype=jnp.float32)
            return jax.vmap(lambda xy: jnp.asarray(Bfield(t_val, xy[0], xy[1]), dtype=jnp.float32))(centers)

        if sparse:
            import jax.experimental.sparse as jsparse
            lap_op = jsparse.BCOO.fromdense(lap_h)

            def apply_laplacian(vec):
                return pi_sqrt * (lap_op @ (vec / pi_sqrt))
        else:
            def apply_laplacian(vec):
                return pi_sqrt * (lap_h @ (vec / pi_sqrt))

        def drift(t, m, args):
            t_val = jnp.asarray(t, dtype=jnp.float32)
            b_vals = eval_bfield(t_val)
            return -jnp.asarray(apply_laplacian(m), dtype=jnp.complex64) + 1j * jnp.asarray(b_vals, dtype=jnp.complex64) * m

        terms = dfx.ODETerm(drift)
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=float(rtol), atol=float(atol))
        saveat = dfx.SaveAt(t1=True) if final_only else dfx.SaveAt(ts=times)
        max_steps = int(jnp.ceil(float(t_end) / float(self.dt)) + 100)

        @eqx.filter_jit
        def solve_mri(y0_in: Complex[Array, "n_cells"]):
            return dfx.diffeqsolve(
                terms,
                solver,
                t0=0.0,
                t1=float(t_end),
                dt0=float(min(self.dt, self.dt_save)),
                y0=y0_in,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                throw=False,
                max_steps=max_steps,
            )

        sol_warmup = solve_mri(y0)
        _ = jax.block_until_ready(sol_warmup.ys)

        if verbose:
            t_start_exec = time()
            sol = solve_mri(y0)
            _ = jax.block_until_ready(sol.ys)
            t_exec = time() - t_start_exec
            print(f"Compiled run complete. Second execution time: {t_exec:.3f} s")
        else:
            sol = sol_warmup
        if not bool(sol.result == dfx.RESULTS.successful):
            raise RuntimeError(f"MRI response integration failed with result {sol.result}.")

        m_t = jnp.asarray(sol.ys, dtype=jnp.complex64)
        if m_t.ndim == 1:
            m_t = m_t[None, :]
        elif m_t.ndim != 2:
            raise ValueError(
                f"Unexpected magnetization output shape {m_t.shape}; expected 1D or 2D array."
            )
        m_per_particle = m_t / jnp.maximum(pi_stat[None, :], 1e-12)

        bfield_values = jax.vmap(eval_bfield)(times)

        y_display = jnp.linspace(
            -self.field.size[1] / 2,
            self.field.size[1] / 2,
            160,
            dtype=jnp.float32,
        )
        if debug2:
            bfield_display = jnp.zeros((times.shape[0], y_display.shape[0]), dtype=jnp.float32)
        else:
            bfield_display = jax.vmap(
                lambda t_val: jax.vmap(lambda y_val: jnp.asarray(Bfield(t_val, jnp.float32(0.0), y_val), dtype=jnp.float32))(y_display)
            )(times)

        plot_output = None
        if plot:
            if final_only:
                fig, ax_cells = plt.subplots(figsize=(6, 5), constrained_layout=True)
                plot_output = plot_simulation_model_mri_response(
                    self,
                    times=np.asarray(times, dtype=np.float32),
                    magnetization=np.asarray(m_per_particle, dtype=np.complex64),
                    bfield_values=np.asarray(bfield_display, dtype=np.float32),
                    bfield_y=np.asarray(y_display, dtype=np.float32),
                    ax=ax_cells,
                    ax_bfield=None,
                    ax_slider=None,
                    show_cell_ids=False,
                    show_complex=show_complex,
                    show_field=show_field,
                    t_index_init=0,
                    video_dir=video_dir,
                )
                ax_cells.set_xlabel(r"$x$")
                ax_cells.set_ylabel(r"$y$")
                ax_cells.tick_params(axis="y", which="both", left=True, labelleft=True)
            else:
                plot_output = plot_simulation_model_mri_response(
                    self,
                    times=np.asarray(times, dtype=np.float32),
                    magnetization=np.asarray(m_per_particle, dtype=np.complex64),
                    bfield_values=np.asarray(bfield_display, dtype=np.float32),
                    bfield_y=np.asarray(y_display, dtype=np.float32),
                    show_cell_ids=False,
                    show_complex=show_complex,
                    show_field=show_field,
                    t_index_init=len(times) - 1,
                    video_dir=video_dir,
                )

        video_path = None
        if plot_output is not None:
            fig_obj = plot_output[0]
            video_path = getattr(fig_obj, "_diffsim_video_path", None)

        return {
            "times": np.asarray(times, dtype=np.float32),
            "magnetization": np.asarray(m_t, dtype=np.complex64),
            "average_magnetization": np.asarray(m_per_particle, dtype=np.complex64),
            "pi_stationary": np.asarray(pi_stat, dtype=np.float32),
            "bfield_values": np.asarray(bfield_values, dtype=np.float32),
            "bfield_display": np.asarray(bfield_display, dtype=np.float32),
            "bfield_display_y": np.asarray(y_display, dtype=np.float32),
            "laplacian_directed": np.asarray(lap_h * pi_sqrt[:, None] / pi_sqrt[None, :], dtype=np.float32),
            "debug1": bool(debug1),
            "debug2": bool(debug2),
            "final_only": bool(final_only),
            "video_path": video_path,
            "plot": plot_output,
        }

    def fit_individual_cell_graph(self, plot: bool = True, recompute: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy graph fit based on individual cell-subgraph inference."""
        if "transitions" not in self.zarr_root.array_keys():
            raise ValueError("No transition data found in the zarr store.")
        if self.field.tesselation is None:
            raise ValueError("Tesselation must be defined to fit the graph model.")

        if not recompute and "graph_transition_matrix" in self.zarr_root.array_keys():
            transition_matrix = np.asarray(self.zarr_root["graph_transition_matrix"])
            eigvals = np.asarray(self.zarr_root["graph_eigvals"])
            eigvecs = np.asarray(self.zarr_root["graph_eigvecs"])
            vertex_labels = list(self.zarr_root.attrs.get("graph_vertex_labels", [])) # type: ignore
        else:
            transition_array = self.zarr_root["transitions"]
            assert isinstance(transition_array, zarr.Array)
            transition_matrix, vertex_labels = fit_graph_to_field(
                transition_array=jnp.asarray(transition_array),
                neighbors=self.field.tesselation.neighbors,
                degrees=self.field.tesselation.degrees,
                dt=float(self.dt),
                verbose=False,
                use_scipy_sparse=False
            )
            if isinstance(transition_matrix, scipy.sparse.spmatrix):
                transition_matrix = np.asarray(transition_matrix)
            transition_matrix = np.asarray(transition_matrix, dtype=np.float32)

            row_sums = transition_matrix.sum(axis=1)
            generator = transition_matrix.copy()
            np.fill_diagonal(generator, -row_sums)

            eigvals, eigvecs = np.linalg.eig(generator)
            eigvals = np.asarray(eigvals, dtype=np.complex64)
            eigvecs = np.asarray(eigvecs, dtype=np.complex64)

            self.zarr_root.create_array(
                name="graph_transition_matrix",
                shape=transition_matrix.shape,
                chunks=transition_matrix.shape,
                dtype=transition_matrix.dtype,
                overwrite=recompute
            )
            self._ts_write(self._open_tensorstore_array("graph_transition_matrix"), transition_matrix)
            self.zarr_root.create_array(
                name="graph_eigvals",
                shape=eigvals.shape,
                chunks=eigvals.shape,
                dtype=eigvals.dtype,
                overwrite=recompute
            )
            self._ts_write(self._open_tensorstore_array("graph_eigvals"), eigvals)
            self.zarr_root.create_array(
                name="graph_eigvecs",
                shape=eigvecs.shape,
                chunks=(eigvecs.shape[0], min(eigvecs.shape[1], 256)),
                dtype=eigvecs.dtype,
                overwrite=recompute
            )
            self._ts_write(self._open_tensorstore_array("graph_eigvecs"), eigvecs)
            self.zarr_root.attrs["graph_vertex_labels"] = vertex_labels

        if plot:
            plot_graph_model_eigenmodes(self, eigvals=eigvals, eigvecs=eigvecs, vertex_labels=vertex_labels)

        return transition_matrix, eigvals, eigvecs

    def fit_graph_model(
        self,
        plot: bool = True,
        recompute: bool = False,
        ignore: int = 0,
        threshold: float = 0.1,
        n_iterations: int = 2**9,
        l1_reg: float = 0.00001,
        l2_reg: float = 0.0,
        solver: optx.AbstractMinimiser | None = None,
        propagator_method: str = "auto",
        sigma: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit full-graph Hermitian Laplacian from propagators.

        Stores the fitted Laplacian in zarr under `edge_weights`.
        """
        if self.field.tesselation is None:
            raise ValueError("Tesselation must be defined to fit the graph model.")
        if "propagator" not in self.zarr_root.array_keys():
            raise ValueError("No propagator data found in the zarr store. Run simulation with save_propagator=True.")
        if "stationary" not in self.zarr_root.array_keys():
            raise ValueError("No stationary data found in the zarr store. Run simulation with save_propagator=True.")

        if not recompute and "edge_weights" in self.zarr_root.array_keys():
            laplacian = np.asarray(self.zarr_root["edge_weights"], dtype=np.float32)
        else:
            prop_array = self.zarr_root["propagator"]
            stationary_array = self.zarr_root["stationary"]
            assert isinstance(prop_array, zarr.Array)
            assert isinstance(stationary_array, zarr.Array)

            props = jnp.asarray(np.asarray(prop_array[:, :, :, 0], dtype=np.float32))
            pi_stat = jnp.asarray(np.asarray(stationary_array[:, 0], dtype=np.float32))
            if "delta_t" not in prop_array.attrs:
                raise ValueError("propagator array is missing attribute 'delta_t'.")
            delta_t_steps = float(cast(float, prop_array.attrs["delta_t"]))
            times = jnp.arange(props.shape[0], dtype=jnp.float32) * jnp.float32(delta_t_steps * float(self.dt))

            start = int(max(ignore, 0))
            if start >= props.shape[0] - 1:
                raise ValueError("ignore is too large; no propagator snapshots left for fitting.")

            kernel = construct_mises_kernel(self.field.tesselation.centers, self.field.size, sigma=sigma)
            laplacian_fit = fit_best_model(
                data_propagators=props[start:],
                times=times[start:],
                pi_stationary=pi_stat,
                kernel=kernel,
                threshold=threshold,
                n_iterations=n_iterations,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                solver=solver,
                verbose=verbose,
                propagator_method=propagator_method,
            )
            laplacian = np.asarray(laplacian_fit, dtype=np.float32)

            self.zarr_root.create_array(
                name="edge_weights",
                shape=laplacian.shape,
                chunks=laplacian.shape,
                dtype=laplacian.dtype,
                overwrite=True,
            )
            self._ts_write(self._open_tensorstore_array("edge_weights"), laplacian)

        eigvals, eigvecs = np.linalg.eigh(np.asarray(laplacian, dtype=np.float64))
        eigvals = np.asarray(eigvals, dtype=np.float32)
        eigvecs = np.asarray(eigvecs, dtype=np.float32)

        self.zarr_root.create_array(
            name="graph_eigvals",
            shape=eigvals.shape,
            chunks=eigvals.shape,
            dtype=eigvals.dtype,
            overwrite=True,
        )
        self._ts_write(self._open_tensorstore_array("graph_eigvals"), eigvals)
        self.zarr_root.create_array(
            name="graph_eigvecs",
            shape=eigvecs.shape,
            chunks=(eigvecs.shape[0], min(eigvecs.shape[1], 256)),
            dtype=eigvecs.dtype,
            overwrite=True,
        )
        self._ts_write(self._open_tensorstore_array("graph_eigvecs"), eigvecs)

        if plot:
            self.plot_graph(show_field=True, recompute_model=False)

        return laplacian, eigvals, eigvecs

    def plot_graph(
        self,
        ax: Axes | None = None,
        show_field: bool = True,
        bounds: Float[Array, "dim 2"] | None = None,
        min_edge_rate: float = 0.0,
        max_edge_width: float = 3.0,
        node_size: float = 45.0,
        show_labels: bool = False,
        recompute_model: bool = False,
        field_kwargs: dict | None = None,
    ) -> tuple[object, Axes]:
        """Plot fitted full-graph Laplacian on top of field geometry."""
        needs_fit = (
            recompute_model
            or "edge_weights" not in self.zarr_root.array_keys()
        )
        if needs_fit:
            self.fit_graph_model(plot=False, recompute=recompute_model)

        if self.field.tesselation is None:
            raise ValueError("Tesselation is required to plot graph.")
        laplacian = np.asarray(self.zarr_root["edge_weights"], dtype=np.float32)
        locations = np.asarray(self.field.tesselation.centers)
        size = np.asarray(self.field.size)
        stationary = None
        if "stationary" in self.zarr_root.array_keys():
            stationary_array = self.zarr_root["stationary"]
            assert isinstance(stationary_array, zarr.Array)
            stationary = np.asarray(stationary_array[:, 0], dtype=np.float32)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        else:
            fig = ax.get_figure()

        if show_field:
            kwargs = {} if field_kwargs is None else dict(field_kwargs)
            kwargs.setdefault("plot_dim", 2)
            kwargs.setdefault("show_cells", False)
            kwargs.setdefault("verbose", False)
            if bounds is not None:
                kwargs.setdefault("bounds", bounds)
            self.field.plot_field(ax_x=ax, ax_k=None, **kwargs)
        elif bounds is not None:
            bounds_np = np.asarray(bounds, dtype=float)
            if bounds_np.shape != (2, 2):
                raise ValueError(f"bounds must have shape (2, 2) for 2D fields, got {bounds_np.shape}.")
            ax.set_xlim(float(bounds_np[0, 0]), float(bounds_np[0, 1]))
            ax.set_ylim(float(bounds_np[1, 0]), float(bounds_np[1, 1]))

        plot_laplacian_graph(
            locations=locations,
            laplacian=laplacian,
            size=size,
            stationary=stationary,
            ax=ax,
            node_size_scale=node_size,
            edge_width_scale=max_edge_width,
            min_edge_weight=min_edge_rate,
        )
        if bounds is not None:
            bounds_np = np.asarray(bounds, dtype=float)
            if bounds_np.shape != (2, 2):
                raise ValueError(f"bounds must have shape (2, 2) for 2D fields, got {bounds_np.shape}.")
            ax.set_xlim(float(bounds_np[0, 0]), float(bounds_np[0, 1]))
            ax.set_ylim(float(bounds_np[1, 0]), float(bounds_np[1, 1]))
            ax.set_aspect("equal", adjustable="box")
        _ = show_labels
        return fig, ax

    def show_fit_diagnostics(
        self,
        j_init: int | bool | None = None,
        ax_lap: Axes | None = None,
        ax_metrics: Axes | None = None,
        ax_data: Axes | None = None,
        ax_model: Axes | None = None,
        ax_err: Axes | None = None,
        ax_std: Axes | None = None,
        ax_slider: Axes | None = None,
    ) -> tuple[object, tuple[Axes, Axes, Axes, Axes, Axes, Axes], Slider | None]:
        """Show six-panel propagator diagnostics with a float-time slider."""
        if "propagator" not in self.zarr_root.array_keys():
            raise ValueError("No propagator data found in zarr store.")
        if "stationary" not in self.zarr_root.array_keys():
            raise ValueError("No stationary data found in zarr store.")
        if "edge_weights" not in self.zarr_root.array_keys():
            self.fit_graph_model(plot=False, recompute=False)

        prop_array = self.zarr_root["propagator"]
        stationary_array = self.zarr_root["stationary"]
        assert isinstance(prop_array, zarr.Array)
        assert isinstance(stationary_array, zarr.Array)
        props = jnp.asarray(np.asarray(prop_array[:, :, :, 0], dtype=np.float32))
        props_std = jnp.asarray(np.asarray(prop_array[:, :, :, 1], dtype=np.float32))
        pi_stat = jnp.asarray(np.asarray(stationary_array[:, 0], dtype=np.float32))
        laplacian = jnp.asarray(np.asarray(self.zarr_root["edge_weights"], dtype=np.float32))
        if self.field.tesselation is None:
            raise ValueError("Field tessellation is required for diagnostics kernel construction.")

        if "delta_t" not in prop_array.attrs:
            raise ValueError("propagator array is missing attribute 'delta_t'.")
        delta_t_steps = float(cast(float, prop_array.attrs["delta_t"]))
        times_all = jnp.arange(props.shape[0], dtype=jnp.float32) * jnp.float32(delta_t_steps * float(self.dt))
        positive_mask = times_all > 0
        if not bool(jnp.any(positive_mask)):
            raise ValueError("No positive times available in propagator data; cannot plot diagnostics for t > 0.")
        props = props[positive_mask]
        props_std = props_std[positive_mask]
        times = times_all[positive_mask]
        times_np = np.asarray(times, dtype=np.float32)

        show_time_marker = (j_init is not False)
        if j_init is None or j_init is False:
            j = int(props.shape[0] // 2)
        else:
            j = int(np.clip(j_init, 0, props.shape[0] - 1))

        centers = jnp.asarray(self.field.tesselation.centers, dtype=jnp.float32)
        box_size = jnp.asarray(self.field.size, dtype=jnp.float32)
        kernel = construct_mises_kernel(centers, box_size, sigma=jnp.float32(1.0))

        detailed_balance_curve = np.asarray(detailed_balance_error(props, pi_stat, kernel), dtype=np.float32)
        mmd_curve = np.asarray(mmd_losses(laplacian, props, times, pi_stat, kernel, data_pi_stationary=pi_stat), dtype=np.float32)
        isotropic_reference = jnp.ones((props.shape[-1], props.shape[-1]), dtype=jnp.float32) * pi_stat[None, :]
        anisotropy_curve = np.asarray(mmd_prop(props, isotropic_reference, kernel, pi_stat), dtype=np.float32)
        variance_curve = np.asarray(mmd_prop_std(props_std, kernel, pi_stat), dtype=np.float32)

        use_widgets = all(a is None for a in [ax_lap, ax_metrics, ax_data, ax_model, ax_err, ax_std, ax_slider])
        metrics_only = (
            ax_metrics is not None
            and ax_lap is None
            and ax_data is None
            and ax_model is None
            and ax_err is None
            and ax_std is None
            and ax_slider is None
        )
        if use_widgets:
            fig = plt.figure(figsize=(14, 16), constrained_layout=True)
            gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 0.09], figure=fig)
            ax_lap = fig.add_subplot(gs[0, 0])
            ax_metrics = fig.add_subplot(gs[0, 1])
            ax_model = fig.add_subplot(gs[1, 0])
            ax_data = fig.add_subplot(gs[1, 1])
            ax_err = fig.add_subplot(gs[2, 0])
            ax_std = fig.add_subplot(gs[2, 1])
            ax_slider = fig.add_subplot(gs[3, :])
        elif metrics_only:
            assert ax_metrics is not None
            fig = ax_metrics.get_figure()
        else:
            assert ax_lap is not None and ax_metrics is not None
            assert ax_data is not None and ax_model is not None and ax_err is not None and ax_std is not None
            fig = ax_data.get_figure()

        assert fig is not None

        assert ax_metrics is not None
        if metrics_only:
            ax_metrics.clear()

        ax_metrics.plot(times_np, anisotropy_curve, label="Uniform Baseline Loss")
        ax_metrics.plot(times_np, mmd_curve, label="Best Fit Loss")
        ax_metrics.plot(times_np, variance_curve, label="Data Variance Loss")
        ax_metrics.plot(times_np, detailed_balance_curve, label="Detailed Balance Loss")
        
        ax_metrics.set_xlabel("t")
        ax_metrics.set_ylabel("MMD Loss")
        
        time_marker = None
        if show_time_marker:
            time_marker = ax_metrics.axvline(float(times_np[j]), color="k", linestyle="--", label="Current time")
        ax_metrics.set_yscale("log")
        ax_metrics.legend()

        if metrics_only:
            # Reuse ax_metrics placeholders to keep return signature without adding zero-size axes.
            return fig, (ax_metrics, ax_metrics, ax_metrics, ax_metrics, ax_metrics, ax_metrics), None

        assert ax_lap is not None and ax_data is not None and ax_model is not None and ax_err is not None and ax_std is not None
        model_prop = jnp.asarray(dehermitianize(propagator_from_laplacian(laplacian, times[j]), pi_stat), dtype=jnp.float32)

        lap_vmax = float(jnp.max(jnp.abs(laplacian))) if laplacian.size > 0 else 1.0
        if lap_vmax <= 0:
            lap_vmax = 1.0
        img_lap = ax_lap.imshow(np.asarray(laplacian, dtype=np.float32), cmap="bwr", vmin=-lap_vmax, vmax=lap_vmax)
        cbar_lap = fig.colorbar(img_lap, ax=ax_lap)
        _ = cbar_lap
        ax_lap.set_title("Fitted Laplacian")

        pos_data = np.asarray(props[j][props[j] > 0], dtype=np.float32)
        pos_model = np.asarray(model_prop[model_prop > 0], dtype=np.float32)
        min_pos = float(np.min(np.concatenate([pos_data, pos_model]))) if (pos_data.size + pos_model.size) > 0 else 1e-12
        min_pos = max(min_pos, 1e-12)
        log_data = np.log(np.clip(np.asarray(props[j], dtype=np.float32), a_min=min_pos, a_max=None))
        log_model = np.log(np.clip(np.asarray(model_prop, dtype=np.float32), a_min=min_pos, a_max=None))
        vmin_data = float(min(np.min(log_data), np.min(log_model)))
        vmax_data = float(max(np.max(log_data), np.max(log_model)))
        if vmin_data == vmax_data:
            vmax_data = vmin_data + 1e-6

        img_data = ax_data.imshow(log_data, vmin=vmin_data, vmax=vmax_data)
        cbar_data = fig.colorbar(img_data, ax=ax_data)
        ax_data.set_title(f"Data propagator at t={float(times[j]):.3f}")

        img_model = ax_model.imshow(log_model, vmin=vmin_data, vmax=vmax_data)
        cbar_model = fig.colorbar(img_model, ax=ax_model)
        ax_model.set_title(f"Model propagator at t={float(times[j]):.3f}")

        err = np.asarray(props[j] - model_prop, dtype=np.float32)
        vmax_err = float(np.max(np.abs(err))) if err.size > 0 else 1.0
        if vmax_err <= 0:
            vmax_err = 1e-6
        img_err = ax_err.imshow(err, cmap="bwr", vmin=-vmax_err, vmax=vmax_err)
        cbar_err = fig.colorbar(img_err, ax=ax_err)
        ax_err.set_title("Data - model")

        std0 = np.asarray(props_std[j], dtype=np.float32)
        std_vmin = float(np.min(std0)) if std0.size > 0 else 0.0
        std_vmax = float(np.max(std0)) if std0.size > 0 else 1.0
        if std_vmin == std_vmax:
            std_vmax = std_vmin + 1e-6
        img_std = ax_std.imshow(std0, vmin=std_vmin, vmax=std_vmax)
        cbar_std = fig.colorbar(img_std, ax=ax_std)
        ax_std.set_title("Data standard deviation")

        slider = None
        if use_widgets and ax_slider is not None:
            slider = Slider(
                ax_slider,
                r"$t$",
                valmin=float(times_np[0]),
                valmax=float(times_np[-1]),
                valinit=float(times_np[j]),
                valstep=times_np,
                valfmt="%.3f",
            )

            def _update(val: float) -> None:
                jj = int(np.argmin(np.abs(times_np - float(val))))
                prop_m = np.asarray(dehermitianize(propagator_from_laplacian(laplacian, times[jj]), pi_stat), dtype=np.float32)
                data_j = np.asarray(props[jj], dtype=np.float32)
                err_loc = data_j - prop_m
                pos_d = np.asarray(props[jj][props[jj] > 0], dtype=np.float32)
                pos_m = prop_m[prop_m > 0]
                min_pos_loc = float(np.min(np.concatenate([pos_d, pos_m]))) if (pos_d.size + pos_m.size) > 0 else 1e-12
                min_pos_loc = max(min_pos_loc, 1e-12)

                data_log = np.log(np.clip(data_j, a_min=min_pos_loc, a_max=None))
                model_log = np.log(np.clip(prop_m, a_min=min_pos_loc, a_max=None))
                vmin_loc = float(min(np.min(data_log), np.min(model_log)))
                vmax_loc = float(max(np.max(data_log), np.max(model_log)))
                if vmin_loc == vmax_loc:
                    vmax_loc = vmin_loc + 1e-6

                img_data.set_data(data_log)
                img_model.set_data(model_log)
                img_data.set_clim(vmin_loc, vmax_loc)
                img_model.set_clim(vmin_loc, vmax_loc)
                cbar_data.update_normal(img_data)
                cbar_model.update_normal(img_model)

                img_err.set_data(err_loc)
                vmax_err_loc = float(np.max(np.abs(err_loc))) if err_loc.size > 0 else 1.0
                if vmax_err_loc <= 0:
                    vmax_err_loc = 1e-6
                img_err.set_clim(-vmax_err_loc, vmax_err_loc)
                cbar_err.update_normal(img_err)

                std_j = np.asarray(props_std[jj], dtype=np.float32)
                std_vmin_loc = float(np.min(std_j)) if std_j.size > 0 else 0.0
                std_vmax_loc = float(np.max(std_j)) if std_j.size > 0 else 1.0
                if std_vmin_loc == std_vmax_loc:
                    std_vmax_loc = std_vmin_loc + 1e-6
                img_std.set_data(std_j)
                img_std.set_clim(std_vmin_loc, std_vmax_loc)
                cbar_std.update_normal(img_std)

                if time_marker is not None:
                    time_marker.set_xdata([times_np[jj], times_np[jj]])
                ax_data.set_title(f"Data propagator at t={float(times[jj]):.3f}")
                ax_model.set_title(f"Model propagator at t={float(times[jj]):.3f}")
                fig.canvas.draw_idle()

            slider.on_changed(_update)

        return fig, (ax_lap, ax_metrics, ax_data, ax_model, ax_err, ax_std), slider

    def plot_cell_residence_times(self, cell_id: Int = 0, padding = 0.3, cmap: str = "tab10", transition_rates=True, t_max_multiplier: Float | None = None, verbose=True, yscale: str = "log", ax_cell: Axes | None = None, ax_all: Axes | None = None, ax_slider: Axes | None = None, ax_radio_in: Axes | None = None, ax_radio_metric: Axes | None = None, ax_check: Axes | None = None, ax_check_toggle: Axes | None = None, in_selection_init: str = "all", out_selection_init: Collection[int] | None = None, metric_init: str | None = None, t_max_init: float | None = None, toggle_data: bool = True, toggle_model: bool = True, toggle_graph: bool = True):
        return plot_cell_residence_times(
            self,
            cell_id=cell_id,
            padding=padding,
            cmap=cmap,
            transition_rates=transition_rates,
            t_max_multiplier=t_max_multiplier,
            verbose=verbose,
            yscale=yscale,
            ax_cell=ax_cell,
            ax_all=ax_all,
            ax_slider=ax_slider,
            ax_radio_in=ax_radio_in,
            ax_radio_metric=ax_radio_metric,
            ax_check=ax_check,
            ax_check_toggle=ax_check_toggle,
            in_selection_init=in_selection_init,
            out_selection_init=out_selection_init,
            metric_init=metric_init,
            t_max_init=t_max_init,
            toggle_data=toggle_data,
            toggle_model=toggle_model,
            toggle_graph=toggle_graph
        )

    # --- data analysis ---
    def effective_diffusion_matrix(self, recompute=False, verbose=False, plot=True, num_subgroups=2**4, ax: Axes | None = None, ax_ell: Axes | None = None) -> Float[Array, "dim dim"]:
        assert "traj_cont" in self.zarr_root.array_keys(), "No continuous trajectory data found in the zarr store."
        if "sq_displacement" in self.zarr_root.array_keys() and not recompute:
            Exsq = self.zarr_root["sq_displacement"]
            assert isinstance(Exsq, zarr.Array)
            num_subgroups = Exsq.shape[0]
            if verbose:
                print(f"using precomputed mean squared displacements for {num_subgroups} subgroups")
        else:
            traj = self.zarr_root["traj_cont"]
            assert isinstance(traj, zarr.Array)
            traj_da = da.from_array(traj, chunks=traj.chunks) # type: ignore
            subgroupsize = jnp.ceil(self.n_particles / num_subgroups).astype(int)
            
            def compute_Exsq(subgroup_idx):
                start = subgroup_idx * subgroupsize
                end = min((subgroup_idx + 1) * subgroupsize, self.n_particles)
                traj_sub = traj_da[int(start):int(end), :, :self.field.dim]
                displacements = traj_sub - traj_sub[:, 0:1]
                Exsq_sub = da.mean(displacements[:,:,:,None] * displacements[:,:,None,:], axis=0)
                return Exsq_sub

            Exsq = self.zarr_root.create_array(name="sq_displacement", shape=(num_subgroups, len(self.ts), self.field.dim, self.field.dim), chunks=(num_subgroups, len(self.ts), self.field.dim, self.field.dim), dtype=traj.dtype,
                                    compressors=BloscCodec(cname="zstd", clevel=9), overwrite=recompute) # type: ignore
            da.stack([compute_Exsq(i) for i in range(num_subgroups)], axis=0).to_zarr(Exsq, compute=True)
            if verbose:
                print(f"computed mean squared displacements for {num_subgroups} subgroups")

        assert isinstance(Exsq, zarr.Array)

        Ds = jnp.array([jnp.polyfit(self.ts, Exsq[i].reshape(len(self.ts), -1), deg=1)[0]/2 for i in range(num_subgroups)]).reshape(num_subgroups, self.field.dim, self.field.dim) # type: ignore
        D = jnp.mean(Ds, axis=0)
        D_err = jnp.std(Ds, axis=0) / jnp.sqrt(num_subgroups)

        if verbose:
            print(f"D_empirical {D:.3f} ± {D_err:.3f}")

        if plot:
            plot_effective_diffusion_matrix(self, Exsq=jnp.asarray(Exsq), D=D, D_err=D_err, ax=ax, ax_ell=ax_ell)
        return D
 
    # --- data management --- 
    def clear_data(self):
        """Delete the data file if it exists."""
        confirm = input("Are you sure you want to delete all simulation data? This action cannot be undone. Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Data deletion cancelled.")
            return

        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
            print("removed files")
        else:
            print("found no files to remove")


def compare_simulators(field: Field, dts: Collection[float], solvers: Collection[str] | str, delta_t_save: Float = 1e-2, t_end: Float = 3., n_particles: Int = 2**10, seed: Int = 42, batch_size: Int = 2**6, verbose: bool = True, colors: list[str] = COLORS, line_styles: list[str] = ['-', '--', '-.', ':'], recompute: bool = True, ids: Int[Array, "n_sample"] | None = None):
    if isinstance(solvers, str):
        solvers = tuple([solvers] * len(dts))
    assert len(dts) == len(solvers), "dts and solvers must have the same length"
    
    # Create comparison subfolder
    comparison_dir = f"data/sim_comparison_n{n_particles:d}_b{batch_size:d}_t{t_end:.1f}"
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir, exist_ok=True)
    
    sims = [Simulator(field, n_particles=n_particles, t_end=t_end, dt=dt, delta_t_save=delta_t_save, seed=seed, solver=solver, storage_dir=os.path.join(comparison_dir, f"dt{dt:.1e}_{solver}"), verbose=verbose) for i, (dt, solver) in enumerate(zip(dts, solvers))]
    roots = [sim.run(save_continuous=True, save_discretized=False, batch_size=batch_size, recompute=recompute, verbose=verbose) for sim in sims]
    times = jnp.array([root["traj_cont"].attrs["mean_compute_time"] for root in roots])
    
    traj_ref = roots[-1]["traj_cont"]
    errors = []
    mean_errors = []
    for i, (sim, root) in enumerate(zip(sims, roots)):
        traj = root["traj_cont"]
        if i == len(sims) - 1:
            errors.append(jnp.nan)
            mean_errors.append(jnp.zeros_like(sim.ts))
            break

        error = jnp.linalg.norm(traj_ref[:] - traj[:], axis=-1)  # type: ignore
        mean_error = jnp.mean(error, axis=0)
        mean_errors.append(mean_error)
        errors.append(jnp.mean(mean_error))

    plot_simulator_comparison(sims, roots, mean_errors=mean_errors, errors=jnp.array(errors), times=times, colors=colors, line_styles=line_styles, ids=ids)
