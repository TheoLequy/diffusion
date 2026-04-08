from typing import Tuple, Dict
from jax import jit, vmap
from equinox import filter_jit
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from functools import partial
import optimistix as optx
import optax
import equinox as eqx
from tqdm.notebook import tqdm
from timeit import timeit
import matplotlib.pyplot as plt


def propagator_method_to_code(method: str) -> int:
    """Map propagator method name to integer code used in JIT-compatible control flow."""
    method_map = {"auto": 0, "expm": 1, "power": 2, "eigh": 3}
    if method not in method_map:
        raise ValueError(f"Unknown propagator method '{method}'. Expected one of {list(method_map.keys())}.")
    return method_map[method]


def cosine_restart_schedule(peak_lr: float = 2e-1, min_lr: float = 2e-2, cycle_steps: int = 20):
    """Cosine-decay schedule with frequent fixed-length restarts."""
    assert peak_lr > 0, "peak_lr must be positive"
    assert min_lr >= 0, "min_lr must be non-negative"
    assert peak_lr >= min_lr, "peak_lr must be >= min_lr"
    assert cycle_steps > 0, "cycle_steps must be positive"

    def schedule(step):
        """Schedule.
        
        Args:
            step: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        step = jnp.asarray(step, dtype=jnp.float32)
        phase = jnp.mod(step, float(cycle_steps)) / float(cycle_steps)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * phase))
        return min_lr + (peak_lr - min_lr) * cosine

    return schedule


def cosine_decay_schedule_no_restarts(peak_lr: float = 2e-1, min_lr: float = 8e-3, decay_steps: int = 256):
    """Smooth cosine-decay schedule without restarts."""
    assert peak_lr > 0, "peak_lr must be positive"
    assert min_lr >= 0, "min_lr must be non-negative"
    assert peak_lr >= min_lr, "peak_lr must be >= min_lr"
    assert decay_steps > 0, "decay_steps must be positive"
    alpha = min_lr / peak_lr
    return optax.cosine_decay_schedule(init_value=peak_lr, decay_steps=decay_steps, alpha=alpha)


def raw_to_edge_weights(raw_edge_weights: Array, eps: float = 1e-12) -> Array:
    """Map unconstrained parameters to strictly-positive edge weights."""
    return jax.nn.softplus(raw_edge_weights) + eps


def edge_weights_to_raw(edge_weights: Array, eps: float = 1e-12) -> Array:
    """Inverse map from positive edge weights to unconstrained parameters."""
    clipped = jnp.clip(edge_weights - eps, a_min=eps)
    return jnp.log(jnp.expm1(clipped))

@jit
def construct_mises_kernel(locations, box_size, sigma: Float = 1.0):
    """Construct mises kernel.
    
    Args:
        locations: Input parameter.
        box_size: Input parameter.
        sigma: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    phases = 2 * jnp.pi * locations / box_size
    cos_distance = 2 - 2 * jnp.cos(phases[:, None, :] - phases[None, :, :])
    return  jnp.exp(- sigma ** (-2) * jnp.sum((box_size[None, None, :] ** 2 / (4 * jnp.pi ** 2) * cos_distance), axis=-1)) * sigma ** 2

@filter_jit
def mmd(p1, p2, kernel, normalize=True):
    """Mmd.
    
    Args:
        p1: Input parameter.
        p2: Input parameter.
        kernel: Input parameter.
        normalize: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    if normalize:
        p1 = p1 / jnp.sum(p1, axis=-1, keepdims=True)
        p2 = p2 / jnp.sum(p2, axis=-1, keepdims=True)
    
    delta = p1 - p2
    return jnp.sum(delta[..., None, :] * delta[..., :, None] * kernel, axis=(-2, -1))

@filter_jit
def mmd_prop(prop1, prop2, kernel, pi_stationary, normalize=True):
    """Mmd prop.
    
    Args:
        prop1: Input parameter.
        prop2: Input parameter.
        kernel: Input parameter.
        pi_stationary: Input parameter.
        normalize: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    return jnp.dot(jnp.sqrt(jnp.clip(mmd(prop1, prop2, kernel, normalize=normalize), a_min=0.0)), pi_stationary)

@filter_jit
def mmd_prop_std(prop_std, kernel, pi_stationary):
    """Mmd prop std.
    
    Args:
        prop_std: Input parameter.
        kernel: Input parameter.
        pi_stationary: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    return jnp.dot(jnp.sqrt(jnp.clip(jnp.sum(prop_std ** 2 * jnp.diag(kernel), axis=-1), a_min=0.0)), pi_stationary)

@filter_jit
def mmd_loss(hermitian_laplacian, data_propagators, times, pi_stationary, kernel, data_pi_stationary=None, propagator_method_code: Int = 0):
    """Mmd loss.
    
    Args:
        hermitian_laplacian: Input parameter.
        data_propagators: Input parameter.
        times: Input parameter.
        pi_stationary: Input parameter.
        kernel: Input parameter.
        data_pi_stationary: Input parameter.
        propagator_method_code: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    model_propagators = dehermitianize(propagator_from_laplacian(hermitian_laplacian, times, propagator_method_code), pi_stationary)
    return jnp.mean(mmd_prop(model_propagators, data_propagators, kernel, pi_stationary if data_pi_stationary is None else data_pi_stationary))

@filter_jit
def mmd_losses(hermitian_laplacian, data_propagators, times, pi_stationary, kernel, data_pi_stationary=None, propagator_method_code: Int = 0):
    """Mmd losses.
    
    Args:
        hermitian_laplacian: Input parameter.
        data_propagators: Input parameter.
        times: Input parameter.
        pi_stationary: Input parameter.
        kernel: Input parameter.
        data_pi_stationary: Input parameter.
        propagator_method_code: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    if data_pi_stationary is None:
        data_pi_stationary = pi_stationary
    model_propagators = dehermitianize(propagator_from_laplacian(hermitian_laplacian, times, propagator_method_code), pi_stationary)
    return mmd_prop(model_propagators, data_propagators, kernel, data_pi_stationary)

@filter_jit
def guesstimate_hermitian_laplacian_from_propagator(propagator, time, pi_stationary, clip=1e-8, enforce_conservation=True):
    """Guesstimate hermitian laplacian from propagator.
    
    Args:
        propagator: Input parameter.
        time: Input parameter.
        pi_stationary: Input parameter.
        clip: Input parameter.
        enforce_conservation: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    prop_eigvals, prop_eigvecs = jnp.linalg.eigh(hermitianize(propagator, pi_stationary))
    eigvals = - jnp.log(jnp.clip(prop_eigvals, a_min=clip)) / time
    herm_laplacian =  (prop_eigvecs * eigvals) @ prop_eigvecs.T
    if enforce_conservation:
        herm_laplacian = herm_laplacian - jnp.diag(jnp.sum(herm_laplacian * jnp.sqrt(pi_stationary)[None,:], axis=-1) / jnp.sqrt(pi_stationary))
    return herm_laplacian


@filter_jit
def guesstimate_hermitian_laplacian_from_propagators(
    propagators: Float[Array, "n_times n_cells n_cells"],
    times: Float[Array, "n_times"],
    pi_stationary: Float[Array, "n_cells"],
    clip: float = 1e-8,
    enforce_conservation: bool = True,
    time_weight_power: float = 2.0,
) -> Float[Array, "n_cells n_cells"]:
    """Estimate Laplacian by averaging per-time eigendecomposition-based estimates."""
    assert propagators.ndim == 3, f"Expected propagators shape (n_times, n_cells, n_cells), got {propagators.shape}"
    assert times.ndim == 1, f"Expected times shape (n_times,), got {times.shape}"
    assert propagators.shape[0] == times.shape[0], "times length must match first axis of propagators"
    safe_times = jnp.clip(times, a_min=clip)

    herm_props = hermitianize(propagators, pi_stationary)
    prop_eigvals, prop_eigvecs = jnp.linalg.eigh(herm_props)
    clipped_prop_eigvals = jnp.clip(prop_eigvals, a_min=clip, a_max=1.0)
    lap_eigvals = -jnp.log(clipped_prop_eigvals) / safe_times[:, None]
    laplacians = jnp.einsum("tki,tk,tkj->tij", prop_eigvecs, lap_eigvals, prop_eigvecs)

    time_weights = 1.0 / (safe_times ** time_weight_power)
    time_weights = time_weights / jnp.sum(time_weights)
    herm_laplacian = jnp.tensordot(time_weights, laplacians, axes=1)

    if enforce_conservation:
        herm_laplacian = herm_laplacian - jnp.diag(
            jnp.sum(herm_laplacian * jnp.sqrt(pi_stationary)[None, :], axis=-1) / jnp.sqrt(pi_stationary)
        )
    return herm_laplacian

@jit
def hermitianize(operator: Float[Array, "... n_cells n_cells"], pi_stationary: Float[Array, "n_cells"]) -> Float[Array, "... n_cells n_cells"]:
    """Hermitianize.
    
    Args:
        operator: Input parameter.
        pi_stationary: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    pi_sqrt = jnp.sqrt(pi_stationary)
    weighted_matrix = pi_sqrt[:,None] * operator / pi_sqrt[None,:]
    return (weighted_matrix + jnp.matrix_transpose(weighted_matrix)) / 2

@jit
def dehermitianize(hermitian_operator: Float[Array, "... n_cells n_cells"], pi_stationary: Float[Array, "n_cells"]) -> Float[Array, "... n_cells n_cells"]:
    """Dehermitianize.
    
    Args:
        hermitian_operator: Input parameter.
        pi_stationary: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    pi_sqrt = jnp.sqrt(pi_stationary)
    return hermitian_operator * pi_sqrt[None,:] / pi_sqrt[:,None]

@jit
def detailed_balance_error(propagator: Float[Array, "n_cells n_cells"], pi_stationary: Float[Array, "n_cells"], kernel) -> Float[Array, "..."]:
    """Detailed balance error.
    
    Args:
        propagator: Input parameter.
        pi_stationary: Input parameter.
        kernel: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    return mmd_prop(propagator, dehermitianize(hermitianize(propagator, pi_stationary), pi_stationary), kernel=kernel, pi_stationary=pi_stationary, normalize=True)

@jit
def hermitianization_error(propagators: Float[Array, "... n_cells n_cells"], pi_stationary: Float[Array, "n_cells"], kernel: Float[Array, "n_cells n_cells"]) -> Float[Array, "..."]:
    """Hermitianization error.
    
    Args:
        propagators: Input parameter.
        pi_stationary: Input parameter.
        kernel: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    return mmd_prop(propagators, dehermitianize(hermitianize(propagators, pi_stationary), pi_stationary), kernel=kernel, pi_stationary=pi_stationary, normalize=True)

@jit
def _propagators_from_base_binary_powers(
    base_propagator: Float[Array, "n_cells n_cells"],
    step_ids: Int[Array, "n_times"],
    max_bits: int = 31,
) -> Float[Array, "n_times n_cells n_cells"]:
    """Compute matrix powers `base_propagator**k` using repeated squaring and bitwise composition."""
    n_cells = base_propagator.shape[0]
    dtype = base_propagator.dtype
    eye = jnp.eye(n_cells, dtype=dtype)

    powers = jnp.broadcast_to(eye, (max_bits, n_cells, n_cells))
    powers = powers.at[0].set(base_propagator)

    max_step_id = jnp.max(jnp.clip(step_ids, a_min=0)).astype(jnp.float32)
    needed_bits = jnp.clip(jnp.floor(jnp.log2(jnp.maximum(max_step_id, 1.0))) + 1.0, a_min=1.0, a_max=float(max_bits)).astype(jnp.int32)

    def square_body(bit_id, powers_arr):
        """Square body.
        
        Args:
            bit_id: Input parameter.
            powers_arr: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        active_bit = jnp.asarray(bit_id, dtype=jnp.int32) < needed_bits
        next_power = powers_arr[bit_id - 1] @ powers_arr[bit_id - 1]
        selected = jnp.where(active_bit, next_power, powers_arr[bit_id])
        return powers_arr.at[bit_id].set(selected)

    powers = jax.lax.fori_loop(1, max_bits, square_body, powers)

    propagators = jnp.broadcast_to(eye, (step_ids.shape[0], n_cells, n_cells))
    step_ids_u = step_ids.astype(jnp.uint32)

    def combine_body(bit_id, current):
        """Combine body.
        
        Args:
            bit_id: Input parameter.
            current: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        active_bit = jnp.asarray(bit_id, dtype=jnp.int32) < needed_bits
        bit_id_u32 = jnp.asarray(bit_id, dtype=jnp.uint32)
        use_bit = ((step_ids_u >> bit_id_u32) & jnp.uint32(1)).astype(bool)
        candidate = current @ powers[bit_id]
        should_use = active_bit & use_bit
        return jnp.where(should_use[:, None, None], candidate, current)

    propagators = jax.lax.fori_loop(0, max_bits, combine_body, propagators)
    return propagators


@jit
def propagator_from_laplacian(laplacian: Float[Array, "n_cells n_cells"], t: Float[Array, "..."], method_code: Int = 0) -> Float[Array, "... n_cells n_cells"]:
    """Propagator from laplacian.
    
    Args:
        laplacian: Input parameter.
        t: Input parameter.
        method_code: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    t = jnp.asarray(t)

    def expm_for_times(lap, times):
        """Expm for times.
        
        Args:
            lap: Input parameter.
            times: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        return jax.scipy.linalg.expm(-lap * times[..., None, None])

    def eigh_for_times(lap, times):
        """Eigh for times.
        
        Args:
            lap: Input parameter.
            times: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        evals, evecs = jnp.linalg.eigh(lap)
        decay = jnp.exp(-times[..., None] * evals[None, :])
        return jnp.einsum("...k,ik,jk->...ij", decay, evecs, evecs)

    if t.ndim == 0:
        return jax.lax.cond(
            method_code == 3,
            lambda vals: eigh_for_times(vals[0], vals[1][None])[0],
            lambda vals: expm_for_times(vals[0], vals[1]),
            (laplacian, t),
        )

    if t.shape[0] < 2:
        return jax.lax.cond(
            method_code == 3,
            lambda vals: eigh_for_times(vals[0], vals[1]),
            lambda vals: expm_for_times(vals[0], vals[1]),
            (laplacian, t),
        )

    def expm_branch(vals):
        """Expm branch.
        
        Args:
            vals: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        lap, times = vals
        return expm_for_times(lap, times)

    def power_branch(vals):
        """Power branch.
        
        Args:
            vals: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        lap, times = vals
        dt = times[1] - times[0]
        safe_dt = jnp.where(jnp.abs(dt) > 0, dt, jnp.array(1.0, dtype=times.dtype))
        step_ids = jnp.rint(times / safe_dt).astype(jnp.int32)
        step_ids = jnp.clip(step_ids, a_min=0)
        base = jax.scipy.linalg.expm(-lap * dt)
        return _propagators_from_base_binary_powers(base, step_ids)

    def eigh_branch(vals):
        """Eigh branch.
        
        Args:
            vals: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        lap, times = vals
        return eigh_for_times(lap, times)

    def auto_branch(vals):
        """Auto branch.
        
        Args:
            vals: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        lap, times = vals
        dt = times[1] - times[0]
        idx = jnp.arange(times.shape[0], dtype=times.dtype)
        expected = times[0] + dt * idx
        nonzero_dt = jnp.abs(dt) > 1e-12
        is_equispaced = nonzero_dt & jnp.all(jnp.isclose(times, expected, rtol=1e-5, atol=1e-8))
        return jax.lax.cond(is_equispaced, power_branch, expm_branch, (lap, times))

    branch_idx = jnp.clip(jnp.asarray(method_code, dtype=jnp.int32), 0, 3)
    return jax.lax.switch(branch_idx, (auto_branch, expm_branch, power_branch, eigh_branch), (laplacian, t))

@jit
def construct_hermitian_laplacian_from_edge_weights(edge_weights: Float[Array, "n_edges"], edge_ids: Int[Array, "n_edges 2"], pi_stationary: Float[Array, "n_cells"]) -> Float[Array, "n_cells n_cells"]:
    """Construct hermitian laplacian from edge weights.
    
    Args:
        edge_weights: Input parameter.
        edge_ids: Input parameter.
        pi_stationary: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    n_cells = pi_stationary.shape[0]
    adjacency_matrix = jnp.zeros((n_cells, n_cells)).at[edge_ids[:,0], edge_ids[:,1]].set(edge_weights).at[edge_ids[:,1], edge_ids[:,0]].set(edge_weights) # Symmetric adjacency matrix
    degree_matrix = jnp.diag(jnp.sum(adjacency_matrix * jnp.sqrt(pi_stationary)[None,:], axis=-1) / jnp.sqrt(pi_stationary))
    return degree_matrix - adjacency_matrix

def get_edge_ids_from_threshold(kernel, threshold):
    """Return edge ids from threshold.
    
    Args:
        kernel: Input parameter.
        threshold: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    n_cells = kernel.shape[0]
    edge_ids = jnp.nonzero(kernel > threshold)
    upper_triangle_mask = edge_ids[0] < edge_ids[1]
    return jnp.stack(edge_ids, axis=-1)[upper_triangle_mask, :]

def get_edge_ids_from_min_degree(distance_matrix, min_degree):
    """Return edge ids from min degree.
    
    Args:
        distance_matrix: Input parameter.
        min_degree: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    n_cells = distance_matrix.shape[0]
    k_nearest_indices = jnp.argpartition(distance_matrix, min_degree + 1, axis=-1)[:, :min_degree + 1]
    edge_ids = jnp.stack([jnp.repeat(jnp.arange(n_cells), min_degree + 1), k_nearest_indices.ravel()], axis=-1)
    symmetric_edge_ids = jnp.concatenate([edge_ids, edge_ids[:, ::-1]], axis=0)
    upper_triangle_mask = symmetric_edge_ids[:,0] < symmetric_edge_ids[:,1]
    unique_edge_ids = jnp.unique(symmetric_edge_ids[upper_triangle_mask], axis=0)
    return unique_edge_ids


def get_default_solver_dict(rtol_higher_order: float = 0.0, atol_higher_order: float = 1e-8, rtol_first_order: float = 0.0, atol_first_order: float = 1e-6) -> Dict[str, optx.AbstractMinimiser]:
    """Return a diverse set of optimizers for solver comparisons."""
    higher_order_tols = {"rtol": rtol_higher_order, "atol": atol_higher_order, "norm": optx.max_norm}
    first_order_tols = {"rtol": rtol_first_order, "atol": atol_first_order, "norm": optx.max_norm}
    adam_fast = optx.OptaxMinimiser(optax.adam(learning_rate=5e-1), **first_order_tols)
    adam_slow = optx.OptaxMinimiser(optax.adam(learning_rate=1.5e-1), **first_order_tols)
    adabelief = optx.OptaxMinimiser(optax.adabelief(learning_rate=2e-1), **first_order_tols)
    adamw = optx.OptaxMinimiser(optax.adamw(learning_rate=2e-1, weight_decay=1e-6), **first_order_tols)
    adamw_fast = optx.OptaxMinimiser(optax.adamw(learning_rate=4e-1, weight_decay=1e-6), **first_order_tols)
    nadam = optx.OptaxMinimiser(optax.nadam(learning_rate=3e-1), **first_order_tols)
    lion = optx.OptaxMinimiser(optax.lion(learning_rate=5e-2), **first_order_tols)
    bfgs = optx.BFGS(**higher_order_tols)
    lbfgs = optx.LBFGS(**higher_order_tols)
    dfp = optx.DFP(**higher_order_tols)
    ncg_pr = optx.NonlinearCG(**higher_order_tols, method=optx.polak_ribiere)
    ncg_fr = optx.NonlinearCG(**higher_order_tols, method=optx.fletcher_reeves)
    ncg_hs = optx.NonlinearCG(**higher_order_tols, method=optx.hestenes_stiefel)
    ncg_dy = optx.NonlinearCG(**higher_order_tols, method=optx.dai_yuan)

    return {
        # "dfp": dfp,
        # "ncg_fr": ncg_fr,
        # "ncg_hs": ncg_hs,
        # "ncg_dy": ncg_dy,
        # "ncg_pr": ncg_pr,
        # "bfgs": bfgs,
        # "lbfgs": lbfgs,
        "adam_fast": adam_fast,
        # "adam_slow": adam_slow,
        # "adabelief": adabelief,
        # "adamw": adamw,
        "adamw_fast": adamw_fast,
        "nadam": nadam,
        # "lion": lion,
    }


def compare_solvers(
    data_propagators: Float[Array, "n_times n_cells n_cells"],
    times: Float[Array, "n_times"],
    pi_stationary: Float[Array, "n_cells"],
    kernel: Float[Array, "n_cells n_cells"],
    threshold: Float = 0.1,
    n_iterations: int = 500,
    l1_reg: Float = 0.0,
    l2_reg: Float = 0.0,
    solvers: Dict[str, optx.AbstractMinimiser] | None = None,
    verbose: bool = True,
    return_diagnostics: bool = False,
    propagator_method: str = "auto",
):
    """Compare optimizer convergence and runtime for `fit_best_model`."""
    assert data_propagators.ndim == 3, f"Expected data_propagators to have shape (n_times, n_cells, n_cells), got {data_propagators.shape}"
    assert times.ndim == 1, f"Expected times to be 1D, got shape {times.shape}"
    assert data_propagators.shape[0] == times.shape[0], "times length must match first axis of data_propagators"
    assert data_propagators.shape[1] == data_propagators.shape[2], "data_propagators must be square on the last two axes"
    assert data_propagators.shape[1] == pi_stationary.shape[0], "pi_stationary length must match n_cells"
    assert l1_reg >= 0, "l1_reg must be non-negative"
    assert l2_reg >= 0, "l2_reg must be non-negative"
    assert n_iterations > 0, "n_iterations must be positive"

    if solvers is None:
        solvers = get_default_solver_dict()
    propagator_method_code = propagator_method_to_code(propagator_method)

    init_time_id = min(20, data_propagators.shape[0] - 1)
    init_laplacian = guesstimate_hermitian_laplacian_from_propagator(
        data_propagators[init_time_id],
        times[init_time_id],
        pi_stationary,
    )
    edge_ids = get_edge_ids_from_threshold(kernel, threshold)
    assert edge_ids.shape[0] > 0, "No edges selected. Increase threshold or use a denser kernel."

    eps = jnp.finfo(data_propagators.dtype).eps
    init_edge_weights = jnp.clip(-init_laplacian[edge_ids[:, 0], edge_ids[:, 1]], a_min=eps)
    y0 = edge_weights_to_raw(init_edge_weights, eps=float(eps))

    def loss_with_aux(raw_edge_weights, args=None):
        """Loss with aux.
        
        Args:
            raw_edge_weights: Input parameter.
            args: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        _ = args
        edge_weights = raw_to_edge_weights(raw_edge_weights, eps=float(eps))
        laplacian = construct_hermitian_laplacian_from_edge_weights(edge_weights, edge_ids, pi_stationary)
        data_loss = mmd_loss(laplacian, data_propagators, times, pi_stationary, kernel, propagator_method_code=propagator_method_code)
        l1_penalty = l1_reg * jnp.mean(jnp.abs(edge_weights))
        l2_penalty = l2_reg * jnp.mean(edge_weights ** 2)
        return data_loss + l1_penalty + l2_penalty, None

    def scalar_loss(raw_edge_weights):
        """Scalar loss.
        
        Args:
            raw_edge_weights: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        return loss_with_aux(raw_edge_weights, None)[0]

    def print_nan_diagnostics(raw_edge_weights, solver_name: str, step_id: int):
        """Print nan diagnostics.
        
        Args:
            raw_edge_weights: Input parameter.
            solver_name: Input parameter.
            step_id: Input parameter.
        """
        edge_weights = raw_to_edge_weights(raw_edge_weights, eps=float(eps))
        laplacian = construct_hermitian_laplacian_from_edge_weights(edge_weights, edge_ids, pi_stationary)
        data_loss = mmd_loss(laplacian, data_propagators, times, pi_stationary, kernel, propagator_method_code=propagator_method_code)
        l1_penalty = l1_reg * jnp.mean(jnp.abs(edge_weights))
        l2_penalty = l2_reg * jnp.mean(edge_weights ** 2)
        total_loss = data_loss + l1_penalty + l2_penalty
        print(
            f"{solver_name}: non-finite loss at iter={step_id}; "
            f"loss={float(total_loss):.3e}, data={float(data_loss):.3e}, l1={float(l1_penalty):.3e}, l2={float(l2_penalty):.3e}"
        )
        print(
            f"{solver_name}: raw_w[min,max]=({float(jnp.min(raw_edge_weights)):.3e}, {float(jnp.max(raw_edge_weights)):.3e}), "
            f"w[min,max]=({float(jnp.min(edge_weights)):.3e}, {float(jnp.max(edge_weights)):.3e})"
        )
        print(
            f"{solver_name}: finite checks -> raw_w={bool(jnp.all(jnp.isfinite(raw_edge_weights)))}, "
            f"w={bool(jnp.all(jnp.isfinite(edge_weights)))}, laplacian={bool(jnp.all(jnp.isfinite(laplacian)))}, "
            f"data_prop={bool(jnp.all(jnp.isfinite(data_propagators)))}, pi={bool(jnp.all(jnp.isfinite(pi_stationary)))}, "
            f"kernel={bool(jnp.all(jnp.isfinite(kernel)))}"
        )

    fn = filter_jit(loss_with_aux)
    default_stuff = {"args": None, "options": {}, "tags": frozenset()}

    losses: Dict[str, Array] = {}
    times_ms: Dict[str, float] = {}
    grad_norms: Dict[str, Array] = {}

    for name, solver in tqdm(solvers.items()):
        step = eqx.filter_jit(eqx.Partial(solver.step, fn=fn, **default_stuff))
        terminate = eqx.filter_jit(eqx.Partial(solver.terminate, fn=fn, **default_stuff))

        y = y0
        state = solver.init(
            fn,
            y,
            **default_stuff,
            f_struct=jax.ShapeDtypeStruct((), jnp.float32),
            aux_struct=None,
        )
        done, result = terminate(y=y, state=state)
        aux = None

        loss_arr = jnp.full(n_iterations, jnp.nan)
        grad_norm_arr = jnp.full(n_iterations, jnp.nan)
        for i in range(n_iterations):
            if done:
                if verbose:
                    finite_losses = loss_arr[jnp.isfinite(loss_arr)]
                    last_loss = float(finite_losses[-1]) if finite_losses.size > 0 else float("nan")
                    print(f"{name}: done after {i} steps with status {result}, last_loss={last_loss:.3e}")
                break
            current_loss = fn(y, None)[0]
            loss_arr = loss_arr.at[i].set(current_loss)
            if not bool(jnp.isfinite(current_loss)):
                print_nan_diagnostics(y, name, i)
                done = True
                break
            grad_norm = float(jnp.linalg.norm(jax.grad(scalar_loss)(y)))
            grad_norm_arr = grad_norm_arr.at[i].set(grad_norm)
            if verbose and (i == 0 or (i + 1) % 25 == 0):
                print(f"{name}: iter={i}, loss={float(current_loss):.3e}, grad_norm={grad_norm:.3e}")
            y, state, aux = step(y=y, state=state)
            done, result = terminate(y=y, state=state)

        _ = solver.postprocess(fn, y, aux=aux, **default_stuff, state=state, result=result)

        @eqx.filter_jit
        def run(y_init):
            """Run.
            
            Args:
                y_init: Input parameter.
            
            Returns:
                Output value computed by this function.
            """
            return optx.minimise(
                fn=fn,
                y0=y_init,
                solver=solver,
                max_steps=n_iterations,
                has_aux=True,
                throw=False,
            ).value

        run(y0)
        times_ms[name] = timeit(stmt=lambda: run(y0).block_until_ready(), number=1) / 1 * 1e3

        if verbose and result != optx.RESULTS.successful:
            print(f"{name}: finished with status {result}")

        losses[name] = loss_arr
        grad_norms[name] = grad_norm_arr

    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    styles = ["-", "--", "-.", ":"]
    for i, (name, loss_arr) in enumerate(losses.items()):
        ax.plot(loss_arr, label=f"{name} ({times_ms[name]:.2g}ms)", ls=styles[i % len(styles)])
    ax.set_ylabel("loss")
    ax.set_xlabel("iteration")
    ax.set_yscale("log")
    ax.legend()

    if return_diagnostics:
        return losses, times_ms, grad_norms
    return losses, times_ms


def fit_best_model(
    data_propagators: Float[Array, "n_times n_cells n_cells"],
    times: Float[Array, "n_times"],
    pi_stationary: Float[Array, "n_cells"],
    kernel: Float[Array, "n_cells n_cells"],
    threshold: Float = 0.1,
    n_iterations: int = 2**9,
    l1_reg: Float = 0.0,
    l2_reg: Float = 0.0,
    solver: optx.AbstractMinimiser | None = None,
    verbose: bool = True,
    propagator_method: str = "auto",
) -> Float[Array, "n_cells n_cells"]:
    """Fit best model.
    
    Args:
        data_propagators: Input parameter.
        times: Input parameter.
        pi_stationary: Input parameter.
        kernel: Input parameter.
        threshold: Input parameter.
        n_iterations: Input parameter.
        l1_reg: Input parameter.
        l2_reg: Input parameter.
        solver: Input parameter.
        verbose: Input parameter.
        propagator_method: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    assert data_propagators.ndim == 3, f"Expected data_propagators to have shape (n_times, n_cells, n_cells), got {data_propagators.shape}"
    assert times.ndim == 1, f"Expected times to be 1D, got shape {times.shape}"
    assert data_propagators.shape[0] == times.shape[0], "times length must match first axis of data_propagators"
    assert data_propagators.shape[1] == data_propagators.shape[2], "data_propagators must be square on the last two axes"
    assert data_propagators.shape[1] == pi_stationary.shape[0], "pi_stationary length must match n_cells"
    assert l1_reg >= 0, "l1_reg must be non-negative"
    assert l2_reg >= 0, "l2_reg must be non-negative"
    assert n_iterations > 0, "n_iterations must be positive"
    propagator_method_code = propagator_method_to_code(propagator_method)

    if solver is None:
        base_solver = optx.OptaxMinimiser(
            optax.adam(learning_rate=2.5e-1),
            rtol=0,
            atol=1e-2,
            norm=optx.max_norm,
        )
        solver = optx.BestSoFarMinimiser(base_solver)

    # Non-jittable preprocessing (dynamic edge extraction from threshold).
    init_time_id = min(20, data_propagators.shape[0] - 1)
    init_laplacian = guesstimate_hermitian_laplacian_from_propagator(
        data_propagators[init_time_id],
        times[init_time_id],
        pi_stationary,
    )
    edge_ids = get_edge_ids_from_threshold(kernel, threshold)
    assert edge_ids.shape[0] > 0, "No edges selected. Increase threshold or use a denser kernel."

    eps = jnp.finfo(data_propagators.dtype).eps
    init_edge_weights = jnp.clip(-init_laplacian[edge_ids[:, 0], edge_ids[:, 1]], a_min=eps)
    init_model = construct_hermitian_laplacian_from_edge_weights(init_edge_weights, edge_ids, pi_stationary)
    loss_init = mmd_loss(init_model, data_propagators, times, pi_stationary, kernel, propagator_method_code=propagator_method_code)
    if verbose:
        print(f"Initial loss: {loss_init:.2e}")

    @filter_jit
    def loss_with_aux(raw_edge_weights, args=None):
        """Loss with aux.
        
        Args:
            raw_edge_weights: Input parameter.
            args: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        _ = args
        edge_weights = raw_to_edge_weights(raw_edge_weights, eps=float(eps))
        laplacian = construct_hermitian_laplacian_from_edge_weights(edge_weights, edge_ids, pi_stationary)
        data_loss = mmd_loss(laplacian, data_propagators, times, pi_stationary, kernel, propagator_method_code=propagator_method_code)
        l1_penalty = l1_reg * jnp.mean(jnp.abs(edge_weights))
        l2_penalty = l2_reg * jnp.mean(edge_weights ** 2)
        return data_loss + l1_penalty + l2_penalty, None

    def loss_fn(raw_edge_weights):
        """Loss fn.
        
        Args:
            raw_edge_weights: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        return loss_with_aux(raw_edge_weights, None)[0]

    def print_nan_diagnostics(raw_edge_weights):
        """Print nan diagnostics.
        
        Args:
            raw_edge_weights: Input parameter.
        """
        edge_weights = raw_to_edge_weights(raw_edge_weights, eps=float(eps))
        laplacian = construct_hermitian_laplacian_from_edge_weights(edge_weights, edge_ids, pi_stationary)
        data_loss = mmd_loss(laplacian, data_propagators, times, pi_stationary, kernel, propagator_method_code=propagator_method_code)
        l1_penalty = l1_reg * jnp.mean(jnp.abs(edge_weights))
        l2_penalty = l2_reg * jnp.mean(edge_weights ** 2)
        total_loss = data_loss + l1_penalty + l2_penalty
        print(
            f"fit_best_model: non-finite loss; loss={float(total_loss):.3e}, "
            f"data={float(data_loss):.3e}, l1={float(l1_penalty):.3e}, l2={float(l2_penalty):.3e}"
        )
        print(
            f"fit_best_model: raw_w[min,max]=({float(jnp.min(raw_edge_weights)):.3e}, {float(jnp.max(raw_edge_weights)):.3e}), "
            f"w[min,max]=({float(jnp.min(edge_weights)):.3e}, {float(jnp.max(edge_weights)):.3e})"
        )
        print(
            f"fit_best_model: finite checks -> raw_w={bool(jnp.all(jnp.isfinite(raw_edge_weights)))}, "
            f"w={bool(jnp.all(jnp.isfinite(edge_weights)))}, laplacian={bool(jnp.all(jnp.isfinite(laplacian)))}, "
            f"data_prop={bool(jnp.all(jnp.isfinite(data_propagators)))}, pi={bool(jnp.all(jnp.isfinite(pi_stationary)))}, "
            f"kernel={bool(jnp.all(jnp.isfinite(kernel)))}"
        )

    @eqx.filter_jit
    def run_optimisation(y_init):
        """Run optimisation.
        
        Args:
            y_init: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        return optx.minimise(
            fn=loss_with_aux,
            y0=y_init,
            solver=solver,
            max_steps=n_iterations,
            has_aux=True,
            throw=False,
        )

    y0 = edge_weights_to_raw(init_edge_weights, eps=float(eps))
    sol = run_optimisation(y0)
    if not bool(jnp.isfinite(loss_fn(sol.value))):
        print_nan_diagnostics(sol.value)
    fitted_edge_weights = raw_to_edge_weights(sol.value, eps=float(eps))
    if verbose:
        print(f"Final loss: {loss_fn(sol.value):.2e} after {sol.stats} iterations with status {sol.result}")

    return construct_hermitian_laplacian_from_edge_weights(fitted_edge_weights, edge_ids, pi_stationary)



def fit_best_model_with_pi_stat(
    data_propagators: Float[Array, "n_times n_cells n_cells"],
    times: Float[Array, "n_times"],
    pi_stat_data: Float[Array, "n_cells"],
    kernel: Float[Array, "n_cells n_cells"],
    threshold: Float = 0.1,
    n_iterations: int = 500,
    l1_reg: Float = 0.0,
    l2_reg: Float = 0.0,
    solver: optx.AbstractMinimiser = optx.BFGS(rtol=0, atol=1e-2),
    propagator_method: str = "auto",
) -> Tuple[Float[Array, "n_cells n_cells"], Float[Array, "n_cells"]]:
    """Fit best model with pi stat.
    
    Args:
        data_propagators: Input parameter.
        times: Input parameter.
        pi_stat_data: Input parameter.
        kernel: Input parameter.
        threshold: Input parameter.
        n_iterations: Input parameter.
        l1_reg: Input parameter.
        l2_reg: Input parameter.
        solver: Input parameter.
        propagator_method: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    assert data_propagators.ndim == 3, f"Expected data_propagators to have shape (n_times, n_cells, n_cells), got {data_propagators.shape}"
    assert times.ndim == 1, f"Expected times to be 1D, got shape {times.shape}"
    assert data_propagators.shape[0] == times.shape[0], "times length must match first axis of data_propagators"
    assert data_propagators.shape[1] == data_propagators.shape[2], "data_propagators must be square on the last two axes"
    assert data_propagators.shape[1] == pi_stat_data.shape[0], "pi_stat_data length must match n_cells"
    assert l1_reg >= 0, "l1_reg must be non-negative"
    assert l2_reg >= 0, "l2_reg must be non-negative"
    propagator_method_code = propagator_method_to_code(propagator_method)

    init_time_id = min(20, data_propagators.shape[0] - 1)
    init_laplacian = guesstimate_hermitian_laplacian_from_propagator(
        data_propagators[init_time_id],
        times[init_time_id],
        pi_stat_data,
    )
    edge_ids = get_edge_ids_from_threshold(kernel, threshold)
    assert edge_ids.shape[0] > 0, "No edges selected. Increase threshold or use a denser kernel."

    eps = jnp.finfo(data_propagators.dtype).eps
    init_edge_weights = jnp.clip(-init_laplacian[edge_ids[:, 0], edge_ids[:, 1]], a_min=eps)
    init_model = construct_hermitian_laplacian_from_edge_weights(init_edge_weights, edge_ids, pi_stat_data)
    loss_init = mmd_loss(init_model, data_propagators, times, pi_stat_data, kernel, propagator_method_code=propagator_method_code)
    print(f"Initial loss: {loss_init:.2e}")

    def loss_fn(y, args=None):
        """Loss fn.
        
        Args:
            y: Input parameter.
            args: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        _ = args
        raw_edge_weights, log_pi_stationary = y
        pi_stationary = jnp.exp(log_pi_stationary) / jnp.sum(jnp.exp(log_pi_stationary))
        edge_weights = raw_to_edge_weights(raw_edge_weights, eps=float(eps))
        laplacian = construct_hermitian_laplacian_from_edge_weights(edge_weights, edge_ids, pi_stationary)
        data_loss = mmd_loss(
            laplacian,
            data_propagators,
            times,
            pi_stationary,
            kernel,
            data_pi_stationary=pi_stat_data,
            propagator_method_code=propagator_method_code,
        )
        l1_penalty = l1_reg * jnp.mean(jnp.abs(edge_weights))
        l2_penalty = l2_reg * jnp.mean(edge_weights ** 2)
        return data_loss + l1_penalty + l2_penalty

    y0 = edge_weights_to_raw(init_edge_weights, eps=float(eps)), jnp.log(pi_stat_data)
    sol = optx.minimise(fn=loss_fn, y0=y0, solver=solver, max_steps=n_iterations, throw=False)
    fitted_edge_weights = raw_to_edge_weights(sol.value[0], eps=float(eps))
    fitted_pi_stationary = jnp.exp(sol.value[1]) / jnp.sum(jnp.exp(sol.value[1]))
    print(f"Final loss: {loss_fn(sol.value):.2e} after {sol.stats} iterations with status {optx.RESULTS[sol.result]}")

    return construct_hermitian_laplacian_from_edge_weights(fitted_edge_weights, edge_ids, fitted_pi_stationary), fitted_pi_stationary



