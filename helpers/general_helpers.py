import numpy as np
from numba import njit, prange
from typing import Tuple
from jax import jit
from functools import partial
from equinox import filter_jit, filter_vmap
from jaxtyping import Array, Float, Int
import jax.numpy as jnp


@njit
def rle_triplets(chunk):
    """Rle triplets.
    
    Args:
        chunk: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    n_p, n_t = chunk.shape
    max_events = (n_p * n_t) // 20 + 1  # Conservative estimate
    ijks = np.empty((max_events, 3), dtype=chunk.dtype)
    ts = np.empty((max_events,), dtype=np.uint16)
    
    event_idx = 0
    for p in range(n_p):
        # i -> j -> k state machine
        c_i = -1 # Previous cell
        c_j = chunk[p, 0] # Current cell
        t_start = 0 # When we entered cell j
        
        for t in range(1, n_t):
            c_current = chunk[p, t]
            
            if c_current != c_j:
                # Transition detected!
                if c_i != -1:
                    # We have a full i -> j -> k triplet
                    # Store [i, j, k, duration]
                    if event_idx < max_events:
                        ijks[event_idx, 0] = c_i
                        ijks[event_idx, 1] = c_j
                        ijks[event_idx, 2] = c_current
                        ts[event_idx] = t - t_start
                        event_idx += 1
                
                # Shift the window
                c_i = c_j
                c_j = c_current
                t_start = t
    return ijks[:event_idx], ts[:event_idx]

@njit(parallel=False)
def update_transition_array(traj_disc_chunk, lookup, transition_array):
    """Update transition array.
    
    Args:
        traj_disc_chunk: Input parameter.
        lookup: Input parameter.
        transition_array: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    n_p, n_t = traj_disc_chunk.shape
    t_max = transition_array.shape[3]

    # We use prange to distribute particles across cores
    for p in prange(n_p):
        c_prev = -1
        c_curr = traj_disc_chunk[p, 0]
        t_start = 0
        
        for t in range(1, n_t):
            c_next = traj_disc_chunk[p, t-1]
            if c_next != c_curr:
                if c_prev != -1:
                    i_idx = lookup[c_curr, c_prev]
                    k_idx = lookup[c_curr, c_next]
                    duration = min(t - t_start, t_max)

                    transition_array[c_curr, i_idx, k_idx, duration-1] += 1
                
                c_prev = c_curr
                c_curr = c_next
                t_start = t
                
    return transition_array

def init_triplet_array(num_cells, max_neighbors, t_max=2**12):
    """Init triplet array.
    
    Args:
        num_cells: Input parameter.
        max_neighbors: Input parameter.
        t_max: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    return np.zeros((num_cells, max_neighbors + 1, max_neighbors + 1, t_max), dtype=np.uint32)

@njit
def create_lookup(neighbors):
    """Create lookup.
    
    Args:
        neighbors: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    num_cells, max_neighbors = neighbors.shape
    
    # Initialize with the padding index (max_neighbors)
    lookup = np.full((num_cells, num_cells), max_neighbors, dtype=neighbors.dtype)
    for c in range(num_cells):
        for idx, nb in enumerate(neighbors[c]):
            if nb < num_cells: # ignore padding
                lookup[c, nb] = idx
    return lookup


def get_chunks_and_shards(shape: Tuple, chunking_axis: int = 0, max_chunk_mbytes: int = 2**4, max_shard_mbytes: int = 2**12, itemsize: int = 1, round_to_power_of_2: bool = True) -> Tuple[Tuple, Tuple]:
    """Get chunk shape for dask array given a maximum chunk size along a specific axis."""
    slice_mbyte_size = np.prod(shape) // shape[chunking_axis] * itemsize / 2**20
    assert slice_mbyte_size > 0, ValueError("Slice byte size is zero, cannot compute chunk shape.")
    assert max_chunk_mbytes >= slice_mbyte_size, ValueError("Max chunk bytes is smaller than slice byte size, cannot compute chunk shape.")
    
    max_chunk_size = max_chunk_mbytes // slice_mbyte_size
    if round_to_power_of_2:
        max_chunk_size = 2 ** int(np.log2(max_chunk_size))
    
    chunk_shape = list(shape)
    chunk_shape[chunking_axis] = min(shape[chunking_axis], max_chunk_size)

    num_chunks_per_shard = max_shard_mbytes // (slice_mbyte_size * max_chunk_size)
    if round_to_power_of_2:
        num_chunks_per_shard = int(2 ** int(np.log2(num_chunks_per_shard)))
    
    num_chunks_per_shard = max(1, num_chunks_per_shard)
    max_chunks_per_shard = min(num_chunks_per_shard, shape[chunking_axis] // chunk_shape[chunking_axis])
    shard_shape = list(shape)
    shard_shape[chunking_axis] = max_chunks_per_shard * chunk_shape[chunking_axis]

    return tuple(chunk_shape), tuple(shard_shape)

@partial(filter_jit, donate="all")
def update_propagator_aggregators(chunk, aggregators, tmax=2**12, delta_t=2**4):
    """Update propagator aggregators.
    
    Args:
        chunk: Input parameter.
        aggregators: Input parameter.
        tmax: Input parameter.
        delta_t: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    n_cells = aggregators[0].shape[1]
    
    subview = chunk[:, ::delta_t]
    n_ts = tmax // delta_t + 1
    def id(c_0, c_1, delta_t):
        """Id.
        
        Args:
            c_0: Input parameter.
            c_1: Input parameter.
            delta_t: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        return (delta_t) * n_cells * n_cells + (c_0 * n_cells + c_1)

    t_len = subview.shape[1] - 1
    t0s_full = jnp.repeat(jnp.arange(t_len, dtype=jnp.int32), n_ts)
    tdeltas_full = jnp.tile(jnp.arange(n_ts, dtype=jnp.int32), t_len)
    mask = t0s_full + tdeltas_full < t_len

    m = min(n_ts, t_len)
    n_pairs = m * t_len - (m * (m - 1)) // 2
    valid_idx = jnp.where(mask, size=n_pairs, fill_value=0)[0]
    t0s = t0s_full[valid_idx]
    tdeltas = tdeltas_full[valid_idx]
    c0s, c1s = subview[:, t0s].astype(jnp.int32), subview[:, t0s + tdeltas].astype(jnp.int32)
    ids = id(c0s, c1s, tdeltas[None, :])
    
    counts = jnp.bincount(ids.ravel(), length=n_cells*n_cells*n_ts).reshape((n_ts, n_cells, n_cells))
    propagator = counts / jnp.sum(counts[:,:,:], axis=-1, keepdims=True)

    agg_sum, agg_sq_sum = aggregators
    aggregators = (agg_sum + propagator, agg_sq_sum + propagator ** 2)

    return aggregators

@partial(filter_jit, donate="all")
def update_mfpt_aggregators(chunk: Int[Array, "n_p n_t"], aggregators) -> Tuple[Float[Array, "n_cells n_cells"], ...]:
    
    """Update mfpt aggregators.
    
    Args:
        chunk: Input parameter.
        aggregators: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    n_cells = aggregators[0].shape[0]

    @filter_vmap
    def get_fpt(traj):
        """Return fpt.
        
        Args:
            traj: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        first_indices = jnp.full((n_cells,), traj.shape[-1], dtype=jnp.int32)
        # Create an array of actual indices [0, 1, 2, ..., N-1]
        indices = jnp.arange(traj.shape[-1], dtype=jnp.int32)
        
        # Atomic scatter-min: for each value in x, store the minimum index found
        # This is the "magic" O(N) step on GPU
        first_indices = first_indices.at[traj].min(indices)
        terminated = first_indices < traj.shape[-1]
        return first_indices, terminated
    
    fpts, terminations = get_fpt(chunk)
    in_masks = (chunk[None, :, 0] == jnp.arange(n_cells)[:,None]) # Only consider trajectories that change state

    termination_counts = jnp.sum(in_masks[:,:,None] * terminations[None,:,:], axis=1)
    mfpt = jnp.sum(in_masks[:,:,None] * fpts[None,:,:], axis=1) / jnp.maximum(termination_counts, 1)

    agg_sum, agg_sq_sum, agg_term = aggregators

    aggregators = (agg_sum + mfpt, agg_sq_sum + mfpt ** 2, agg_term + termination_counts/jnp.max(termination_counts, axis=-1, keepdims=True))
    
    return aggregators


@partial(filter_jit, donate="all")
def update_aggregators(
    chunk: Int[Array, "n_p n_t"],
    prop_aggs,
    mfpt_aggs,
    stationary_aggs,
    tmax: int = 2**12,
    delta_t: int = 2**4
):
    """Update aggregators.
    
    Args:
        chunk: Input parameter.
        prop_aggs: Input parameter.
        mfpt_aggs: Input parameter.
        stationary_aggs: Input parameter.
        tmax: Input parameter.
        delta_t: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    prop_aggs = update_propagator_aggregators(chunk, prop_aggs, tmax=tmax, delta_t=delta_t)
    mfpt_aggs = update_mfpt_aggregators(chunk, mfpt_aggs)

    n_cells = stationary_aggs[0].shape[0]

    counts = jnp.bincount(chunk.ravel(), length=n_cells)
    total = jnp.sum(counts)
    stationary = counts / total
    stationary_aggs = (stationary_aggs[0] + stationary, stationary_aggs[1] + stationary ** 2)

    return prop_aggs, mfpt_aggs, stationary_aggs


    

    
    tuple_ids, times = get_tuple_ids(chunk)

    # sum_times 

