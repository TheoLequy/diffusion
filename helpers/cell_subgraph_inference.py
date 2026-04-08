import os, sys
sys.path.append(os.path.abspath(".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import networkx as nx
import optax
import optimistix as optx
import jax 
from jax import jit, vmap
import equinox as eqx
from equinox import filter_jit
import jax.numpy as jnp
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from helpers.plotting_graphs import (
    _parse_vertex_label,
    _get_vertex_position,
    _darken_color,
    _lighten_color,
    _round_sig,
    plot_subgraph,
    plot_cell_subgraph,
    plot_cell_subgraph_interactive,
)
from helpers.plotting_fields import COLORS
from helpers.geometry_helpers import Tesselation
from functools import partial
from jaxtyping import Array, Float, Int, Bool, UInt, PyTree
from typing import Tuple, List, Sequence, Iterable, Collection, Dict, Callable, Any
from tqdm.notebook import tqdm
from timeit import timeit
import numpy as np
from IPython.display import display_html, display
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

TTTree = PyTree[Array, "TransitionTripleTree"] # (t, in_vert, out_vert, count, neighbor_ids)

STABILITY_CONSTANT = jnp.finfo(jnp.float32).smallest_normal

def data_dict_from_df(df: pd.DataFrame, neighbor_ids: Array, id_dtype=jnp.uint8, verbose=True) -> Dict[str, Array]:
    """
    Convert a DataFrame to a JAX array of shape (n_rows, 4) containing the neighbor IDs.
    """
    cols = ["i", "k", "t", "count"]
    assert all(col in df.columns for col in cols), f"DataFrame must contain columns: {cols}"

    count_tot = df["count"].sum()
    count_dtype = jnp.uint32 if count_tot < 2**32 else jnp.uint64
    tmax = df["t"].max()
    t_type = jnp.int16 if tmax < 2**16 else jnp.int32 if tmax < 2**32 else jnp.float32

    true_neighbors_map = pd.Series(jnp.arange( len(neighbor_ids)), index=neighbor_ids, name="neighbor_id") # type: ignore

    df["in_vert"] = df["i"].map(true_neighbors_map)
    df["out_vert"] = df["k"].map(true_neighbors_map)
    if verbose:
        nan_frac = (df[["in_vert", "out_vert"]].isna().any(axis=1) * df["count"]).sum() / count_tot
        if nan_frac > 0:
            print(f"Warning: dropped {nan_frac:.1g} of events to cells not in the given neighbor list.")
    df = df.dropna(subset=["in_vert", "out_vert"])

    data_dict = {"in_vert": jnp.array(df["i"].map(true_neighbors_map), dtype=id_dtype), "out_vert": jnp.array(df["k"].map(true_neighbors_map), dtype=id_dtype), 
            "t": jnp.array(df["t"], dtype=t_type), "count": jnp.array(df["count"], dtype=count_dtype), "neighbor_ids": jnp.asarray(neighbor_ids, dtype=id_dtype)}
    return data_dict

def data_arr_from_df(df: pd.DataFrame, neighbor_ids: Array, id_dtype=jnp.uint8, verbose=True) -> Array:
    """Data arr from df.
    
    Args:
        df: Input parameter.
        neighbor_ids: Input parameter.
        id_dtype: Input parameter.
        verbose: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    cols = ["i", "k", "t", "count"]
    assert all(col in df.columns for col in cols), f"DataFrame must contain columns: {cols}"

    count_max = df["count"].max()
    count_tot = df["count"].sum()
    count_dtype = jnp.uint16 if count_max < 2**16 else jnp.uint32 if count_max < 2**32 else jnp.uint64
    t_max = df["t"].max()

    true_neighbors_map = pd.Series(jnp.arange( len(neighbor_ids)), index=neighbor_ids, name="neighbor_id") # type: ignore

    df["in_vert"] = df["i"].map(true_neighbors_map)
    df["out_vert"] = df["k"].map(true_neighbors_map)
    if verbose:
        nan_frac = (df[["in_vert", "out_vert"]].isna().any(axis=1) * df["count"]).sum() / count_tot
        if nan_frac > 0:
            print(f"Warning: dropped {nan_frac:.1g} of events to cells not in the given neighbor list.")

    counts = df.dropna(subset=["in_vert", "out_vert"]).set_index(["in_vert", "out_vert", "t"])["count"] 

    neighbor_index = pd.Index(range(len(neighbor_ids)), name="neighbor_id")
    time_index = pd.Index(range(1, int(t_max)+1), name="t")
    counts = counts.reindex(pd.MultiIndex.from_product([neighbor_index, neighbor_index, time_index], names=["in_vert", "out_vert", "t"]), fill_value=0)
    
    return jnp.asarray(counts, dtype=count_dtype).reshape(len(neighbor_ids), len(neighbor_ids), -1)

def get_sample_cell_df(file_name: str = "data/cell_6.parquet",  neighbor_ids: Array = jnp.array([1,4,5,8,10,11,26], dtype=jnp.uint8), verbose=True) -> pd.DataFrame:
    
    """Return sample cell df.
    
    Args:
        file_name: Input parameter.
        neighbor_ids: Input parameter.
        verbose: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    fmt = file_name.split(".")[-1]
    if fmt == "parquet":
        df = pd.read_parquet(file_name)
    elif fmt == "csv":
        df = pd.read_csv(file_name)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")
    assert df is not None

    cols = ["i", "k", "t", "count"]
    assert all(col in df.columns for col in cols), f"DataFrame must contain columns: {cols}"

    true_neighbors_map = pd.Series(jnp.arange(len(neighbor_ids), dtype=jnp.uint8), index=neighbor_ids, name="neighbor_id") # type: ignore

    df["in_vert"] = df["i"].map(true_neighbors_map)
    df["out_vert"] = df["k"].map(true_neighbors_map)

    if verbose:
        count_tot = df["count"].sum()
        nan_frac = (df[["in_vert", "out_vert"]].isna().any(axis=1) * df["count"]).sum() / count_tot
        if nan_frac > 0:
            print(f"Warning: dropped {nan_frac:.1g} of events to cells not in the given neighbor list.")
    df = df.dropna(subset=["in_vert", "out_vert"])
    df[["in_vert", "out_vert"]] = df[["in_vert", "out_vert"]].astype("uint8")
    return df


def display_rates(r, neighbor_ids: None | Array = None):
    """Display rates.
    
    Args:
        r: Input parameter.
        neighbor_ids: Input parameter.
    """
    num_neighbors = r.shape[0] - 1
    if neighbor_ids is None:
        neighbor_ids = jnp.arange(num_neighbors)
    assert len(neighbor_ids) == num_neighbors, f"Expected {num_neighbors} neighbor IDs, got {len(neighbor_ids)}"

    verts =     ["c"] + neighbor_ids.tolist()
    df = pd.DataFrame(r, columns=pd.Series(verts, name="out_vert"), index=pd.Series(verts, name="in_vert"), dtype=float)
    df["lambda"] =  df.fillna(0.).sum(axis=1)
    display(df.style.format("{:.2f}").background_gradient(cmap="Reds", vmin=0, axis=None, subset=(list(neighbor_ids), verts)) # type: ignore
            .background_gradient(cmap="Blues", vmin=0, axis=None, subset=["lambda"])
            .background_gradient(cmap="Oranges", vmin=0, axis=None, subset=(["c"], list(neighbor_ids))).set_caption("Transition Rates")) # type: ignore
    

@filter_jit
def p_model_one_center(t: Float, in_vert: UInt =0, out_vert: UInt =1, r: Float[Array, "n+1 n+1"]=(20 * jnp.eye(8) + 0.1 * jnp.ones((8,8))), eps=STABILITY_CONSTANT) -> Float:
    """
    probability density of transition from in_vert to out_vert at time t, given transition rates r[i,j] from vertex i to vertex j. 
    r is a (n+1, n+1) array where the first row and column correspond to the center vertex. 
    The probability is computed using the formula derived from the master equation of the Markov process.
    This function broadcasts jointly over in_vert, out_vert, t so they can be arrays of compatible shapes.
    """
    # shift neighbor indices (0..n-1) to (1..n) to account for center at index 0
    in_vert += 1
    out_vert += 1

    # total decay rates for each vertex (including center)
    l_in = jnp.sum(r, axis=-1)
    l_center = jnp.sum(r[0, 1:])
    
    coef1 = r[0, out_vert] * r[in_vert, 0] / (jnp.abs(l_in[in_vert] - l_center) + eps)
    expc = jnp.exp(-l_center * t)
    expin = jnp.exp(-l_in[in_vert] * t)
    coef2 = r[in_vert, out_vert]
    return coef1 * (expc - expin) + coef2 * expin


@filter_jit
def p_remain_model_one_center(t: Float, in_vert: UInt =0, r: Float[Array, "n+1 n+1"]=(20 * jnp.eye(8) + 0.1 * jnp.ones((8,8))), eps=STABILITY_CONSTANT) -> Float:
    """
    probability of remaining in the system (not yet transitioned) from in_vert at time t, given transition rates r[i,j] from vertex i to vertex j. 
    r is a (n+1, n+1) array where the first row and column correspond to the center vertex. 
    The probability is computed using the formula derived from the master equation of the Markov process.
    This function broadcasts jointly over in_vert, t so they can be arrays of compatible shapes.
    """
    # shift neighbor indices (0..n-1) to (1..n) to account for center at index 0
    in_vert += 1

    # total decay rates for each vertex (including center)
    l_in = jnp.sum(r, axis=-1)
    l_center = jnp.sum(r[0, 1:])
    
    coef1 = l_center * r[in_vert, 0] / (jnp.abs(l_in[in_vert] - l_center) + eps)
    expc = jnp.exp(-l_center * t) / l_center
    expin = jnp.exp(-l_in[in_vert] * t) / l_in[in_vert]
    coef2 = l_in[in_vert] - r[in_vert, 0]

    return coef1 * (expc - expin) + coef2 * expin


@filter_jit
def transition_rate_model_one_center(t: Float, in_vert: UInt =0, out_vert: UInt =1, r: Float[Array, "n+1 n+1"]=(20 * jnp.eye(8) + 0.1 * jnp.ones((8,8))), eps=STABILITY_CONSTANT) -> Float:
    """
    probability density of transition from in_vert to out_vert at time t, given transition rates r[i,j] from vertex i to vertex j. 
    r is a (n+1, n+1) array where the first row and column correspond to the center vertex. 
    The probability is computed using the formula derived from the master equation of the Markov process.
    This function broadcasts jointly over in_vert, out_vert, t so they can be arrays of compatible shapes.
    """
    return p_model_one_center(t, in_vert, out_vert, r, eps) / (p_remain_model_one_center(t, in_vert, r, eps) + eps)

@filter_jit
def log_likelihood(r: Float[Array, "n+1 n+1"], data: TTTree | Array | pd.DataFrame, p_model: Callable = p_model_one_center, dt: Float=jnp.float32(0.002), normalize: Bool=True, eps: Float=STABILITY_CONSTANT, num_neighbors: Int | None = None) -> Float:
    """Log likelihood.
    
    Args:
        r: Input parameter.
        data: Input parameter.
        p_model: Input parameter.
        dt: Input parameter.
        normalize: Input parameter.
        eps: Input parameter.
        num_neighbors: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    if num_neighbors is None:
        num_neighbors = r.shape[0] - 1
    if isinstance(data, (pd.DataFrame, dict)):
        count_in_vert = jnp.sum((jnp.arange(num_neighbors, dtype=int).reshape(1, -1) == data["in_vert"][:, None]) * data["count"][:, None], axis=0) if isinstance(data, dict) else data.groupby("in_vert")["count"].sum()

        p_from_data = data["count"] / count_in_vert[data["in_vert"]] / dt # type: ignore
        p_from_model = p_model(data["t"] * dt, data["in_vert"], data["out_vert"], r)

        return jnp.sum(data["count"] * (jnp.log(p_from_model + eps) - normalize * jnp.log(p_from_data + eps)))
    elif isinstance(data, jnp.ndarray):
        count_in_vert = jnp.sum(data, axis=(1,2))
        p_from_data = data / count_in_vert[:, None, None] / dt
        ts = jnp.arange(1, data.shape[-1] + 1) * dt
        in_verts = jnp.arange(data.shape[0])[:, None, None]
        out_verts = jnp.arange(data.shape[1])[None, :, None]
        mask= (in_verts < num_neighbors) & (out_verts < num_neighbors)
        p_from_model = p_model(ts, in_verts, out_verts, r)

        return jnp.sum(jnp.where(mask, data * (jnp.log(p_from_model + eps) - normalize * jnp.log(p_from_data + eps)), 0.)) # type: ignore
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def compare_models_to_data(in_vert: Int, out_vert: Int, rs: Iterable[Float[Array, "n+1 n+1"]], data: TTTree | Array | pd.DataFrame, ax: Axes | None = None, p_model: Callable = p_model_one_center, dt: Float = 0.002, tmax: Float = 0.5, linestyles: Sequence[str] = ("-", "--", "-.", ":")):
    """Compare models to data.
    
    Args:
        in_vert: Input parameter.
        out_vert: Input parameter.
        rs: Input parameter.
        data: Input parameter.
        ax: Input parameter.
        p_model: Input parameter.
        dt: Input parameter.
        tmax: Input parameter.
        linestyles: Input parameter.
    """
    if isinstance(data, pd.DataFrame):
        df = data.query(f"in_vert=={in_vert} & out_vert=={out_vert}")
        total_count = data.query(f"in_vert=={in_vert}")["count"].sum()
        df_crop = df[df["t"] * dt <= tmax]
        ts = jnp.asarray(df_crop["t"]) * dt
        y = df_crop["count"].values/total_count/dt
    elif isinstance(data, dict):
        mask = (data["in_vert"] == in_vert) & (data["out_vert"] == out_vert) & (data["t"] * dt <= tmax)
        df_crop = {"t": data["t"][mask], "count": data["count"][mask]}
        ts = data["t"][mask] * dt
        total_count = jnp.sum((data["in_vert"] == in_vert) * data["count"])
        y = data["count"][mask]/total_count/dt
    elif isinstance(data, jnp.ndarray):
        total_count = data[in_vert, :, :].sum(dtype=jnp.float32)
        y = data[in_vert, out_vert, :int(tmax/dt)] / total_count / dt
        ts = jnp.arange(1, y.shape[0] + 1) * dt 
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))

    y_err = jnp.sqrt(y/total_count/dt) # poisson error bars
    ax.errorbar(ts, y, yerr=y_err, fmt='o', alpha=0.2, label="data")
    for i, r in enumerate(rs):
        prob = p_model(ts, in_vert, out_vert, r)
        ax.plot(ts, prob, label=f"model {i}", linestyle=linestyles[i % len(linestyles)], zorder=10)
    ax.legend(loc="upper right")
    ax.set_yscale("log")

def display_log_likelihood_contributions(r: Float[Array, "n+1 n+1"], data: TTTree | Array | pd.DataFrame, p_model: Callable = p_model_one_center, dt: Float=jnp.float32(0.002), normalize: Bool=True, axs: Collection | None = None, eps: Float=STABILITY_CONSTANT, tmax=0.2, cs=COLORS):
    """Display log likelihood contributions.
    
    Args:
        r: Input parameter.
        data: Input parameter.
        p_model: Input parameter.
        dt: Input parameter.
        normalize: Input parameter.
        axs: Input parameter.
        eps: Input parameter.
        tmax: Input parameter.
        cs: Input parameter.
    """
    num_neighbors = r.shape[0] - 1
    if axs is None:
        fig, axs = plt.subplots((num_neighbors + 1) // 2, 2, figsize=(15,3 * ((num_neighbors + 1) // 2)), sharex=True)
    assert axs is not None

    if isinstance(data, (pd.DataFrame, dict)):
        data = pd.DataFrame(data) if isinstance(data, dict) else data
        count_in_vert = data.groupby("in_vert")["count"].sum()

        p_from_data = jnp.asarray(data["count"]) / jnp.asarray(count_in_vert[data["in_vert"]]) / dt
        p_from_model = p_model(jnp.asarray(data["t"]) * dt, jnp.asarray(data["in_vert"]), jnp.asarray(data["out_vert"]), r)

        ll_contributions = jnp.asarray(data["count"]) * (jnp.log(p_from_model + eps) - normalize * jnp.log(p_from_data + eps))

        t_mask = data["t"] * dt <= tmax
        
        for i, a in enumerate(axs.flat): # type: ignore
            if i >= num_neighbors:
                a.set_axis_off()    
                break
            in_mask = data["in_vert"] == i
            for j in range(num_neighbors):
                out_mask = data["out_vert"] == j
                mask = in_mask & out_mask & t_mask
                a.scatter(jnp.asarray(data["t"])[mask] * dt, ll_contributions[mask], alpha=0.5, label=j, c=cs[j % len(cs)])
            if i >= num_neighbors - 2:
                a.set_xlabel("time")
                a.tick_params(labelbottom=True)
            else:
                a.tick_params(labelbottom=False)
            a.set_title(f"in_vert={i}", c=cs[i % len(cs)])
            a.set_ylabel("log-likelihood contribution")
            a.legend(title="out_vert", loc="lower right")
            lt = jnp.quantile(jnp.abs(ll_contributions[in_mask & t_mask]), 0.95)
            a.set_yscale("symlog", linthresh=float(lt))
        
    elif isinstance(data, jnp.ndarray):
        count_in_vert = jnp.sum(data, axis=(1,2))
        p_from_data = data / count_in_vert[:, None, None] / dt
        ts = jnp.arange(1, data.shape[-1] + 1) * dt
        in_verts = jnp.arange(data.shape[0])[:, None, None]
        out_verts = jnp.arange(data.shape[1])[None, :, None]
        p_from_model = p_model(ts, in_verts, out_verts, r)
        ll_contributions = data * (jnp.log(p_from_model + eps) - normalize * jnp.log(p_from_data + eps))

        if axs is None:
            fig, axs = plt.subplots(num_neighbors, figsize=(10,2 * num_neighbors), sharex=True)
        assert axs is not None
        t_max_int = int(tmax/dt)

        for i, a in enumerate(axs.flat): # type: ignore
            if i >= num_neighbors:
                a.set_axis_off()    
                break
            for j in range(num_neighbors):
                a.scatter(np.asarray(ts[:t_max_int]), np.asarray(ll_contributions[i, j, :t_max_int]), alpha=0.5, label=j, c=cs[j % len(cs)])
            if i >= num_neighbors - 2:
                a.set_xlabel("time")
                a.tick_params(labelbottom=True)
            else:
                a.tick_params(labelbottom=False)
            if i % 2 == 0:
                a.set_ylabel("llh")
            a.set_title(f"in_vert={i}", c=cs[i % len(cs)])
            a.legend(title="out_vert", loc="lower right")
            lt = jnp.quantile(jnp.abs(ll_contributions[i, :t_max_int][ll_contributions[i, :t_max_int] > 0]), 0.95)
            a.set_yscale("symlog", linthresh=float(lt))
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    return

@filter_jit
def get_mean_time(data_dict: TTTree, mask: Bool | Int = True, dt: Float=0.002):
    """Return mean time.
    
    Args:
        data_dict: Input parameter.
        mask: Input parameter.
        dt: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    num = (mask * data_dict["t"] * dt * data_dict["count"]).sum(axis=-1)  # already float, to avoid overflow
    denom = (mask * data_dict["count"]).sum(axis=-1)
    return num / denom

@filter_jit
def get_initial_decay_rate(data_dict: TTTree, dt=0.002, nsteps=20):
    """Return initial decay rate.
    
    Args:
        data_dict: Input parameter.
        dt: Input parameter.
        nsteps: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    return jnp.log(data_dict["count"][0] / data_dict["count"][nsteps]) / (data_dict["t"][nsteps] - data_dict["t"][0]) * dt


@filter_jit
def guesstimate_rates_model_one_center(data: TTTree | Array, dt: Float=0.002, verbose: Bool=True) -> Float[Array, "n+1 n+1"]:
    """Guesstimate rates model one center.
    
    Args:
        data: Input parameter.
        dt: Input parameter.
        verbose: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    if isinstance(data, dict):
        required_keys = {"t", "in_vert", "out_vert", "count"}
        assert all(key in data for key in required_keys), f"Data dict must contain keys: {required_keys}"
    elif isinstance(data, jnp.ndarray):
        assert data.ndim == 3, f"Data array must be 3-dimensional, got shape {data.shape}"
        assert data.shape[0] == data.shape[1], f"First two dimensions of data array must be equal, got shape {data.shape}"
    else:
        raise ValueError(f"Data must be either a dict or a 3D array, got type {type(data)}")

    if isinstance(data, dict):
        num_neighbors = len(data["neighbor_ids"])
        mean_time = get_mean_time(data, dt=dt)
        max_time = data["t"].max() * dt
        ts = None
    else:
        num_neighbors = data.shape[0]
        ts = jnp.arange(1, data.shape[-1] + 1) * dt
        mean_time = jnp.sum(ts * data.sum(axis=(-3,-2), dtype=jnp.float32)) / data.sum()
        max_time = ts[-1]
    
    t_crop_end = jnp.minimum(0.5 * mean_time, 0.2 * max_time)

    if isinstance(data, dict):
        end_mask = data["t"] >= t_crop_end / dt
        lambda_c = 1. / (get_mean_time(data, mask=end_mask, dt=dt) - t_crop_end)

        in_vert_masks = jnp.arange(num_neighbors, dtype=int)[:,None] == data["in_vert"][None, :]
        end_count_in_vert = jnp.sum(end_mask[None,:] * in_vert_masks * data["count"][None, :], axis=-1)
        total_count_in_vert = jnp.sum(in_vert_masks * data["count"][None, :], axis=-1)
        tau_is = vmap(get_mean_time, in_axes=(None, 0, None))(data, in_vert_masks, dt) # mean time for each in_vert

    else:
        assert ts is not None
        end_mask = ts >= t_crop_end
        lambda_c = 1. / (jnp.sum(end_mask * ts * data.sum(axis=(0,1), dtype=jnp.float32), axis=-1)/jnp.sum(end_mask * data, dtype=jnp.float32) - t_crop_end) # heuristic to estimate the decay rate of the center vertex, considering only transitions that happen after t_crop_end
        
        in_vert_masks = None
        end_count_in_vert = jnp.sum(end_mask[None,:] * data.sum(axis=1), axis=-1)
        total_count_in_vert = jnp.sum(data, axis=(1,-1))
        tau_is = jnp.sum(ts[None,:] * data.sum(axis=1, dtype=jnp.float32), axis=-1) / total_count_in_vert # mean time for each in_vert, considering only transitions that happen after t_crop_end 
    
    q_is = end_count_in_vert * jnp.exp(lambda_c * t_crop_end) / total_count_in_vert # coefficient in front of the exponential term
    lambda_is = jnp.nan_to_num((1 - q_is)/(tau_is * lambda_c - q_is) * lambda_c) # solving the set of equations

    t_crop_start = jnp.clip(0.1 / jnp.nan_to_num(lambda_is, nan=jnp.inf).min(), min= 3.5 * dt, max=0.5 * t_crop_end) # heuristic to avoid noise from very short times, but also not to crop too much data

    # if verbose: # not inside jit
    #     print(f"Using t_crop_start={t_crop_start}, t_crop_end={t_crop_end}, lambda_c={lambda_c}")

    if isinstance(data, dict):
        out_vert_masks = jnp.arange(num_neighbors, dtype=int)[:, None] == data["out_vert"][None, :]
        end_count_out_vert = jnp.sum(end_mask[None,:] * out_vert_masks * data["count"][None, :], axis=-1)

        assert in_vert_masks is not None
        in_out_vertex_masks = in_vert_masks[:, None, :] & out_vert_masks[None,:, :]
        start_mask = data["t"] <= (t_crop_start / dt).astype(jnp.uint16)
        start_count_in_vert = jnp.sum(start_mask[None,:] * in_vert_masks * data["count"][None, :], axis=-1)
        fractions = jnp.sum(start_mask[None, None, :] * in_out_vertex_masks * data["count"][None, None, :], axis=-1) / start_count_in_vert[:, None]

    else:
        end_count_out_vert = jnp.sum(end_mask[None,:] * data.sum(axis=0), axis=-1)
        start_mask = ts <= t_crop_start # type: ignore
        fractions = jnp.sum(start_mask[None, None, :] * data, axis=-1)
        fractions /= fractions.sum(axis=1, keepdims=True)

    end_count_tot = jnp.sum(end_count_in_vert)

    rci = lambda_c * end_count_out_vert / end_count_tot

    ric = jnp.nan_to_num((lambda_is - lambda_c) * q_is)

    rij = (lambda_is - ric)[:, None] * jnp.nan_to_num(fractions)

    r0 = jnp.zeros((num_neighbors + 1, num_neighbors + 1))
    r0 = r0.at[1:, 1:].set(rij)
    r0 = r0.at[1:, 0].set(ric)
    r0 = r0.at[0, 1:].set(rci)

    return r0

def get_model_one_center_statistics(
    r: Float[Array, "n+1 n+1"],
    num_neighbors: int | None = None,
    as_df: Bool = False
) -> Tuple[Float[Array, "n+ n"],...] | Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute model transition probabilities and MFPTs.

    Args:
        r: Transition rate matrix with center at index 0.
        num_neighbors: If provided, discard entries with in/out >= num_neighbors.
    """
    if num_neighbors is None:
        num_neighbors = r.shape[0] - 1
    assert num_neighbors >= 0, "num_neighbors must be non-negative"
    assert r.shape[0] >= num_neighbors + 1, "r does not include requested num_neighbors"

    r = r[:num_neighbors + 1, :num_neighbors + 1]
    l_center = jnp.sum(r[0, 1:])
    l_in = jnp.sum(r[1:], axis=1, keepdims=True)
    transition_probabilities = r[[0], 1:] * r[1:, [0]] / (l_in * l_center) + r[1:, 1:] / l_in
    mfpt = (r[[0], 1:] * r[1:, [0]] / (l_in * l_center) * (1 / l_in + 1 / l_center) + r[1:, 1:] / l_in ** 2) / transition_probabilities
    neighbors = jnp.arange(num_neighbors)
    if not as_df:
        return transition_probabilities, mfpt
    return (
        pd.DataFrame(transition_probabilities, columns=pd.Series(neighbors, name="out_vert"), index=pd.Series(neighbors, name="in_vert")),
        pd.DataFrame(mfpt, columns=pd.Series(neighbors, name="out_vert"), index=pd.Series(neighbors, name="in_vert"))
    )

def get_data_statistics(
    data: TTTree | Array | pd.DataFrame,
    dt: Float = 0.002,
    num_neighbors: int | None = None,
    as_df: Bool = False
) -> Tuple[Float[Array, "n+ n"],...] | Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute data transition probabilities and MFPTs.

    Args:
        data: Transition count data.
        num_neighbors: If provided, discard entries with in/out >= num_neighbors.
    """
    if isinstance(data, jnp.ndarray):
        if num_neighbors is None:
            num_neighbors = data.shape[0]
        assert data.shape[0] >= num_neighbors and data.shape[1] >= num_neighbors, "data does not include requested num_neighbors"
        data = data[:num_neighbors, :num_neighbors]

        ts = jnp.arange(1, data.shape[-1] + 1) * dt
        count_in_vert = jnp.sum(data, axis=(1, 2))
        total_in_out_counts = data.sum(axis=-1)
        transition_probabilities = total_in_out_counts / count_in_vert[:, None]
        mfpt = jnp.sum(ts[None, None, :] * data, axis=-1) / total_in_out_counts
        neighbors = jnp.arange(num_neighbors)
        if not as_df:
            return transition_probabilities, mfpt
        return (
            pd.DataFrame(transition_probabilities, columns=pd.Series(neighbors, name="out_vert"), index=pd.Series(neighbors, name="in_vert")),
            pd.DataFrame(mfpt, columns=pd.Series(neighbors, name="out_vert"), index=pd.Series(neighbors, name="in_vert"))
        )

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if num_neighbors is not None:
        df = df[(df["in_vert"] < num_neighbors) & (df["out_vert"] < num_neighbors)]

    grouped = df.groupby(["in_vert", "out_vert"])
    transition_probabilities = grouped["count"].sum().div(df.groupby("in_vert")["count"].sum())
    mfpt = grouped.apply(lambda df: get_mean_time(df, dt=dt))
    if not as_df:
        return jnp.asarray(transition_probabilities.values), jnp.asarray(mfpt.values)
    return transition_probabilities.unstack(), mfpt.unstack()

@filter_jit
def fit_model_one_center(data: TTTree | Array | pd.DataFrame, n_neighbors: int | None = None, solver: optx.AbstractMinimiser = optx.BFGS(rtol=0, atol=1e-2), max_steps: int = 2**6, dt: Float = 0.002, verbose=True, eps: Float = STABILITY_CONSTANT, zero_clip = 1e-5) -> Tuple[Float[Array, "n+1 n+1"], Bool]:
    """Fit model one center.
    
    Args:
        data: Input parameter.
        n_neighbors: Input parameter.
        solver: Input parameter.
        max_steps: Input parameter.
        dt: Input parameter.
        verbose: Input parameter.
        eps: Input parameter.
        zero_clip: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    r_init = guesstimate_rates_model_one_center(data, dt=dt, verbose=verbose)
    loglikelihood_init = log_likelihood(r_init, data, dt=dt, num_neighbors=n_neighbors, eps=eps)
    loss = lambda y, args=None: - log_likelihood(jnp.exp(y), data, dt=dt, num_neighbors=n_neighbors, eps=eps) / jnp.abs(loglikelihood_init)
    y0 = jnp.log(jnp.nan_to_num(r_init) + eps)
    sol = optx.minimise(fn=loss, y0=y0, solver=solver, max_steps=max_steps, throw=False)
    y = sol.value
    success = sol.result == optx.RESULTS.successful
    # y, success = y0, False
    return jnp.clip(jnp.exp(y) - zero_clip, 0, jnp.inf), success

def get_default_solver_dict(rtol=0, atol=1e-2):
    """Return default solver dict.
    
    Args:
        rtol: Input parameter.
        atol: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    tols = {"rtol": rtol, "atol": atol, "norm": optx.max_norm}
    adam_fast = optx.OptaxMinimiser(optax.adam(learning_rate=5e-2), **tols)
    adam_slow = optx.OptaxMinimiser(optax.adam(learning_rate=2e-2), **tols)
    adabelief = optx.OptaxMinimiser(optax.adabelief(learning_rate=7e-2), **tols)
    adamw = optx.OptaxMinimiser(optax.adamw(learning_rate=7e-2), **tols)
    nadam = optx.OptaxMinimiser(optax.nadam(learning_rate=3e-2), **tols)
    lion = optx.OptaxMinimiser(optax.lion(learning_rate=1e-2), **tols)
    bfgs = optx.LBFGS(**tols)
    lbfgs = optx.LBFGS(**tols)
    dfp = optx.DFP(**tols)    
    ncg_pr = optx.NonlinearCG(**tols, method=optx.polak_ribiere)
    ncg_fr = optx.NonlinearCG(**tols, method=optx.fletcher_reeves)
    ncg_hs = optx.NonlinearCG(**tols, method=optx.hestenes_stiefel)
    ncg_dy = optx.NonlinearCG(**tols, method=optx.dai_yuan)

    return { "dfp": dfp, "ncg_fr": ncg_fr, "ncg_hs": ncg_hs, "ncg_dy": ncg_dy, "ncg_pr": ncg_pr, "bfgs": bfgs, "lbfgs": lbfgs, "adam_fast": adam_fast, "adam_slow": adam_slow, "adabelief": adabelief, "adamw": adamw, "nadam": nadam, "lion": lion}

def compare_solvers(data: TTTree | Array | pd.DataFrame, num_neighbors: int | None = None, solvers: dict | None = None, max_steps: int = 2**5, dt: Float = 0.002, verbose=True, eps: Float = STABILITY_CONSTANT):
    """Compare solvers.
    
    Args:
        data: Input parameter.
        num_neighbors: Input parameter.
        solvers: Input parameter.
        max_steps: Input parameter.
        dt: Input parameter.
        verbose: Input parameter.
        eps: Input parameter.
    
    Returns:
        Output value computed by this function.
    """
    if solvers is None:
        solvers = get_default_solver_dict()
    losses = {}
    times = {}
    r0 = guesstimate_rates_model_one_center(data, dt=dt, verbose=verbose)
    y0 = jnp.log(r0 + eps)
    loglikelihood_init = log_likelihood(r0, data, dt=dt, eps=eps)
    fn = filter_jit(lambda y, args=None: (-log_likelihood(jnp.exp(y), data, dt=dt, eps=eps) / jnp.abs(loglikelihood_init), None))
    
    default_stuff = {"args": None, "options": {}, "tags": frozenset()}

    for name, solver in tqdm(solvers.items()):

        step = eqx.filter_jit(
            eqx.Partial(solver.step, fn=fn, **default_stuff)
        )
        terminate = eqx.filter_jit(
            eqx.Partial(solver.terminate, fn=fn, **default_stuff)
        )

        y = y0
        state = solver.init(fn, y, **default_stuff, f_struct=jax.ShapeDtypeStruct((), jnp.float32), aux_struct=None)
        done, result = terminate(y=y, state=state)

        loss_arr = jnp.full(max_steps, jnp.nan)
        for i in range(max_steps): 
            if done:
                if verbose:
                    print("done after", i, "steps")
                break
            loss_arr = loss_arr.at[i].set(fn(y, None)[0])
            y, state, aux = step(y=y, state=state)
            done, result = terminate(y=y, state=state)
        
        y, _, _ = solver.postprocess(fn, y, aux=None, **default_stuff, state=state, result=result)

        @eqx.filter_jit
        def run(y0):
            """Run.
            
            Args:
                y0: Input parameter.
            
            Returns:
                Output value computed by this function.
            """
            if verbose:
                print("compiling...")
            return optx.minimise(fn=fn, y0=y0, solver=solver, max_steps=max_steps, has_aux=True, throw=False).value
        
        run(y0) # warmup
        times[name] = timeit(stmt=lambda: run(y0).block_until_ready(), number=5)/5

        if verbose and result != optx.RESULTS.successful:
            print(f"Oh no! Got error {result}.")
        
        losses[name] = loss_arr

    fig, ax = plt.subplots(figsize=(12,8))
    styles = ["-", "--", "-.", ":"]
    for i, (name, loss_arr) in enumerate(losses.items()):
        ax.plot(loss_arr, label=name + f" ({times[name] * 1e3 :.2g}ms)", ls=styles[i%len(styles)])
    ax.set_ylabel("loss")
    ax.set_xlabel("iteration")
    ax.legend()

def goodness_of_fit(
    r: Float[Array, "n+1 n+1"],
    data: Array | pd.DataFrame,
    dt: Float = 0.002,
    eps: Float = STABILITY_CONSTANT,
    display_dfs: Bool = True,
    neighbor_ids: Array | None = None,
    num_neighbors: int | None = None
):
    """Compute goodness of fit between model and data.

    Args:
        num_neighbors: If provided, discard entries with in/out >= num_neighbors.
    """
    model_tp, model_mfpt = get_model_one_center_statistics(r, num_neighbors=num_neighbors, as_df=False)
    data_tp, data_mfpt = get_data_statistics(data, dt=dt, num_neighbors=num_neighbors, as_df=False)
    delta_tp = jnp.asarray(model_tp - data_tp)
    delta_log_mfpt = jnp.log(jnp.asarray(model_mfpt/data_mfpt)) # type: ignore

    if isinstance(data, pd.DataFrame):
        df = data
        if num_neighbors is not None:
            df = df[(df["in_vert"] < num_neighbors) & (df["out_vert"] < num_neighbors)]
        counts_df = df.groupby(["in_vert", "out_vert"])["count"].sum().unstack().fillna(0)
        if num_neighbors is not None:
            counts_df = counts_df.reindex(index=range(num_neighbors), columns=range(num_neighbors), fill_value=0)
        counts = jnp.asarray(counts_df.values)
    elif isinstance(data, jnp.ndarray):
        if num_neighbors is None:
            num_neighbors = data.shape[0]
        counts = jnp.sum(data[:num_neighbors, :num_neighbors], axis=-1)
    else:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")
    
    fpt_weights = counts / counts.sum()
    tp_weights = counts.sum(axis=-1, keepdims=True) / counts.sum()
    fpt_score = jnp.sqrt(jnp.sum(fpt_weights * delta_log_mfpt ** 2))
    tp_score = jnp.sum(tp_weights * jnp.abs(delta_tp))

    if display_dfs:
        if neighbor_ids is None:
            if num_neighbors is None:
                neighbor_ids = jnp.arange(r.shape[0]-1)
            else:
                neighbor_ids = jnp.arange(num_neighbors)
        verts = neighbor_ids.tolist()
        def to_df(arr, name, cmap="Blues", vmin=None, vmax=None, axis=None, fmt="{:.2g}"):
            """To df.
            
            Args:
                arr: Input parameter.
                name: Input parameter.
                cmap: Input parameter.
                vmin: Input parameter.
                vmax: Input parameter.
                axis: Input parameter.
                fmt: Input parameter.
            
            Returns:
                Output value computed by this function.
            """
            return pd.DataFrame(arr, columns=pd.Series(verts, name="out_vert"), index=pd.Series(verts, name="in_vert")).style.format(fmt).background_gradient(cmap=cmap, vmin=vmin, vmax=vmax, axis=axis).set_caption(name).set_table_attributes('style="display:inline"')

        df_tp = to_df(data_tp, "Data Transition Probabilities", cmap="Reds", vmin=0, vmax=1, fmt="{:.2%}")
        df_mfpt = to_df(data_mfpt, "Data MFPTs", cmap="cool_r")
        max_tp_error = jnp.max(jnp.abs(delta_tp))
        df_tp_error = to_df(delta_tp, "Difference in Transition Probabilities", cmap="RdBu_r", vmin=-max_tp_error, vmax=max_tp_error, fmt="{:.2%}")
        max_mfpt_error = jnp.max(jnp.abs(delta_log_mfpt))
        df_mfpt_error = to_df(delta_log_mfpt, "Difference in log MFPTs", cmap="PuOr", vmin=-max_mfpt_error, vmax=max_mfpt_error, fmt="{:.2%}")
        df_weight = to_df(fpt_weights, "Weights (%)", cmap="Greens", vmin=0, vmax=jnp.max(fpt_weights), fmt="{:.2%}")
        display_html(df_tp._repr_html_() + df_tp_error._repr_html_(), raw=True) # type: ignore
        display_html(df_mfpt._repr_html_() + df_mfpt_error._repr_html_(), raw=True) # type: ignore
        display_html(df_weight._repr_html_(), raw=True) # type: ignore

        print(f"root mean squared log MFPT error: {fpt_score:.4g}, mean absolute TP error: {tp_score:.4g}")
    
    return fpt_score, tp_score


def fit_graph_to_field(
    transition_array: Float[Array, "n_cells max_neighbors+1 max_neighbors+1 t_max"],
    neighbors: Int[Array, "n_cells max_neighbors"],
    degrees: Int[Array, "n_cells"],
    model_fit_function: Callable = fit_model_one_center,
    dt: Float = 0.002,
    solver: optx.AbstractMinimiser = optx.BFGS(rtol=0, atol=1e-2),
    max_steps: int = 2**5,
    eps: Float = STABILITY_CONSTANT,
    use_scipy_sparse: Bool = True,
    verbose: Bool = False
) -> Tuple[Float[Array, "n_vertices n_vertices"] | Any, List[str]]:
    """
    Fit transition rate models to all cells in a field and construct the global graph.
    
    Args:
        transition_array: Shape (n_cells, max_neighbors+1, max_neighbors+1, t_max), transition counts
        neighbors: Shape (n_cells, max_neighbors), neighbor indices (padded)
        degrees: Shape (n_cells,), number of actual neighbors per cell
        model_fit_function: Function to fit each cell's model
        use_scipy_sparse: If True, return scipy.sparse.csr_matrix, else dense array
        
    Returns:
        Transition rate matrix (sparse or dense) and list of vertex labels
        Labels format: "c(i)" for centers, "e(i,j)" for directed edges
    """
    n_cells = transition_array.shape[0]
    max_neighbors = neighbors.shape[1]
    
    # Fit models for each cell
    fitted_rates = []
    fitted_success = []
    
    if verbose:
        print(f"Fitting {n_cells} cell models...")
    
    for cell_id in range(n_cells):
        degree = int(degrees[cell_id])
        # Extract data for this cell (shape: degree+1, degree+1, t_max)
        cell_data = transition_array[cell_id, :degree, :degree, :]
        
        r, success = model_fit_function(
            cell_data, n_neighbors=degree, solver=solver, 
            max_steps=max_steps, dt=dt, verbose=False, eps=eps
        )
        fitted_rates.append(r)
        fitted_success.append(success)
    
    if verbose:
        n_success = sum(fitted_success)
        print(f"Successfully fitted {n_success}/{n_cells} cells")
    
    # Create vertex labels
    vertex_labels = []
    label_to_idx = {}
    
    # Add all center nodes first
    for i in range(n_cells):
        label = f"c({i})"
        label_to_idx[label] = len(vertex_labels)
        vertex_labels.append(label)
    
    # Add all directed edge nodes
    for i in range(n_cells):
        degree_i = int(degrees[i])
        for j_local in range(degree_i):
            j = int(neighbors[i, j_local])
            # Add both incoming and outgoing edge nodes for this cell
            label_in = f"e({j},{i})"
            label_out = f"e({i},{j})"
            
            if label_in not in label_to_idx:
                label_to_idx[label_in] = len(vertex_labels)
                vertex_labels.append(label_in)
            if label_out not in label_to_idx:
                label_to_idx[label_out] = len(vertex_labels)
                vertex_labels.append(label_out)
    
    n_vertices = len(vertex_labels)
    
    if verbose:
        print(f"Global graph has {n_vertices} vertices ({n_cells} centers + {n_vertices - n_cells} edges)")
    
    # Build sparse matrix using lists of (row, col, data)
    rows, cols, data = [], [], []
    
    for cell_id in range(n_cells):
        r = fitted_rates[cell_id]
        degree = int(degrees[cell_id])
        neighbors_i = neighbors[cell_id, :degree]
        
        center_idx = label_to_idx[f"c({cell_id})"]
        
        # Transitions from incoming edges to center
        # Center is at index 0, so we access r[j_local+1, 0]
        for j_local in range(degree):
            j = int(neighbors_i[j_local])
            in_edge_idx = label_to_idx[f"e({j},{cell_id})"]
            rate = float(r[j_local + 1, 0])  # neighbor to center (center is at index 0)
            if rate > 0:
                rows.append(in_edge_idx)
                cols.append(center_idx)
                data.append(rate)
        
        # Transitions from incoming edges to outgoing edges
        # Neighbors are at indices 1..degree, so we access r[j_local+1, k_local+1]
        for j_local in range(degree):
            j = int(neighbors_i[j_local])
            in_edge_idx = label_to_idx[f"e({j},{cell_id})"]
            for k_local in range(degree):
                k = int(neighbors_i[k_local])
                out_edge_idx = label_to_idx[f"e({cell_id},{k})"]
                rate = float(r[j_local + 1, k_local + 1])  # neighbor to neighbor
                if rate > 0:
                    rows.append(in_edge_idx)
                    cols.append(out_edge_idx)
                    data.append(rate)
        
        # Transitions from center to outgoing edges
        # Center is at index 0, so we access r[0, k_local+1]
        for k_local in range(degree):
            k = int(neighbors_i[k_local])
            out_edge_idx = label_to_idx[f"e({cell_id},{k})"]
            rate = float(r[0, k_local + 1])  # center to neighbor (center is at index 0)
            if rate > 0:
                rows.append(center_idx)
                cols.append(out_edge_idx)
                data.append(rate)
    
    if verbose:
        print(f"Global graph has {len(data)} nonzero transitions")
        sparsity = 100 * (1 - len(data) / (n_vertices ** 2))
        print(f"Sparsity: {sparsity:.2f}% (matrix size: {n_vertices}x{n_vertices})")
    
    if use_scipy_sparse:
        import scipy.sparse
        transition_matrix = scipy.sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(n_vertices, n_vertices)
        )
    else:
        transition_matrix = jnp.zeros((n_vertices, n_vertices))
        for row, col, val in zip(rows, cols, data):
            transition_matrix = transition_matrix.at[row, col].set(val)
    
    return transition_matrix, vertex_labels


