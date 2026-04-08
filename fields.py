import jax
from abc import ABC
import interpax
from xxhash import xxh64_hexdigest
import time 
from skimage.measure import marching_cubes
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optax
import equinox as eqx
from timeit import timeit
from jax import jit, vmap
import jax.random as jr
from jax.typing import ArrayLike
from jaxtyping import Float, Array, Int, UInt, Complex, PyTree, Bool
from typing import Iterable, Tuple, Callable, Protocol
import zarr, pickle
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

class Discretizer(Protocol):
    def __call__(self, xs: Float[Array, "n_points dim"], brute_force: Bool = False) -> Int[Array, "n_points"]:
        ...
    
from functools import partial
from tqdm.notebook import tqdm, tnrange
import os
import sys
import os
sys.path.append(os.path.abspath(".."))

from helpers.plotting_fields import (
    plot_field,
    plot_stationary_distribution_sample,
    plot_stationary_distribution_test,
)
from helpers.geometry_helpers import Tesselation, tesselation_1d, tesselation_2d, tesselation_3d
import jax.scipy.stats as js
import freud


class Field(eqx.Module, ABC):
    """Abstract base class describing a spatial field.

    The default evaluators are Fourier-based and use the coefficient
    tensor ``A`` and wavevector axes ``ks``. Subclasses can override
    methods when needed, but plotting and derived quantities work out of
    the box for any field exposing ``A`` and ``ks``.
    """

    dim: int  # number of spatial dimensions
    size: Float[Array, "dim"]  # physical size per dimension
    ks: tuple[Array, ...]  # k-space axes, one per dimension
    A: Complex[Array, "ki ..."]  # complex amplitudes on k-grid
    EA: Float[Array, "ki ..."] | None  # sqrt(E[|A|^2])
    k0: Float  # typical wavenumber
    lambda0: Float # typical wavelength
    num_vert: UInt # number of discretization vertices in the base cell
    discr_points: Float[Array, "num_vert dim"]
    discr_points_extended: Float[Array, "num_vert_extended dim"]
    discr_indices_extended: Int[Array, "num_vert_extended"]
    phi_interpolator: interpax.Interpolator1D | interpax.Interpolator2D | interpax.Interpolator3D | None
    force_interpolators: tuple[interpax.Interpolator1D | interpax.Interpolator2D | interpax.Interpolator3D, ...] | None
    phi_min: Float
    phi_moments: Float[Array, "n_moments"]
    normalization_constant: Float
    lower_step: Array | None
    lower_step_margin: Float | None
    lower_step_bins: tuple[int, ...] | None
    lower_step_side: Float[Array, "dim"] | None
    lower_step_origin: Float[Array, "dim"] | None
    voro: freud.locality.Voronoi | None
    tesselation: Tesselation | None

    def __hash__(self) -> int:
        return hash((str(self.A), str(self.size)))

    def static_hash(self) -> str:
        return xxh64_hexdigest(str((self.size, self.A, self.num_vert, self.discr_points)))
    
    # --- core pointwise evaluators (Fourier defaults based on A, ks) ---
    @partial(jit, static_argnames=('self',))
    def potential(self, x: Float[Array, "dim"]) -> Float:
        """Pointwise evaluation of the scalar potential at position ``x``.

        Parameters
        - x: array of shape (dim,) giving a single point in space.

        Returns
        - scalar potential at ``x``.
        """
        exp_ikx = jnp.exp(1j * sum(jnp.meshgrid(*[xi * ki for xi, ki in zip(x, self.ks)], indexing='ij')))
        phi = jnp.sum(exp_ikx * self.A)
        return jnp.real(phi)

    @partial(jit, static_argnames=('self',)) 
    def force(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """Pointwise evaluation of the force (negative gradient) at ``x``."""
        k_mesh = jnp.array(jnp.meshgrid(*self.ks, indexing='ij'))
        ik_exp_ikx = 1j * k_mesh * jnp.exp(1j * sum(jnp.meshgrid(*[xi * ki for xi, ki in zip(x, self.ks)], indexing='ij')))
        force = -jnp.tensordot(ik_exp_ikx, self.A, axes=self.dim)
        return jnp.real(force)

    @partial(jit, static_argnames=('self',))
    def hessian(self, x: Float[Array, "dim"]) -> Float[Array, "dim dim"]:
        """Pointwise Hessian (matrix) at ``x``."""
        k_mesh = jnp.array(jnp.meshgrid(*self.ks, indexing='ij'))
        kk_t_exp_ikx = -jnp.expand_dims(k_mesh, axis=0) * jnp.expand_dims(k_mesh, axis=1) * jnp.exp(
            1j * sum(jnp.meshgrid(*[xi * ki for xi, ki in zip(x, self.ks)], indexing='ij'))
        )
        hess = jnp.tensordot(kk_t_exp_ikx, self.A, axes=self.dim)
        return jnp.real(hess)

    @partial(jit, static_argnames=('self',))
    def laplacian(self, x: Float[Array, "dim"]) -> Float:
        """Pointwise Laplacian (trace of Hessian) at ``x``."""
        k_mesh = jnp.array(jnp.meshgrid(*self.ks, indexing='ij'))
        k_t_k_exp_ikx = -jnp.sum(k_mesh ** 2, axis=0) * jnp.exp(1j * sum(jnp.meshgrid(*[xi * ki for xi, ki in zip(x, self.ks)], indexing='ij')))
        lap = jnp.tensordot(k_t_k_exp_ikx, self.A, axes=self.dim)
        return jnp.real(lap)
    
    # --- convenience batched wrappers ---
    @partial(jit, static_argnames=('self',))
    def batch_potential(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points"]:
        """Default batched potential using ``vmap`` over ``potential``."""
        return vmap(self.potential)(xs)

    @partial(jit, static_argnames=('self',))
    def batch_force(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points dim"]:
        """Default batched force using ``vmap`` over ``force``."""
        return vmap(self.force)(xs)

    @partial(jit, static_argnames=('self',))
    def batch_hessian(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points dim dim"]:
        """Default batched Hessian using ``vmap`` over ``hessian``."""
        return vmap(self.hessian)(xs)
    
    @partial(jit, static_argnames=('self',))
    def batch_laplacian(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points"]:
        """Default batched Laplacian using ``vmap`` over ``laplacian``."""
        return vmap(self.laplacian)(xs)

    # --- convenience grid evaluation ---
    @partial(jit, static_argnames=('self',))
    def potential_on_grid(self, xs: tuple[Float[Array, "n0"], ...]) -> Float[Array, "n0 ..."]:
        """
        Evaluate the random field on a grid defined by xs (tuple of 1D arrays).
        """
        exp_ikxs = [jnp.exp(1j * jnp.outer(ki, xi)) for xi, ki in zip(xs, self.ks)]
        phi = self.A
        for exp_ikx in exp_ikxs:
            phi = jnp.tensordot(phi, exp_ikx, axes=(0, 0))
        return jnp.real(phi)
    
    @partial(jit, static_argnames=('self',))
    def force_on_grid(self, xs: tuple[Float[Array, "n0"], ...]) -> Float[Array, "d n0 ..."]:
        """
        Evaluate the force of the random field on a grid defined by xs (tuple of 1D arrays).
        """
        k_mesh = jnp.array(jnp.meshgrid(*self.ks, indexing='ij'))
        exp_ikxs = [jnp.exp(1j * jnp.outer(ki, xi)) for xi, ki in zip(xs, self.ks)]
        ik_a = 1j * k_mesh * self.A
        for exp_ikx in exp_ikxs:
            ik_a = jnp.tensordot(ik_a, exp_ikx, axes=(1, 0))
        return -jnp.real(ik_a)
    
    @partial(jit, static_argnames=('self',))
    def hessian_det_on_grid(self, xs: tuple[Float[Array, "n0"], ...]) -> Float[Array, "n0 ..."]:
        """
        Evaluate the determinant of the Hessian of the random field on a grid defined by xs (tuple of 1D arrays).
        """
        k_mesh = jnp.array(jnp.meshgrid(*self.ks, indexing='ij'))
        exp_ikxs = [jnp.exp(1j * jnp.outer(ki, xi)) for xi, ki in zip(xs, self.ks)]
        kk_t_a = -jnp.expand_dims(k_mesh, axis=0) * jnp.expand_dims(k_mesh, axis=1) * self.A
        for exp_ikx in exp_ikxs:
            kk_t_a = jnp.tensordot(kk_t_a, exp_ikx, axes=(2, 0))
        return jnp.linalg.det(jnp.real(kk_t_a).transpose(*range(2, 2 + len(xs)), 0, 1))
    
    @partial(jit, static_argnames=('self',))
    def laplacian_on_grid(self, xs: tuple[Float[Array, "n0"], ...]) -> Float[Array, "n0 ..."]:
        """
        Evaluate the laplacian of the random field on a grid defined by xs (tuple of 1D arrays).
        """
        k_mesh = jnp.array(jnp.meshgrid(*self.ks, indexing='ij'))
        exp_ikxs = [jnp.exp(1j * jnp.outer(ki, xi)) for xi, ki in zip(xs, self.ks)]
        k_t_k_a = -jnp.sum(k_mesh ** 2, axis=0) * self.A
        for exp_ikx in exp_ikxs:
            k_t_k_a = jnp.tensordot(k_t_k_a, exp_ikx, axes=(0, 0))
        return jnp.real(k_t_k_a)

    @partial(jit, static_argnames=('self',))
    def qm_potential_on_grid(self, xs: tuple[Float[Array, "n0"], ...]) -> Float[Array, "n0 ..."]:
        """
        Evaluate the quantum-mechanical effective potential on a grid defined by xs (tuple of 1D arrays).
        """
        return jnp.sum(self.force_on_grid(xs) ** 2, axis=0) / 4 - self.laplacian_on_grid(xs) / 2

    # --- interpolation helpers ---
    @partial(jit, static_argnames=('self',))
    def potential_interp(self, x: Float[Array, "dim"]) -> Float:
        if self.phi_interpolator is None:
            raise ValueError("No potential interpolator available.")
        return self.phi_interpolator(*x.T)

    @partial(jit, static_argnames=('self',))
    def force_interp(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        if self.force_interpolators is None:
            raise ValueError("No force interpolators available.")
        return jnp.array([interp(*x.T) for interp in self.force_interpolators]).T

    # --- utilities ---
    @partial(jit, static_argnames=('self',))
    def fold_to_base_cell(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points dim"]:
        """Fold points to the base periodic cell defined by ``self.size``."""
        return (xs + self.size / 2) % self.size - self.size / 2
    
    @partial(jit, static_argnames=('self', "brute_force"))
    def discretize(self, xs: Float[Array, "n_points dim"], brute_force: Bool | None = None) -> Int[Array, "n_points"]: 
        valid_mask = jnp.isfinite(xs).all(axis=-1)
        xs = jnp.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)  # replace invalid points with a dummy value
        if self.dim == 1:
            @vmap
            def discretize_bf_1d(xs: Float[Array, "n_points dim"]) -> Int[Array, "n_points"]:
                xs = self.fold_to_base_cell(xs)  
                distances_sq = jnp.sum((xs[None, :] - self.discr_points_extended[:, :]) ** 2, axis=-1)
                return self.discr_indices_extended[jnp.argmin(distances_sq, axis=0)]
            
            def discretize_kd_1d(xs: Float[Array, "n_points dim"]) -> Int[Array, "n_points"]:
                return (jnp.searchsorted(0.5*(self.discr_points_extended[1:, 0] + self.discr_points_extended[:-1, 0]), jnp.ravel(xs)) - 1) % self.num_vert

            res = discretize_bf_1d(xs) if brute_force else discretize_kd_1d(xs)
            
        
        else:
            if brute_force == False:
                raise NotImplementedError("For KD-tree discretization use discretize_kd.")
            @partial(vmap)
            def discretize_bf(xs: Float[Array, "n_points dim"]) -> Int[Array, "n_points"]:
                xs = self.fold_to_base_cell(xs)  
                distances_sq = jnp.sum((xs[None, :] - self.discr_points_extended[:, :]) ** 2, axis=-1)
                return self.discr_indices_extended[jnp.argmin(distances_sq, axis=0)]
        
            res = discretize_bf(xs)
        res = jnp.nan_to_num(res, nan=self.num_vert, posinf=self.num_vert, neginf=self.num_vert)  # set invalid points
        valid_mask = valid_mask & (res < self.num_vert) & (res >= 0)
        return jnp.astype(jnp.where(valid_mask, res, self.num_vert), self.num_vert.dtype) # type: ignore # set invalid points
    
    def discretize_kd(self, xs: Float[Array, "n_points dim"]) -> Int[Array, "n_points"]:
        box = freud.box.Box(*self.size)
        valid_mask = jnp.isfinite(xs).all(axis=-1)
        xs = jnp.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)  # replace invalid points with a dummy value

        if self.dim == 2:
            crit_points_padded = jnp.concatenate((self.discr_points, jnp.zeros((self.num_vert,1))), axis=-1)
            neighbor_query = freud.locality.AABBQuery(box, crit_points_padded)
        
            assert xs.shape[-1] == 2, f"wrong dimension"
            res = jnp.array(neighbor_query.query(jnp.concatenate((self.fold_to_base_cell(xs),jnp.zeros((xs.shape[0],1))), axis=-1), query_args={"num_neighbors": 1}).toNeighborList().point_indices, dtype=jnp.int32)
        elif self.dim == 3:
            neighbor_query = freud.locality.AABBQuery(box, self.discr_points)
            res = jnp.array(neighbor_query.query(xs, query_args={"num_neighbors": 1}).toNeighborList().point_indices, dtype=jnp.int32)
        else:
            raise NotImplementedError("Neighbor query discretization not implemented for dim > 3")
        
        return jnp.astype(jnp.where(valid_mask, res, self.num_vert), self.num_vert.dtype) # type: ignore # set invalid points to -1


    # --- initialization helpers ---
    def _find_points(
        self,
        points_per_grid: float,
        max_steps: Int | None = None,
        atol: float | None = None,
        use_interpolation: bool = False,
        verbose: bool = True,
        discretization_points: str = "minima",
        dup_tol: float | None = None,
    ) -> Float[Array, "n_points dim"]:
        """Find and deduplicate minima/critical points from a seeded grid.

        Parameters
        - points_per_grid: target seeding density per characteristic cell.
        - discretization_points: either ``"minima"`` or ``"critical_points"``.

        Returns
        - unique periodic points sorted by ascending first coordinate.
        """
        if discretization_points not in ["minima", "critical_points"]:
            raise ValueError(
                f"Invalid discretization_points: {discretization_points}, must be one of 'minima' or 'critical_points'"
            )
        if points_per_grid <= 0:
            raise ValueError(f"points_per_grid must be positive, got {points_per_grid}.")

        n_grid_points = (self.size / self.lambda0 * points_per_grid ** (1 / self.dim)).astype(int)
        coordinate_ticks = [jnp.linspace(-s / 2, s / 2, n) for s, n in zip(self.size, n_grid_points)]
        initial_points = jnp.array(jnp.meshgrid(*coordinate_ticks, indexing='ij')).reshape(self.dim, -1).T

        if max_steps is None:
            max_steps = 10 * int(4 + self.dim)

        if atol is None:
            atol = max(float(0.02 ** self.dim * self.lambda0 / 5), 1e-5) * ( 1 + 9*(discretization_points == "critical_points"))
            if verbose:
                print(f"Using default heuristic atol = {atol:.2e} for tesselation")

        assert isinstance(atol, float) and atol > 0, "atol must be positive float"

        if discretization_points == "minima":
            if self.phi_interpolator is not None and use_interpolation:
                fn = lambda x, args: self.potential_interp(x)
            else:
                fn = lambda x, args: self.potential(x)

            @jit
            @vmap
            def converge_to_points(x0):
                sol = optx.minimise(fn=fn, solver=optx.DFP(atol=atol, rtol=0.), y0=x0, max_steps=max_steps, throw=False)
                return sol.value, sol.stats['num_steps']

        else:
            if self.force_interpolators is not None and use_interpolation:
                fn = lambda x, args: self.force_interp(x)
            else:
                fn = lambda x, args: self.force(x)

            @jit
            @vmap
            def converge_to_points(x0):
                sol = optx.root_find(fn=fn, solver=optx.Newton(rtol=0., atol=atol / 10), y0=x0, max_steps=max_steps, throw=False)
                return sol.value, sol.stats['num_steps']

        time_start = time.time()
        discr_points, num_steps = converge_to_points(initial_points)
        if verbose:
            print(
                f"{jnp.sum(num_steps < max_steps)} out of {len(initial_points)} seeds converged in "
                f"{time.time() - time_start:.2f} seconds, median number of steps: {jnp.median(num_steps)}"
            )

        if dup_tol is None:
            dup_tol = min(5e4 * atol, 0.5 * self.lambda0)
            if discretization_points == "critical_points":
                dup_tol = min(dup_tol, 0.3 * self.lambda0)  # critical points can be more densely packed, so use a smaller dup_tol
            if verbose:
                print(f"Using default dup_tol = {dup_tol:.2e} for duplicate removal")

        discr_points = discr_points[num_steps < max_steps]
        crit_indices = jnp.unique((self.fold_to_base_cell(discr_points) / dup_tol).astype(int), axis=0, return_index=True, equal_nan=True)[1]
        discr_points = discr_points[crit_indices]
        crit_indices = jnp.unique((self.fold_to_base_cell(discr_points + 0.5 * dup_tol) / dup_tol).astype(int), axis=0, return_index=True)[1]
        discr_points = self.fold_to_base_cell(discr_points[crit_indices])

        if verbose:
            print(f"Found {len(discr_points)} unique discretization points")
        if discretization_points == "minima":
            mask = jnp.linalg.det(self.batch_hessian(discr_points)) > 0
            discr_points = discr_points[mask]
            if verbose:
                print(f"discarded {jnp.sum(~mask)} candidate points due to non-positive Hessian determinant")

        return discr_points[jnp.argsort(discr_points[:, 0])]

    def _tesselate(self, points_per_cell: float = 5., atol: float | None = None, max_steps: Int | None = None , use_interpolation: bool = False, verbose: bool = True, discretization_points: str = "minima", border_threshold_multiplier: float = 0., edge_detection_threshold: float = 1e-5, dup_tol: float | None = None):
        if discretization_points == "grid":
            grid_dimensions = (self.size / self.lambda0 * points_per_cell ** (1 / self.dim)).astype(int)
            self.discr_points = jnp.array(jnp.meshgrid(*[jnp.linspace(-s / 2, s / 2, n) * (n - 1)/n for s, n in zip(self.size, grid_dimensions)], indexing='ij')).reshape(self.dim, -1).T
            if verbose:
                print(f"Constructed {len(self.discr_points)} discretization points on an {grid_dimensions} grid.")

        elif discretization_points in ["minima", "critical_points"]:
            self.discr_points = self._find_points(
                points_per_grid=points_per_cell,
                max_steps=max_steps,
                atol=atol,
                use_interpolation=use_interpolation,
                verbose=verbose,
                discretization_points=discretization_points,
                dup_tol=dup_tol,
            )
        else:
            raise ValueError(f"Invalid discretization_points: {discretization_points}, must be one of 'grid', 'minima' or 'critical_points'")
        
        num_vert = len(self.discr_points)
        if num_vert < 2**8:
            dtype = jnp.uint8
        elif num_vert < 2**16:
            dtype = jnp.uint16
        else:
            dtype = jnp.uint32

        self.num_vert = jnp.array(num_vert, dtype=dtype) 
        self.voro = freud.locality.Voronoi()

        if self.dim == 1:
            self.discr_points_extended = jnp.concatenate([self.discr_points[-1:] - self.size, self.discr_points, self.discr_points[0:1] + self.size], axis=0)
            self.discr_indices_extended = (jnp.arange(-1, jnp.int32(self.num_vert) + 1, dtype=jnp.int32) % self.num_vert).astype(dtype)

            if verbose:
                print(f"Extended to {len(self.discr_points_extended)} discretization points including periodic images.")
            
            time_start = time.time()
            
            self.tesselation = tesselation_1d(self.discr_points_extended, dtype=dtype)

            if verbose:
                print(f"Constructed tesselation in {time.time() - time_start:.2f} seconds.")
            return
        
        # for dims 2 and 3
        
        time_start = time.time()
        box = freud.box.Box(*self.size)
        if self.dim == 2:
            crit_points_padded = jnp.concatenate((self.discr_points, jnp.zeros((self.num_vert,1))), axis=-1)
            self.voro.compute((box, crit_points_padded))
            polytopes = [jnp.array(poly[:,:2]) for poly in self.voro.polytopes]
        elif self.dim == 3:
            self.voro.compute((box, self.discr_points))
            polytopes = [jnp.array(poly) for poly in self.voro.polytopes]
        else:
            raise NotImplementedError("Voronoi tesselation not implemented for dim > 3")

        if verbose:
            print(f"Voronoi tesselation computed in {time.time() - time_start:.2f} seconds. Constructing cells...")
        time_start = time.time()
        nlist = self.voro.nlist
        num_poly_verts = jnp.array([len(poly) for poly in polytopes])
        degrees = nlist.neighbor_counts
        polytope_vertices_all = np.full((self.num_vert, jnp.max(num_poly_verts), self.dim), jnp.nan, dtype=np.float32)
        neighbor_indices_all = np.full((self.num_vert, jnp.max(degrees)), jnp.iinfo(dtype).max, dtype=dtype) # overflow
        vectors_all = np.full((self.num_vert, jnp.max(degrees), self.dim), jnp.nan, dtype=np.float32)
        for i, poly in enumerate(polytopes):
            mask = nlist[:,0] == i
            neighbor_indices = nlist[mask][:,1]
            vectors = nlist.vectors[mask][:, :self.dim]
            polytope_vertices_all[i, :len(poly), :] = poly
            neighbor_indices_all[i, :len(neighbor_indices)] = neighbor_indices
            vectors_all[i, :len(neighbor_indices), :] = vectors

        if self.dim == 2:
            self.tesselation = tesselation_2d(self.discr_points, neighbor_indices_all, vectors_all, polytope_vertices_all, degrees, edge_detection_threshold=edge_detection_threshold)
        if self.dim == 3:
            self.tesselation = tesselation_3d(self.discr_points, neighbor_indices_all, vectors_all, polytope_vertices_all, num_poly_verts, degrees, edge_detection_threshold=edge_detection_threshold)

        if verbose:
            print(f"Constructed tesselation in {time.time() - time_start:.2f} seconds.")

        # construct image points for border cells to ensure correct neighbor detection and cell construction at boundaries
        time_start = time.time()
        if discretization_points == "grid":
            border_threshold = 1.2 * (self.discr_points[1,0] - self.discr_points[0,0]) # for grid discretization, use the grid spacing as border threshold
        border_threshold = (jnp.prod(self.size)/self.num_vert) ** (1/self.dim) * border_threshold_multiplier

        shifts = jnp.array(jnp.meshgrid(*([jnp.arange(-1, 2, dtype=jnp.float32)] * self.dim), indexing='ij')).reshape(self.dim, -1).T
        discr_points_extended = jnp.repeat(self.discr_points, shifts.shape[0], axis=0) + jnp.tile(shifts * self.size, (self.num_vert, 1))
        discr_indices_extended = jnp.repeat(jnp.arange(self.num_vert), shifts.shape[0], axis=0)
        mask = jnp.all(jnp.abs(discr_points_extended) < (self.size / 2 + border_threshold), axis=-1)
        self.discr_points_extended = discr_points_extended[mask]
        self.discr_indices_extended = discr_indices_extended[mask]
        
        if verbose:
            print(f"Extended to {len(self.discr_points_extended)} discretization points including periodic images in {time.time() - time_start:.2f} s.")
        
    def _dont_tesselate(self):
        self.discr_points = jnp.zeros((0,self.dim), dtype=jnp.float32)
        self.discr_points_extended = self.discr_points
        self.discr_indices_extended = jnp.zeros((0,), dtype=jnp.int32)
        self.num_vert = 0
        self.voro = None
        self.tesselation = None

    def _clear_stepfunction(self):
        self.lower_step = None
        self.lower_step_margin = None
        self.lower_step_bins = None
        self.lower_step_side = None
        self.lower_step_origin = None

    def _initialize_stepfunction(self, step_side_length: Float | None = None, verbose: bool = True):
        if step_side_length is None:
            step_side_length = float(self.lambda0) / int((1000) ** (1/self.dim)) 
        step_side_length = float(step_side_length)
        if step_side_length <= 0:
            raise ValueError(f"step_side_length must be positive, got {step_side_length}.")

        bins = tuple(max(1, int(jnp.ceil(float(s) / step_side_length))) for s in self.size)
        step_side = self.size / jnp.array(bins, dtype=jnp.float32)
        step_origin = -self.size / 2

        n_points = int(np.prod(np.array(bins, dtype=np.int64)))
        if n_points <= 0:
            raise ValueError("Stepfunction grid must contain at least one hypercube.")

        center_axes = tuple(step_origin[d] + (jnp.arange(bins[d], dtype=jnp.float32) + 0.5) * step_side[d] for d in range(self.dim))
        phi_grid = self.potential_on_grid(center_axes)
        force_grid = self.force_on_grid(center_axes)
        grad_norm = jnp.linalg.norm(force_grid, axis=0)
        neighbor_grad_norms = [
            jnp.roll(grad_norm, shift=shift, axis=ax)
            for ax in range(self.dim)
            for shift in (-1, 1)
        ]
        local_max_grad = jnp.max(jnp.stack([grad_norm, *neighbor_grad_norms], axis=0), axis=0)
        box_length = jnp.mean(step_side)
        # Conservative margin: 2.0x sqrt(dim) + 0.2x mean gradient to handle high-amplitude fields and off-grid dips
        pointwise_margin = 2.0 * jnp.sqrt(self.dim) * box_length * (local_max_grad + 0.5 * jnp.mean(grad_norm))
        lower_step = phi_grid - pointwise_margin

        self.lower_step = lower_step
        self.lower_step_margin = jnp.log(jnp.mean(jnp.exp(pointwise_margin)))
        self.lower_step_bins = bins
        self.lower_step_side = step_side
        self.lower_step_origin = step_origin
        if verbose:
            print(f"Initialized lower step surrogate on grid {bins} (n={n_points}, side~{float(step_side_length):.3g}).")

    def _initialize_interpolators_and_stationary_dist(self, interpolate: bool = True, phi_interp_args: PyTree = {"res": 10., "method": "cubic2"}, 
                                                      force_interp_args: PyTree = {"res": 10., "method": "cubic2"}, verbose: bool = True, stationary_dist_npoints: int = 100000, n_moments: int = 10, seed: Int = 42):
        
        interpolator_class = {1: interpax.Interpolator1D, 2: interpax.Interpolator2D, 3: interpax.Interpolator3D}.get(self.dim, None)

        if interpolator_class is not None and interpolate:
            start_time = time.time()
            xs = tuple(jnp.linspace(-s / 2, s / 2, int(phi_interp_args.get("res", 10.) * s * self.k0 + 1)) for s in self.size)
            phis = self.potential_on_grid(xs)
            self.phi_interpolator = interpolator_class(*xs, phis, method=phi_interp_args.get("method", "cubic2"), period=self.size)

            xs = tuple(jnp.linspace(-s / 2, s / 2, int(force_interp_args.get("res", 10.) * s * self.k0 + 1)) for s in self.size)
            forces = self.force_on_grid(xs)
            self.force_interpolators = tuple(
                interpolator_class(*xs, forces[i], method=force_interp_args.get("method", "cubic2"), period=self.size) for i in range(self.dim)
            )
            if verbose:
                print(f"Interpolation on grids of sizes {phis.shape}, {forces.shape} setup took {time.time() - start_time:.2f} seconds")

        else:
            self.phi_interpolator = None
            self.force_interpolators = None
            xs = jr.uniform(key=jr.key(0), shape=(stationary_dist_npoints, self.dim), minval=-0.5, maxval=0.5) * self.size
            phis = self.batch_potential(xs)
        mu = jnp.mean(phis)
        self.phi_min = jnp.min(phis) - 10.* jnp.sqrt(jnp.var(phis))/jnp.sqrt(phis.size) # safety margin
        self.phi_moments = jnp.array([mu] + [jnp.mean((phis - mu) ** n) for n in range(2, n_moments + 1)])
        self.normalization_constant = jnp.mean(jnp.exp(self.phi_min - phis))

    # --- sampling from stationary distribution ---
    def _generate_and_reject_pure(self, key: Array, batch_n: Int, beta: Float, verbose: bool = False) -> tuple[Array, Array]:
        new_key, key = jr.split(key)
        position_key, randomness_key = jr.split(new_key)
        new_samples = jr.uniform(key=position_key, shape=(batch_n, self.dim), minval=-0.5, maxval=0.5) * self.size
        randomness = jr.uniform(key=randomness_key, shape=(batch_n,))
        thresholds = jnp.exp(-beta * (self.batch_potential(new_samples) - self.phi_min))
        accepted_samples = new_samples[randomness < thresholds]
        if verbose:
            print(f"pure sampler batch: accepted {len(accepted_samples)} / {batch_n}")
        return accepted_samples, key

    def _generate_and_reject_step(self, key: Array, batch_n: Int, beta: Float, verbose: bool = False) -> tuple[Array, Array]:
        if self.lower_step is None or self.lower_step_side is None or self.lower_step_origin is None or self.lower_step_margin is None:
            raise ValueError("Stepfunction surrogate is not initialized. Set initialize_stepfunction=True during Field construction.")

        lower_step_flat = jnp.ravel(self.lower_step)
        logits = -beta * lower_step_flat
        logits = logits - jnp.max(logits)
        probs = jnp.exp(logits)
        probs = probs / jnp.sum(probs)

        extra_margin = 0.0
        violations_detected = 0
        if verbose:
            print(f"step sampler: starting violation retry loop (max 16 iterations)")
        for retry_iter in range(16):
            new_key, key = jr.split(key)
            cube_key, local_key, randomness_key = jr.split(new_key, 3)
            cube_ids = jr.choice(cube_key, lower_step_flat.size, shape=(batch_n,), replace=True, p=probs)
            cube_indices = jnp.stack(jnp.unravel_index(cube_ids, self.lower_step.shape), axis=1)
            local_offsets = jr.uniform(local_key, shape=(batch_n, self.dim), minval=0., maxval=1.)
            candidates = self.lower_step_origin + cube_indices * self.lower_step_side + local_offsets * self.lower_step_side

            phi_values = self.batch_potential(candidates)
            lower_values = lower_step_flat[cube_ids] - extra_margin
            violation = jnp.max(jnp.maximum(lower_values - phi_values, 0.0))
            if float(violation) > 0.0:
                violations_detected += 1
                margin_increase = 1.05 * float(violation)
                extra_margin += margin_increase
                if verbose:
                    print(f"  retry {retry_iter+1}/16: violation={float(violation):.3e}, increased margin by {margin_increase:.3e}, cumulative extra_margin={extra_margin:.3e}")
                continue

            if verbose:
                if violations_detected > 0:
                    print(f"  valid batch found on retry {retry_iter+1} after {violations_detected} violations, final extra_margin={extra_margin:.3e}")
                else:
                    print(f"  valid batch found on retry {retry_iter+1}, no violations")
            thresholds = jnp.exp(-beta * (phi_values - lower_values))
            randomness = jr.uniform(randomness_key, shape=(batch_n,))
            accepted_samples = candidates[randomness < thresholds]
            if verbose:
                print(f"step sampler batch: accepted {len(accepted_samples)} / {batch_n}")
            return accepted_samples, key

        raise RuntimeError(f"Step sampler could not produce a valid batch after 16 margin recalibration attempts (detected {violations_detected} violations total).")

    def _sample_stationary_distribution_common(
        self,
        key: Array,
        n_samples: Int,
        beta: Float,
        verbose: bool,
        max_it: Int,
        max_batchsize: Int | None,
        acceptance_est: float,
        generate_and_reject_fn: Callable[[Array, Int, Float, bool], tuple[Array, Array]],
        acceptance_label: str,
    ) -> Float[Array, "n_samples dim"]:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}.")

        if max_batchsize is None:
            # Use a conservative batch size: 100k for step sampler (due to retry loop overhead)
            max_batchsize = 2**21
            if verbose:
                print(f"{acceptance_label}: using default max_batchsize={max_batchsize}")

        n_samples_left = int(n_samples)
        samplelist = []
        for outer_it in range(max_it):
            n_samples_with_margin = max(1, int(((n_samples_left + jnp.sqrt(n_samples_left)) / max(acceptance_est, 1e-3)).astype(int)))
            proposal_batch_size = min(int(max_batchsize), n_samples_with_margin)
            n_generated = 0
            n_accepted_this_round = 0
            if verbose:
                print(
                    f"{acceptance_label} outer {outer_it + 1}/{max_it}: need {n_samples_left}, "
                    f"estimate={acceptance_est:.3e}, proposing {n_samples_with_margin} in batches of {proposal_batch_size}"
                )
            while n_generated < n_samples_with_margin:
                batch_n = min(proposal_batch_size, n_samples_with_margin - n_generated)
                if verbose:
                    print(f"{acceptance_label} generating batch of {batch_n}")
                accepted_samples, key = generate_and_reject_fn(key, batch_n, beta, verbose)
                if verbose:
                    acceptance_rate = len(accepted_samples) / batch_n
                    print(f"{acceptance_label} kept {len(accepted_samples)} / {batch_n} ({acceptance_rate:.3%}), need {n_samples}")
                n_samples_left -= len(accepted_samples)
                n_accepted_this_round += len(accepted_samples)
                samplelist.append(accepted_samples)
                if n_samples_left <= 0:
                    samples = jnp.concatenate(samplelist, axis=0)
                    return samples[:n_samples]
                n_generated += batch_n

            acceptance_est = max(1e-3, n_accepted_this_round / max(n_generated, 1))

        raise RuntimeError(f"Could not generate enough samples after {max_it} iterations, still need {n_samples_left} samples.")

    def _sample_stationary_distribution_step(self, key: Array = jr.key(42), n_samples: Int = 1000, beta: Float = 1., verbose: bool = False, max_it: Int = 3, max_batchsize: Int | None = None) -> Float[Array, "n_samples dim"]:
        if self.lower_step is None or self.lower_step_margin is None:
            raise ValueError("Stepfunction surrogate is not initialized. Set initialize_stepfunction=True during Field construction.")

        acceptance_est = float(jnp.exp(-beta * self.lower_step_margin))
        return self._sample_stationary_distribution_common(
            key=key,
            n_samples=n_samples,
            beta=beta,
            verbose=verbose,
            max_it=max_it,
            max_batchsize=max_batchsize,
            acceptance_est=acceptance_est,
            generate_and_reject_fn=self._generate_and_reject_step,
            acceptance_label="step sampler",
        )

    def sample_stationary_distribution(self, key: Array = jr.key(42), n_samples: Int = 1000, beta: Float = 1., verbose=False, max_it=3, max_batchsize: Int | None = None, use_step: bool = True) -> Float[Array, "n_samples dim"]:
        """Sample initial positions from the stationary distribution."""
        if jnp.abs(beta-1.) < 1e-6:
            acceptance_est = self.normalization_constant
        else:
            exponents = jnp.arange(2, len(self.phi_moments) + 1)
            acceptance_est = jnp.exp(- beta * (self.phi_moments[0] - self.phi_min)) * (1 + jnp.sum((-beta)**exponents * self.phi_moments[1:] / jnp.cumprod(exponents))) # rough approximation
            if verbose and not use_step:
                print(f"Using approximate normalization constant {acceptance_est:.3%}")

        if use_step:
            if self.lower_step is None or self.lower_step_margin is None:
                raise ValueError("Stepfunction surrogate is not initialized. Set initialize_stepfunction=True during Field construction.")
            acceptance_est = float(jnp.exp(-beta * self.lower_step_margin))
            return self._sample_stationary_distribution_common(
                key=key,
                n_samples=n_samples,
                beta=beta,
                verbose=verbose,
                max_it=max_it,
                max_batchsize=max_batchsize,
                acceptance_est=acceptance_est,
                generate_and_reject_fn=self._generate_and_reject_step,
                acceptance_label="step sampler",
            )

        return self._sample_stationary_distribution_common(
            key=key,
            n_samples=n_samples,
            beta=beta,
            verbose=verbose,
            max_it=max_it,
            max_batchsize=max_batchsize,
            acceptance_est=float(acceptance_est),
            generate_and_reject_fn=self._generate_and_reject_pure,
            acceptance_label="pure sampler",
        )

    def get_discrete_stationary_distribution(self, res: Float = 50.) -> Float[Array, "num_vert"]:
        """Approximate the stationary distribution on Voronoi cells via grid quadrature."""
        if self.tesselation is None:
            raise ValueError("Tesselation must be defined before computing a discrete stationary distribution.")
        if float(res) <= 0:
            raise ValueError(f"res must be positive, got {res}.")

        num_vert_int = int(self.num_vert)
        if num_vert_int <= 0:
            raise ValueError("num_vert must be positive.")

        xs = tuple(
            jnp.linspace(-float(s) / 2, float(s) / 2, int(float(res) * float(s)) + 1, endpoint=False)
            for s in self.size
        )
        phi = self.potential_on_grid(xs)
        x_grids = jnp.meshgrid(*xs, indexing='ij')
        x_list = jnp.array(x_grids).reshape(self.dim, -1).T
        cell_ids = self.discretize(x_list)

        weights = jnp.exp(-jnp.ravel(phi))
        weighted_counts = jnp.bincount(
            cell_ids,
            weights=weights,
            length=num_vert_int
        )

        dxs = jnp.array([x[1] - x[0] for x in xs])
        cell_integrals = weighted_counts * jnp.prod(dxs)
        z = jnp.sum(cell_integrals)
        if z <= 0:
            raise ValueError("Failed to compute a positive normalization constant for the discrete stationary distribution.")
        return cell_integrals / z
    
    # --- tests ---
    def test_stationary_distribution(self, n_samples: Int | None = None, beta: Float = 1., seed: Int = 42, res=30., verbose: bool = True, plot: bool = True, alpha=0.2, ax: Axes | None = None, use_step: bool = False):
        if n_samples is None:
            n_samples = min(int(jnp.prod(1e1 * res * self.size)), 500000)
            if verbose:
                print(f"Using n_samples = {n_samples}")
        key = jr.key(seed)
        samples = self.sample_stationary_distribution(key=key, n_samples=n_samples, beta=beta, verbose=verbose, use_step=use_step)
        
        # build histogram
        hist_bins = [jnp.linspace(-s / 2, s / 2, int(res * s + 1)) for s in self.size]
        hist, edges = jnp.histogramdd(samples, bins=hist_bins)
        xs = [0.5 * (edges[i][:-1] + edges[i][1:]) for i in range(self.dim)]
        Phi = self.potential_on_grid(xs)

        # theoretical stationary distribution
        dvol = jnp.prod(jnp.array([edges[i][1] - edges[i][0] for i in range(self.dim)]))
        Z = jnp.sum(jnp.exp(-beta * Phi))
        p_theory = jnp.exp(-beta * Phi) / Z

        if plot:
            plot_stationary_distribution_test(hist=hist, p_theory=p_theory, n_samples=n_samples, alpha=alpha, ax=ax)

        # chi2 test
        chi2 = jnp.sum((hist - p_theory * n_samples) ** 2 / (p_theory * n_samples))
        if verbose:
            print(f"Chi2 test statistic: {chi2:.2f}, p-value: {js.chi2.sf(chi2, df=hist.size - 1):.2e}")
    
    def test_interpolations(self, batch_size: Int = 10000, seed: Int = 42):
        key = jr.key(seed)
        xs = jnp.stack([jr.uniform(key, (batch_size,)) * s - s / 2 for s in self.size], axis=1)
        print("Testing interpolations...")
        absolute_errors = jnp.abs(self.batch_potential(xs) - self.potential_interp(xs))
        pot_mae = jnp.mean(absolute_errors)
        pot_maxae = jnp.max(absolute_errors)
        mean_amplitude = jnp.mean(jnp.abs(self.batch_potential(xs)))
        pot_relmae = pot_mae/mean_amplitude
        pot_relmaxae = pot_maxae/mean_amplitude
        time1 = timeit("self.batch_potential(xs)", globals={"self": self, "xs": xs}, number=100)
        time2 = timeit("self.potential_interp(xs)", globals={"self": self, "xs": xs}, number=100)
        print(f"Potential: Absolute Error: mean {pot_mae:.2e}, ({pot_relmae:.2e} relative), max {pot_maxae:.2e}, Speedup: {time1/time2:.2e}")
        
        absolute_errors = jnp.abs(self.batch_force(xs) - self.force_interp(xs))
        force_mae = jnp.mean(absolute_errors)
        force_maxae = jnp.max(absolute_errors)
        force_relmae = force_mae/jnp.mean(jnp.abs(self.batch_force(xs)))
        time1 = timeit("self.batch_force(xs)", globals={"self": self, "xs": xs}, number=100)
        time2 = timeit("self.force_interp(xs)", globals={"self": self, "xs": xs}, number=100)
        print(f"Force: Absolute Error: mean {force_mae:.2e}, ({force_relmae:.2e} relative), max {force_maxae:.2e}, Speedup: {time1/time2:.2e}")

    def test_discretization(self, n_samples: Int = 100, n_it: Int = 100, seed: Int = 42):
            # testing discretization point extension:
            folded_points = self.fold_to_base_cell(self.discr_points_extended)
            assert jnp.allclose(folded_points, self.discr_points[self.discr_indices_extended]), "fold_to_base_cell does not correctly fold extended discretization points back to original discretization points"
            print("extendsion of discretization points and indices seems correct.")

            key = jr.key(seed)
            xs = jr.uniform(key, (n_samples, self.dim), minval=-0.5, maxval=0.5) * self.size
            print("compute brute force")
            indices_brute = self.discretize(xs, brute_force=True)
            print("compute kd-tree / searchsorted")
            indices_kd_tree = self.discretize_kd(xs) if self.dim > 1 else self.discretize(xs, brute_force=False)
            
            time_brute = timeit("self.discretize(xs, brute_force=True)", globals={"self": self, "xs": xs}, number=n_it)
            if self.dim == 1:
                time_kd_tree = timeit("self.discretize(xs, brute_force=False)", globals={"self": self, "xs": xs}, number=n_it)
            else:
                time_kd_tree = timeit("self.discretize_kd(xs)", globals={"self": self, "xs": xs}, number=n_it)
            accuracy = jnp.mean(indices_brute == indices_kd_tree)
            assert accuracy > 0.999, f"Discretization agreement between both methods is too low: {accuracy:.3%}"

            print(f"Discretization accuracy: {accuracy:.5%} ({jnp.sum(indices_brute == indices_kd_tree)} out of {n_samples} samples)")
            print(f"Brute force: {time_brute/n_it:.6g} seconds, kd_tree query/searchsorted time: {time_kd_tree/n_it:.6g} seconds, Speedup: {time_brute/time_kd_tree - 1:.2f}")
            
            assert self.tesselation is not None, "Tesselation not constructed, cannot test cell volume consistency."
            cell_volumes = self.tesselation.volumes
            total_volume = jnp.prod(self.size)
            print(f"Sum of cell volumes {jnp.sum(cell_volumes):.2f}, total box volume {total_volume:.2f}, absolute error: {jnp.sum(cell_volumes) - total_volume:.2e}")
            assert jnp.isclose(jnp.sum(cell_volumes), total_volume), f"Sum of cell volumes does not match total box volume, check cell construction."    
    # === plotting ===

    def plot_field(self, ax_x: Axes | None = None, ax_k: Axes | None = None, res: Float | None = None, plot_dim: Int | None = None, threshold: Float = 0.0003, n_levels: Int = 5, show_cells: Bool | None = None, cell_plot_args: PyTree = {}, verbose: Bool = True, flip_potential_axes: Bool = False, bounds: Float[Array, "dim 2"] | None = None, pad: Float = 0.04, slider_k_init: float = 0.0, slider_x_init: float = 0.0, slider_y_init: float = 0.0, slider_z_init: float = 0.0, slider_mag_init: float | None = None, sliders: bool = True, transformation: str | None = None, bw: bool = False):  
        return plot_field(
            self,
            ax_x=ax_x,
            ax_k=ax_k,
            res=res,
            plot_dim=plot_dim,
            threshold=threshold,
            n_levels=n_levels,
            show_cells=show_cells,
            cell_plot_args=cell_plot_args,
            verbose=verbose,
            flip_potential_axes=flip_potential_axes,
            bounds=bounds,
            pad=pad,
            slider_k_init=slider_k_init,
            slider_x_init=slider_x_init,
            slider_y_init=slider_y_init,
            slider_z_init=slider_z_init,
            slider_mag_init=slider_mag_init,
            sliders=sliders,
            transformation=transformation,
            bw=bw,
        )

    def plot_stationary_distribution_sample(self, n_points: Int = 10000, scatter_args: PyTree = {"c": "C5", "alpha": 0.2, "s": 2}, beta: Float = 1., res: Int = 30., seed: Int = 42, ax: Axes | None = None):
        return plot_stationary_distribution_sample(self, n_points=n_points, scatter_args=scatter_args, beta=beta, res=res, seed=seed, ax=ax)

    # === saving and loading ===

    def to_pickle_hex(self) -> str:
        """
        Serializes the Field to a hex string. 
        Removes unpickleable C++ objects (voro) before saving.
        """
        # 1. Strip the unpickleable freud object
        # We use tree_at to create a temporary copy with voro = None
        field_to_save = eqx.tree_at(lambda f: f.voro, self, None, is_leaf=lambda x: x is None)
        
        # 2. Serialize to bytes
        return pickle.dumps(field_to_save).hex()




class RandomField(Field):

    def __init__(self, size: Float[Array, "dim"] = jnp.array([3., 3.]),
                 k0: Float = 2 * jnp.pi, 
                 delta_k: Float = 0.1 * jnp.pi, 
                 seed: Int = 42,
                 initialize_stepfunction: bool = True,
                 step_side_length: Float | None = None,
                 tesselate: bool = True,
                 tesselation_args: PyTree = {"points_per_cell": 10., "atol": None, "max_steps": None, 'border_threshold_multiplier': 0.8, "use_interpolation": False, "discretization_points": "minima"},
                 interpolate = False,
                 phi_interp_args: PyTree = {"method": "cubic2", "res": 10.},
                 force_interp_args: PyTree = {"method": "cubic2", "res": 10.},
                 rms_amplitude: float = 1.,
                 continuum_normalization: bool = False,
                 verbose: bool = True):
        
        self.size = size
        self.dim = len(size)
        self.k0 = k0
        self.lambda0 = 2 * jnp.pi / k0

        # k-space mode counts (ensure integer)
        n_modes = tuple(int(jnp.ceil((k0 + 5 * delta_k) / (2 * jnp.pi) * s)) for s in size)
        self.ks = tuple(jnp.arange(-n_m, n_m + 1) * 2 * jnp.pi / s for n_m, s in zip(n_modes, size))

        # build random amplitude A on k-grid (complex Gaussian) with given spectrum envelope
        R = jnp.sqrt(sum(jnp.meshgrid(*[k ** 2 for k in self.ks], indexing='ij')))
        EA = jnp.exp(-(R - k0) ** 2 / (4 * delta_k ** 2))

        if continuum_normalization:
            gauss_integral = jnp.sqrt(2 * jnp.pi) * delta_k
            sphere_area =  2 * jnp.pi**(self.dim / 2) / jax.scipy.special.gamma(self.dim / 2) * k0 ** (self.dim - 1)
            dkx = [k[1] - k[0] for k in self.ks]
            mode_density = 1 / jnp.prod(jnp.array(dkx))
            EA /= jnp.sqrt(gauss_integral * mode_density * sphere_area)
        else:
            EA /= jnp.sqrt(jnp.sum(EA**2)) / rms_amplitude

        self.EA = EA

        normals = jr.normal(key=jr.key(seed), shape=EA.shape, dtype=jnp.complex64)
        hermitian_normals = (normals + jnp.flip(jnp.conj(normals))) / jnp.sqrt(2)
        self.A = EA.astype(jnp.complex64) * hermitian_normals

        self._initialize_interpolators_and_stationary_dist(interpolate=interpolate, verbose=verbose, 
                                                           phi_interp_args=phi_interp_args, 
                                                           force_interp_args=force_interp_args)
        if initialize_stepfunction:
            self._initialize_stepfunction(step_side_length=step_side_length, verbose=verbose)
        else:
            self._clear_stepfunction()
        if tesselate:
            self._tesselate(**tesselation_args, verbose=verbose)
        else:
            self._dont_tesselate()
        
        if verbose:
            print(f"Σ|A|^2 = E[Re(Phi)^2] = {jnp.sum(jnp.abs(self.A)**2):3f},  Σ |EA|^2 = {jnp.sum(self.EA**2):3f}")


class CustomImageField(Field):
    image_file: str
    image_bw: Float[Array, "h w"]

    def __init__(self, image_file: str,
                 width: Float = 160.,
                 k0: Float = 2 * jnp.pi,
                 delta_k: Float = 0.1 * jnp.pi,
                 seed: Int = 42,
                 initialize_stepfunction: bool = True,
                 step_side_length: Float | None = None,
                 tesselate: bool = True,
                 tesselation_args: PyTree = {"points_per_cell": 50., "atol": None, "max_steps": None, "border_threshold_multiplier": 0.8, "use_interpolation": False, "discretization_points": "minima"},
                 interpolate: bool = False,
                 phi_interp_args: PyTree = {"method": "cubic2", "res": 10.},
                 force_interp_args: PyTree = {"method": "cubic2", "res": 10.},
                 rms_amplitude: float = 1.,
                 continuum_normalization: bool = False,
                 amplitude_regularization: float = 0.8,
                 sigma: float = 0.2,
                 rectification_range: tuple[float, float] | list[float] = (0.2, 0.8),
                 kmax: float = 3. * jnp.pi,
                 debug: bool = True,
                 debug_res: int = 128,
                 verbose: bool = True):

        if width <= 0:
            raise ValueError(f"width must be positive, got {width}.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        if amplitude_regularization < 0:
            raise ValueError(f"amplitude_regularization must be non-negative, got {amplitude_regularization}.")
        if kmax <= 0:
            raise ValueError(f"kmax must be positive, got {kmax}.")
        if len(rectification_range) != 2:
            raise ValueError("rectification_range must contain exactly two values.")

        rect_low = float(rectification_range[0])
        rect_high = float(rectification_range[1])
        if not (0. <= rect_low < rect_high <= 1.):
            raise ValueError(f"rectification_range must satisfy 0 <= low < high <= 1, got {rectification_range}.")

        self.image_file = image_file
        image_bw = self._load_and_prepare_image(image_file=image_file)
        self.image_bw = image_bw

        h, w = image_bw.shape
        aspect_ratio = float(h) / float(w)
        self.dim = 2
        self.size = jnp.array([float(width), float(width) * aspect_ratio], dtype=jnp.float32)
        self.k0 = k0
        self.lambda0 = 2 * jnp.pi / k0

        n_modes = tuple(int(jnp.ceil(kmax / (2 * jnp.pi) * s)) for s in self.size)
        self.ks = tuple(jnp.arange(-n_m, n_m + 1) * 2 * jnp.pi / s for n_m, s in zip(n_modes, self.size))

        r_k = jnp.sqrt(sum(jnp.meshgrid(*[k ** 2 for k in self.ks], indexing='ij')))
        ea = jnp.exp(-(r_k - k0) ** 2 / (4 * delta_k ** 2))

        if continuum_normalization:
            gauss_integral = jnp.sqrt(2 * jnp.pi) * delta_k
            sphere_area = 2 * jnp.pi ** (self.dim / 2) / jax.scipy.special.gamma(self.dim / 2) * k0 ** (self.dim - 1)
            dkx = [k[1] - k[0] for k in self.ks]
            mode_density = 1 / jnp.prod(jnp.array(dkx))
            ea /= jnp.sqrt(gauss_integral * mode_density * sphere_area)
        else:
            ea /= jnp.sqrt(jnp.sum(ea ** 2)) / rms_amplitude

        self.EA = ea

        normals = jr.normal(key=jr.key(seed), shape=ea.shape, dtype=jnp.complex64)
        hermitian_normals = (normals + jnp.flip(jnp.conj(normals))) / jnp.sqrt(2)
        self.A = ea.astype(jnp.complex64) * hermitian_normals
        self.phi_interpolator = None
        self.force_interpolators = None
        self.voro = None
        self.tesselation = None
        self.discr_points = jnp.zeros((0, self.dim), dtype=jnp.float32)
        self.discr_points_extended = self.discr_points
        self.discr_indices_extended = jnp.zeros((0,), dtype=jnp.int32)
        self.num_vert = jnp.array(0, dtype=jnp.uint32)

        fig = None
        axs = None
        debug_res_eff = int(2 * debug_res)
        xs_dbg: tuple[Float[Array, "n0"], ...] | None = None
        if debug:
            fig, axs = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
            axs = np.array(axs).reshape(-1)
            xs_dbg = tuple(jnp.linspace(-float(s) / 2, float(s) / 2, debug_res_eff, endpoint=False) for s in self.size)
            phi0 = self.evaluate_on_grid(xs_dbg)
            vmax0 = float(jnp.max(jnp.abs(phi0)))
            im0 = axs[0].imshow(np.array(phi0).T, origin='lower', cmap='bwr', vmin=-vmax0, vmax=vmax0, extent=(-float(self.size[0]) / 2, float(self.size[0]) / 2, -float(self.size[1]) / 2, float(self.size[1]) / 2), aspect='equal')
            axs[0].set_title("(a)", pad=-10)
            if fig is not None:
                fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        image_rectified = rect_low + (rect_high - rect_low) * (1. - image_bw)

        points_per_grid = float(tesselation_args.get("points_per_cell", 10.))
        max_steps = tesselation_args.get("max_steps", None)
        atol = tesselation_args.get("atol", None)
        use_interpolation = bool(tesselation_args.get("use_interpolation", False))
        dup_tol = tesselation_args.get("dup_tol", None)
        radius = float(sigma * self.lambda0)

        n_fft = tuple(2 * n_m + 1 for n_m in n_modes)
        xs_fft = tuple(jnp.linspace(0., float(s), n, endpoint=False) for s, n in zip(self.size, n_fft))
        grids_fft = jnp.meshgrid(*xs_fft, indexing='ij')
        points_fft = jnp.array(grids_fft).reshape(self.dim, -1).T

        if amplitude_regularization > 0:
            critical_points_reg = self._find_points(
                points_per_grid=points_per_grid,
                max_steps=max_steps,
                atol=atol,
                use_interpolation=use_interpolation,
                verbose=verbose,
                discretization_points="critical_points",
                dup_tol=dup_tol,
            )

            hessians_reg = self.batch_hessian(critical_points_reg)
            det_h_reg = jnp.linalg.det(hessians_reg)
            tr_h_reg = jnp.trace(hessians_reg, axis1=-2, axis2=-1)
            minima_mask_reg = (det_h_reg > 0) & (tr_h_reg > 0)
            maxima_mask_reg = (det_h_reg > 0) & (tr_h_reg < 0)

            extrema_points_reg = jnp.concatenate(
                [critical_points_reg[minima_mask_reg], critical_points_reg[maxima_mask_reg]],
                axis=0,
            )
            n_min_reg = int(jnp.sum(minima_mask_reg))
            n_max_reg = int(jnp.sum(maxima_mask_reg))

            if n_min_reg + n_max_reg > 0:
                extrema_vals_reg = self.batch_potential(extrema_points_reg)
                extrema_targets_reg = jnp.concatenate(
                    [
                        -2.0 * rms_amplitude * jnp.ones((n_min_reg,), dtype=jnp.float32),
                        2.0 * rms_amplitude * jnp.ones((n_max_reg,), dtype=jnp.float32),
                    ],
                    axis=0,
                )
                desired_shift_reg = amplitude_regularization * (extrema_targets_reg - extrema_vals_reg)

                extrema_points_reg_np = np.array(extrema_points_reg, dtype=np.float32)
                desired_shift_reg_np = np.array(desired_shift_reg, dtype=np.float64)
                size_np_reg = np.array(self.size, dtype=np.float32)

                kernel_reg = np.zeros((len(extrema_points_reg_np), len(extrema_points_reg_np)), dtype=np.float64)
                for j, center in enumerate(extrema_points_reg_np):
                    d2 = self._periodic_distances_np(center, extrema_points_reg_np, size_np_reg)
                    kernel_reg[j, :] = np.exp(-0.5 * d2 / (radius ** 2))

                reg_eps = 1e-8
                try:
                    amplitudes_reg_np = np.linalg.solve(kernel_reg + reg_eps * np.eye(kernel_reg.shape[0]), desired_shift_reg_np)
                except np.linalg.LinAlgError:
                    amplitudes_reg_np = np.linalg.lstsq(kernel_reg + reg_eps * np.eye(kernel_reg.shape[0]), desired_shift_reg_np, rcond=None)[0]

                centers_reg = jnp.array(extrema_points_reg_np, dtype=jnp.float32)
                amplitudes_reg = jnp.array(amplitudes_reg_np, dtype=jnp.float32)
                correction_grid_reg = self._gaussian_sum(points_fft, centers_reg, amplitudes_reg, radius).reshape(n_fft)
                correction_hat_reg = jnp.fft.fftshift(jnp.fft.fftn(correction_grid_reg)) / correction_grid_reg.size
                correction_hat_reg = (correction_hat_reg + jnp.flip(jnp.conj(correction_hat_reg))) / 2
                self.A = self.A + correction_hat_reg.astype(self.A.dtype)

                if verbose:
                    residual_reg = kernel_reg @ amplitudes_reg_np - desired_shift_reg_np
                    print(
                        f"Applied extrema regularization on {n_min_reg} minima and {n_max_reg} maxima "
                        f"(strength={amplitude_regularization:.2f}); max residual: {np.max(np.abs(residual_reg)):.2e}"
                    )

        if debug and axs is not None and xs_dbg is not None:
            phi_reg = self.evaluate_on_grid(xs_dbg)
            vmax_reg = float(jnp.max(jnp.abs(phi_reg)))
            im_reg = axs[2].imshow(
                np.array(phi_reg).T,
                origin='lower',
                cmap='bwr',
                vmin=-vmax_reg,
                vmax=vmax_reg,
                extent=(-float(self.size[0]) / 2, float(self.size[0]) / 2, -float(self.size[1]) / 2, float(self.size[1]) / 2),
                aspect='equal'
            )
            axs[2].set_title("(c)", pad=-10)
            if fig is not None:
                fig.colorbar(im_reg, ax=axs[2], fraction=0.046, pad=0.04)

        critical_points = self._find_points(
            points_per_grid=points_per_grid,
            max_steps=max_steps,
            atol=atol,
            use_interpolation=use_interpolation,
            verbose=verbose,
            discretization_points="critical_points",
            dup_tol=dup_tol,
        )
        size_np = np.array(self.size)
        image_rectified_np = np.array(image_rectified)

        all_target_values_np = np.array([
            self._sample_image_at_point(image_rectified_np, np.array(point), size_np) for point in np.array(critical_points)
        ], dtype=np.float32)

        if debug and axs is not None:
            if len(critical_points) > 0:
                crit_np = np.array(critical_points)
                shifts = np.array(jnp.meshgrid(jnp.arange(-1, 2), jnp.arange(-1, 2), indexing='ij')).reshape(2, -1).T
                tiled_points = np.concatenate([crit_np + shift[None, :] * size_np[None, :] for shift in shifts], axis=0)
                tiled_targets = np.tile(all_target_values_np, shifts.shape[0])
                mask = (
                    (tiled_points[:, 0] >= -float(self.size[0]) / 2)
                    & (tiled_points[:, 0] < float(self.size[0]) / 2)
                    & (tiled_points[:, 1] >= -float(self.size[1]) / 2)
                    & (tiled_points[:, 1] < float(self.size[1]) / 2)
                )
                sc = axs[1].scatter(
                    tiled_points[mask, 0],
                    tiled_points[mask, 1],
                    c=tiled_targets[mask],
                    cmap='gray',
                    vmin=rect_low,
                    vmax=rect_high,
                    s=16,
                    edgecolors='none'
                )
                if fig is not None:
                    fig.colorbar(sc, ax=axs[1], fraction=0.046, pad=0.04)
            axs[1].set_xlim(-float(self.size[0]) / 2, float(self.size[0]) / 2)
            axs[1].set_ylim(-float(self.size[1]) / 2, float(self.size[1]) / 2)
            axs[1].set_aspect('equal')
            axs[1].set_title("(b)", pad=-10)

        hessians = self.batch_hessian(critical_points)
        det_h = jnp.linalg.det(hessians)
        tr_h = jnp.trace(hessians, axis1=-2, axis2=-1)
        minima_mask = (det_h > 0) & (tr_h > 0)
        maxima_mask = (det_h > 0) & (tr_h < 0)
        saddle_mask = det_h < 0

        minima_points = critical_points[minima_mask]
        maxima_points = critical_points[maxima_mask]
        saddle_points = critical_points[saddle_mask]

        minima_vals = self.batch_potential(minima_points) if len(minima_points) > 0 else jnp.zeros((0,), dtype=jnp.float32)
        maxima_vals = self.batch_potential(maxima_points) if len(maxima_points) > 0 else jnp.zeros((0,), dtype=jnp.float32)
        saddle_vals = self.batch_potential(saddle_points) if len(saddle_points) > 0 else jnp.zeros((0,), dtype=jnp.float32)

        saddle_centers = []
        saddle_amplitudes = []

        if len(minima_points) >= 2 and len(maxima_points) >= 2 and len(saddle_points) > 0:
            minima_points_np = np.array(minima_points)
            maxima_points_np = np.array(maxima_points)
            minima_vals_np = np.array(minima_vals)
            maxima_vals_np = np.array(maxima_vals)
            saddle_points_np = np.array(saddle_points)
            saddle_vals_np = np.array(saddle_vals)

            for saddle_point, saddle_value in zip(saddle_points_np, saddle_vals_np):
                d_min = self._periodic_distances_np(saddle_point, minima_points_np, size_np)
                d_max = self._periodic_distances_np(saddle_point, maxima_points_np, size_np)
                min_ids = np.argsort(d_min)[:2]
                max_ids = np.argsort(d_max)[:2]

                local_floor = float(np.max(minima_vals_np[min_ids]))
                local_ceiling = float(np.min(maxima_vals_np[max_ids]))
                if not (local_ceiling > local_floor):
                    continue

                target_height = float(self._sample_image_at_point(image_rectified_np, saddle_point, size_np))
                target_value = local_floor + target_height * (local_ceiling - local_floor)
                amplitude = float(target_value - saddle_value)

                saddle_centers.append(saddle_point)
                saddle_amplitudes.append(amplitude)

        if len(saddle_centers) > 0 and len(minima_points) > 0:
            minima_points_np = np.array(minima_points, dtype=np.float32)
            saddle_centers_np = np.array(saddle_centers, dtype=np.float32)
            saddle_amplitudes_np = np.array(saddle_amplitudes, dtype=np.float64)

            saddle_shift_at_minima = np.zeros((len(minima_points_np),), dtype=np.float64)
            for center, amplitude in zip(saddle_centers_np, saddle_amplitudes_np):
                d2 = self._periodic_distances_np(center, minima_points_np, size_np)
                saddle_shift_at_minima += amplitude * np.exp(-0.5 * d2 / (radius ** 2))

            kernel_mm = np.zeros((len(minima_points_np), len(minima_points_np)), dtype=np.float64)
            for j, center in enumerate(minima_points_np):
                d2 = self._periodic_distances_np(center, minima_points_np, size_np)
                kernel_mm[j, :] = np.exp(-0.5 * d2 / (radius ** 2))

            reg = 1e-8
            try:
                minima_compensation_amplitudes = np.linalg.solve(kernel_mm + reg * np.eye(kernel_mm.shape[0]), -saddle_shift_at_minima)
            except np.linalg.LinAlgError:
                minima_compensation_amplitudes = np.linalg.lstsq(kernel_mm + reg * np.eye(kernel_mm.shape[0]), -saddle_shift_at_minima, rcond=None)[0]

            saddle_centers.extend([point for point in minima_points_np])
            saddle_amplitudes.extend([float(amplitude) for amplitude in minima_compensation_amplitudes])

            if verbose:
                residual = kernel_mm @ minima_compensation_amplitudes + saddle_shift_at_minima
                print(f"Added {len(minima_points_np)} minima compensation Gaussians; max residual at minima: {np.max(np.abs(residual)):.2e}")

        if len(saddle_centers) > 0:
            centers = jnp.array(np.array(saddle_centers), dtype=jnp.float32)
            amplitudes = jnp.array(np.array(saddle_amplitudes), dtype=jnp.float32)
        else:
            centers = jnp.zeros((0, self.dim), dtype=jnp.float32)
            amplitudes = jnp.zeros((0,), dtype=jnp.float32)

        correction_grid = self._gaussian_sum(points_fft, centers, amplitudes, radius).reshape(n_fft)

        correction_hat = jnp.fft.fftshift(jnp.fft.fftn(correction_grid)) / correction_grid.size
        correction_hat = (correction_hat + jnp.flip(jnp.conj(correction_hat))) / 2
        self.A = self.A + correction_hat.astype(self.A.dtype)

        correction_bandlimited = jnp.real(jnp.fft.ifftn(jnp.fft.ifftshift(correction_hat * correction_grid.size)))
        if debug and axs is not None:
            v_abs = float(jnp.max(jnp.abs(correction_bandlimited))) if correction_bandlimited.size > 0 else 1e-9
            correction_bandlimited_centered = jnp.fft.fftshift(correction_bandlimited)
            im2 = axs[3].imshow(np.array(correction_bandlimited_centered).T, origin='lower', cmap='bwr', vmin=-v_abs, vmax=v_abs, extent=(-float(self.size[0]) / 2, float(self.size[0]) / 2, -float(self.size[1]) / 2, float(self.size[1]) / 2), aspect='equal')
            axs[3].set_title("(d)", pad=-10)
            if fig is not None:
                fig.colorbar(im2, ax=axs[3], fraction=0.046, pad=0.04)

        self._initialize_interpolators_and_stationary_dist(interpolate=interpolate, verbose=verbose,
                                                           phi_interp_args=phi_interp_args,
                                                           force_interp_args=force_interp_args)
        if initialize_stepfunction:
            self._initialize_stepfunction(step_side_length=step_side_length, verbose=verbose)
        else:
            self._clear_stepfunction()
        if tesselate:
            self._tesselate(**tesselation_args, verbose=verbose)
        else:
            self._dont_tesselate()

        if debug and fig is not None:
            fig.canvas.draw_idle()

        if verbose:
            print(f"Σ|A|^2 = E[Re(Phi)^2] = {jnp.sum(jnp.abs(self.A)**2):3f},  Σ |EA|^2 = {jnp.sum(self.EA**2):3f}")

    def evaluate_on_grid(self, xs: tuple[Float[Array, "n0"], ...]) -> Float[Array, "n0 ..."]:
        return self.potential_on_grid(xs)

    @staticmethod
    def _load_and_prepare_image(image_file: str) -> Float[Array, "h w"]:
        image = plt.imread(image_file)
        if image.ndim == 3:
            image = image[..., :3]
            image = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
        elif image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale or 3D RGB(A) image, got shape {image.shape}.")

        image = image.astype(np.float32)
        finite_image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        min_val = float(np.min(finite_image))
        max_val = float(np.max(finite_image))
        if max_val <= min_val:
            return jnp.zeros_like(jnp.array(finite_image))
        image_bw = (finite_image - min_val) / (max_val - min_val)
        return jnp.array(image_bw, dtype=jnp.float32)

    @staticmethod
    def _periodic_distances_np(center: np.ndarray, points: np.ndarray, size: np.ndarray) -> np.ndarray:
        dx = points - center[None, :]
        dx = (dx + size[None, :] / 2) % size[None, :] - size[None, :] / 2
        return np.sum(dx ** 2, axis=-1)

    @staticmethod
    def _sample_image_at_point(image_rectified: np.ndarray, point: np.ndarray, size: np.ndarray) -> float:
        h, w = image_rectified.shape
        x_norm = ((point[0] + size[0] / 2) % size[0]) / size[0]
        y_norm = ((point[1] + size[1] / 2) % size[1]) / size[1]
        ix = min(max(int(np.floor(x_norm * w)), 0), w - 1)
        iy_top = min(max(int(np.floor(y_norm * h)), 0), h - 1)
        iy = h - 1 - iy_top
        return float(image_rectified[iy, ix])

    @partial(jit, static_argnames=("self",))
    def _gaussian_sum(self,
                      points: Float[Array, "n_points dim"],
                      centers: Float[Array, "n_centers dim"],
                      amplitudes: Float[Array, "n_centers"],
                      radius: Float) -> Float[Array, "n_points"]:
        if centers.shape[0] == 0:
            return jnp.zeros((points.shape[0],), dtype=jnp.float32)
        deltas = points[:, None, :] - centers[None, :, :]
        deltas = (deltas + self.size / 2) % self.size - self.size / 2
        dist2 = jnp.sum(deltas ** 2, axis=-1)
        gaussians = jnp.exp(-0.5 * dist2 / (radius ** 2))
        return jnp.sum(gaussians * amplitudes[None, :], axis=-1)

class SingleKField(Field):
    k: Float[Array, "dim"]
    a: Float

    def __init__(self, size: Float[Array, "dim"] = jnp.array([1., 1., 1.]),
                 k_int: Int[Array, "dim"] = jnp.array([10, 0, 0]),
                 a: Float = 1.,
                 initialize_stepfunction: bool = True,
                 step_side_length: Float | None = None,
                 verbose: bool = True):
        
        self.size = size
        self.dim = len(size)
        self.k = k_int * 2 * jnp.pi / size
        self.a = a
        self.k0 = jnp.linalg.norm(self.k) 
        self.lambda0 = 2 * jnp.pi / jnp.maximum(self.k0, 1e-6)
        self.ks = tuple(jnp.arange(-jnp.abs(ki), jnp.abs(ki) + 1) * 2 * jnp.pi / s for ki, s in zip(k_int, size))
    
        A = jnp.zeros(tuple(len(ki) for ki in self.ks), dtype=jnp.complex64)
        k_indices = jnp.where(k_int > 0, -1, 0)
        minus_k_indices = jnp.where(k_int > 0, 0, -1)
        A = A.at[k_indices].set(a)
        self.A = A.at[minus_k_indices].set(a)/jnp.sqrt(2)

        self.EA = jnp.abs(self.A)

        self.discr_points = jnp.array([[0.] * self.dim])
        self.hessians = jnp.array([jnp.zeros((self.dim, self.dim))])
        self.volumes = jnp.array([0.])
        self.num_vert = 1
        self.edges = jnp.array([])
        self.num_edges = 0
        self.phi_interpolator = None
        self.force_interpolators = None
        self.phi_min = -jnp.abs(a)
        self.phi_moments = jnp.array([0., a ** 2 / 2], dtype=jnp.float32)
        self.normalization_constant = jnp.exp(self.phi_min) * jax.scipy.special.i0(a)
        self.voro = None
        self.tesselation = None
        self.discr_points_extended = self.discr_points
        self.discr_indices_extended = jnp.zeros((0,), dtype=jnp.int32)

        if initialize_stepfunction:
            self._initialize_stepfunction(step_side_length=step_side_length, verbose=verbose)
        else:
            self._clear_stepfunction()

    @partial(jit, static_argnames=('self',))
    def potential(self, x: Float[Array, "dim"]) -> Float:
        """Pointwise potential at ``x``."""
        return self.a * jnp.cos(jnp.dot(self.k, x))
    
    @partial(jit, static_argnames=('self',))
    def force(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """Pointwise force at ``x``."""
        return self.a * jnp.sin(jnp.dot(self.k, x)) * self.k
    
    @partial(jit, static_argnames=('self',))
    def hessian(self, x: Float[Array, "dim"]) -> Float[Array, "dim dim"]:
        """Pointwise Hessian at ``x``."""
        cos_kx = jnp.cos(jnp.dot(self.k, x))
        return self.a * cos_kx * jnp.outer(self.k, self.k)
    
    @partial(jit, static_argnames=('self',))
    def batch_potential(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points"]:
        """Batched potential at ``xs``."""
        return self.a * jnp.cos(jnp.tensordot(self.k, xs, axes=(0,-1)))
    
    @partial(jit, static_argnames=('self',))
    def batch_force(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points dim"]:
        """Batched force at ``xs``."""
        sin_kx = jnp.sin(jnp.tensordot(self.k, xs, axes=(0,-1)))
        return self.a * sin_kx[:, None] * self.k[None, :]

    @partial(jit, static_argnames=('self',))
    def batch_hessian(self, xs: Float[Array, "n_points dim"]) -> Float[Array, "n_points dim dim"]:
        """Batched Hessian at ``xs``."""
        cos_kx = jnp.cos(jnp.tensordot(self.k, xs, axes=(0,-1)))
        return self.a * cos_kx[:, None, None] * jnp.outer(self.k, self.k)[None, :, :]
    
    
