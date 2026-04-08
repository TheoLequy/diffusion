from typing import NamedTuple

import jax.numpy as jnp
import optimistix as optx
import optax
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int

from coarse_graining.fields import Field


class NEBResult(NamedTuple):
    """Outputs of a nudged elastic band optimization."""

    path: Float[Array, "n_images dim"]
    saddle_index: Int
    saddle_point: Float[Array, "dim"]
    saddle_potential: Float
    saddle_hessian_eigenvalues: Float[Array, "dim"]
    total_path_length: Float
    converged: Bool
    max_displacement: Float


def neb(
    field: Field,
    start: Float[ArrayLike, "dim"],
    end: Float[ArrayLike, "dim"],
    n_images: int = 32,
    max_steps: int = 3000,
    step_size: float = 1e-2,
    spring_constant: float = 1.0,
    tol: float = 1e-6,
    solver: optx.AbstractMinimiser | None = None,
) -> NEBResult:
    """Find a minimum-energy path between two points using NEB + Optimistix.

    Parameters
    - field: potential field exposing `force`, `hessian`, and optional `fold_to_base_cell`.
    - start/end: endpoints of the path, shape `(dim,)`; endpoints remain fixed.
    - n_images: total number of images along the path (including endpoints).

    - solver: optional optimistix minimizer. If None, uses Adam via `OptaxMinimiser`.

    Returns
    - `NEBResult` containing the converged path, saddle information, and total path length.
    """
    if n_images < 3:
        raise ValueError(f"n_images must be >= 3, got {n_images}.")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}.")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}.")
    if spring_constant <= 0:
        raise ValueError(f"spring_constant must be positive, got {spring_constant}.")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")

    x_start = jnp.asarray(start, dtype=jnp.float32)
    x_end = jnp.asarray(end, dtype=jnp.float32)
    if x_start.ndim != 1 or x_end.ndim != 1:
        raise ValueError("start and end must be 1D vectors.")
    if x_start.shape != x_end.shape:
        raise ValueError(f"start and end must have same shape, got {x_start.shape} and {x_end.shape}.")
    if x_start.shape[0] != field.dim:
        raise ValueError(f"Point dimension ({x_start.shape[0]}) does not match field.dim ({field.dim}).")

    use_fold = hasattr(field, "fold_to_base_cell")

    def assemble_path(mid_images: Float[Array, "n_mid dim"]) -> Float[Array, "n_images dim"]:
        """Assemble path.
        
        Args:
            mid_images: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        images = jnp.concatenate((x_start[None, :], mid_images, x_end[None, :]), axis=0)
        if use_fold:
            images = field.fold_to_base_cell(images)
            images = images.at[0].set(x_start)
            images = images.at[-1].set(x_end)
        return images

    alphas = jnp.linspace(0.0, 1.0, n_images, dtype=x_start.dtype)[:, None]
    init_images = (1.0 - alphas) * x_start[None, :] + alphas * x_end[None, :]
    init_mid = init_images[1:-1]

    target_spacing = jnp.linalg.norm(x_end - x_start) / (n_images - 1)

    def loss(mid_images: Float[Array, "n_mid dim"], _args: object) -> Float:
        """Loss.
        
        Args:
            mid_images: Input parameter.
            _args: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        images = assemble_path(mid_images)
        potentials = field.batch_potential(images)
        segments = images[1:] - images[:-1]
        segment_lengths = jnp.linalg.norm(segments, axis=1)
        spring_penalty = 0.5 * spring_constant * jnp.sum((segment_lengths - target_spacing) ** 2)
        return jnp.sum(potentials[1:-1]) + spring_penalty

    if solver is None:
        solver = optx.OptaxMinimiser(
            optax.adam(learning_rate=step_size),
            rtol=0.0,
            atol=tol,
            norm=optx.max_norm,
        )

    sol = optx.minimise(
        fn=loss,
        y0=init_mid,
        solver=solver,
        max_steps=max_steps,
        throw=False,
    )

    images = assemble_path(sol.value)
    converged = sol.result == optx.RESULTS.successful
    max_displacement = jnp.max(jnp.linalg.norm(sol.value - init_mid, axis=1))

    potentials = field.batch_potential(images)
    saddle_index = jnp.argmax(potentials[1:-1]) + 1
    saddle_point = images[saddle_index]
    saddle_potential = potentials[saddle_index]

    saddle_hessian = field.hessian(saddle_point)
    saddle_hessian_eigenvalues = jnp.linalg.eigvalsh(saddle_hessian)

    segment_lengths = jnp.linalg.norm(images[1:] - images[:-1], axis=1)
    total_path_length = jnp.sum(segment_lengths)

    return NEBResult(
        path=images,
        saddle_index=saddle_index,
        saddle_point=saddle_point,
        saddle_potential=saddle_potential,
        saddle_hessian_eigenvalues=saddle_hessian_eigenvalues,
        total_path_length=total_path_length,
        converged=jnp.asarray(converged),
        max_displacement=max_displacement,
    )