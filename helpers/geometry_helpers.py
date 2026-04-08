from turtle import color
import jax.numpy as jnp
from jax import jit, vmap
import jax.random as jr
import equinox as eqx
from equinox import filter_jit, filter_vmap
from functools import partial
from jaxtyping import Array, Float, Int, UInt, PyTree
from typing import Tuple, List, Collection, Callable
from timeit import timeit
import time
import sys
import freud
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
sys.path.append('/home/tlequy/PycharmProjects/DiffSim/')
from helpers.plotting_fields import plot_tesselation

SHOW_COMPILE_MESSAGES = False
CHECKS = True

@jit
def roll_padded(a: Array, n: int, shift: int) -> Array:
    """Rolls a padded array along a given axis, keeping the padding values in place."""
    return a[(jnp.arange(a.shape[0]) - shift) % n]

@jit
def fill_padding(a: Array, n: int, pad_value: float = jnp.nan) -> Array:
    """Fills the padding values of a ragged array with a specified value."""
    mask = jnp.arange(a.shape[0]) < n
    return jnp.where(mask[:, *([None] * (a.ndim - 1))], a, pad_value)


@jit
def reorder_polygon_vertices_and_area_3d(polygon_vertices: Float[Array, "max_num_polygon_vert 3"], normal_vector: Float[Array, "3"], num_vert: Int | None = None) -> Tuple[Float[Array, "max_num_polygon_vert 3"], Float]:
    """Reorder polygon vertices in counterclockwise order around the normal vector."""
    if SHOW_COMPILE_MESSAGES:
        print("recompiling reorder_polygon_vertices_and_area_3d with max_num_polygon_vert =", polygon_vertices.shape[0])
    normal_vector = normal_vector / jnp.linalg.norm(normal_vector)
    if num_vert is not None:
        mask = jnp.arange(polygon_vertices.shape[0]) < num_vert
    else:
        num_vert = polygon_vertices.shape[0]
        mask = jnp.ones(polygon_vertices.shape[0], dtype=bool)
    
    barycenter = jnp.sum(jnp.nan_to_num(polygon_vertices) * mask[:, None], axis=0) / num_vert
    directions = polygon_vertices - barycenter
    x = jnp.dot(directions, directions[0])
    y = jnp.dot(directions, jnp.cross(normal_vector, directions[0]))
    angles = jnp.arctan2(x, y)
    angles = jnp.where(mask, angles, jnp.inf)
    polygon_vertices = polygon_vertices[jnp.argsort(angles)]
    next_polygon_vertices = roll_padded(polygon_vertices, num_vert, -1)
    area = 0.5 * jnp.sum(mask[:, None] * jnp.nan_to_num(jnp.cross(next_polygon_vertices, polygon_vertices)), axis=0).dot(normal_vector)
    return fill_padding(polygon_vertices, num_vert), area


class Tesselation(eqx.Module):
    n_cells: Int
    dim: Int
    centers: Float[Array, "n_cells dim"]
    inner_radii: Float[Array, "n_cells"]
    outer_radii: Float[Array, "n_cells"]
    polytope_vertices: Float[Array, "n_cells max_num_vert dim"] # ragged
    num_polytope_vertices: Int[Array, "n_cells"] # cutoffs
    volumes: Float[Array, "n_cells"]
    neighbors: Int[Array, "n_cells max_degree"] # ragged
    degrees: Int[Array, "n_cells"] # cutoffs
    edge_polygon_vertices: Float[Array, "n_cells max_num_edges max_num_polygon_vert dim"] # ragged
    edge_num_polygon_vertices: Int[Array, "n_cells max_num_edges"] # cutoffs
    edge_boundary_areas: Float[Array, "n_cells max_num_edges"]
    edge_center_distances: Float[Array, "n_cells max_num_edges"]
    edge_midpoints: Float[Array, "n_cells max_num_edges dim"]
    normals: Float[Array, "n_cells max_num_edges dim"]
    
    def plot(
        self,
        cell_ids: Int[Array, "n_plot_cells"] | None = None,
        ax: Axes3D | Axes | None = None,
        shift_vectors: Float[Array, "n_plot_cells dim"] | None = None,
        show_faces: bool | None = None,
        edge_color: str | List = "k",
        edge_width: float = 0.5,
        colors: List[Tuple[Float, ...]] | None = None,
        cmap: str = "Paired",
        edge_alpha: float | Float[Array, "n_plot_cells"] = 0.6,
        face_alpha: float | Float[Array, "n_plot_cells"] = 0.2,
        interior_alpha: float | Float[Array, "n_plot_cells"] | None = None,
        center_marker_size: Float = 20.0,
        batch_potential_fun: Callable | None = None,
        annotation_size: float = 15.0,
        z_level: Float | None = None,
        size_decay_length: Float = 0.5,
        **kwargs
    ) -> List:
        return plot_tesselation(
            tesselation=self,
            cell_ids=cell_ids,
            ax=ax,
            shift_vectors=shift_vectors,
            show_faces=show_faces,
            edge_color=edge_color,
            edge_width=edge_width,
            colors=colors,
            cmap=cmap,
            edge_alpha=edge_alpha,
            face_alpha=face_alpha,
            interior_alpha=interior_alpha,
            center_marker_size=center_marker_size,
            batch_potential_fun=batch_potential_fun,
            annotation_size=annotation_size,
            z_level=z_level,
            size_decay_length=size_decay_length,
            **kwargs
        )
    

@partial(jit, static_argnames=('dtype'))
def tesselation_1d(discretization_points_extended: Float[Array, "n_cells 1"], dtype=jnp.uint8) -> Tesselation:
    n_cells = discretization_points_extended.shape[0] - 2
    dim = 1
    centers = discretization_points_extended[1:-1]
    midpoints = 0.5 * (discretization_points_extended[:-1] + discretization_points_extended[1:])
    polytope_vertices = jnp.stack([midpoints[:-1], midpoints[1:]], axis=1) # shape (n_cells, 2, 1)
    num_polytope_vertices = jnp.full(n_cells, 2, dtype=jnp.int8)
    volumes = jnp.diff(midpoints[:, 0])
    inner_radii = volumes/2
    outer_radii = inner_radii
    neighbors = (jnp.stack([jnp.arange(-1, n_cells-1), jnp.arange(1, n_cells+1)], axis=1) % n_cells).astype(dtype)
    degrees = jnp.full(n_cells, 2, dtype=jnp.int8)
    edge_polygon_vertices = polytope_vertices[:, :, None, :] # shape (n_cells, 2, 1, 1)
    edge_num_polygon_vertices = jnp.full((n_cells, 2), 1, dtype=jnp.int8)
    edge_boundary_areas = jnp.full((n_cells, 2), 1., dtype=jnp.float32)
    edge_center_distances = jnp.abs(centers[:, None, 0] - edge_polygon_vertices[:,:,0,0]) # shape (n_cells, 2)
    edge_midpoints = polytope_vertices
    normals = jnp.stack([jnp.full(n_cells, -1.), jnp.full(n_cells, 1.)], axis=1)[:, :, None] # shape (n_cells, 2, 1)
    
    return Tesselation(n_cells=n_cells, dim=dim, centers=centers, inner_radii=inner_radii, outer_radii=outer_radii,
                       polytope_vertices=polytope_vertices, num_polytope_vertices=num_polytope_vertices, volumes=volumes, 
                       neighbors=neighbors, degrees=degrees, edge_polygon_vertices=edge_polygon_vertices, 
                       edge_num_polygon_vertices=edge_num_polygon_vertices, edge_boundary_areas=edge_boundary_areas, 
                       edge_center_distances=edge_center_distances, edge_midpoints=edge_midpoints, normals=normals)


@jit
@partial(vmap, in_axes=(0, 0, 0, 0, 0))
def compute_voronoi_cells_2d(polygon_vertices: Float[Array, "max_degree 2"], center: Float[Array, "2"], vectors: Float[Array, "max_degree 2"], neighbor_ids: Int[Array, "max_degree"], degree: Int | None = None) -> PyTree:
    """returns: polygon_vertices, neighbors_ordering, midpoints, normals, center_to_midpoint_distances, side_lengths, area, error"""
    if SHOW_COMPILE_MESSAGES:
        assert polygon_vertices.shape[0] == vectors.shape[0], f"Expected number of polygon vertices to match number of neighbors for 2D Voronoi cell. Found {polygon_vertices.shape[0]} polygon vertices and {vectors.shape[0]} neighbors."
        print("recompiling compute_voronoi_polygon_2d with max_degree =", vectors.shape[0])
    if degree is None:
        degree = vectors.shape[0]

    if degree is None:
        degree = polygon_vertices.shape[0]

    mask = jnp.arange(polygon_vertices.shape[0]) < degree
    
    x = polygon_vertices[:, 0] - center[0]
    y = polygon_vertices[:, 1] - center[1]
    vertex_angles = jnp.arctan2(y, x)
    vertex_angles = jnp.where(mask, vertex_angles, jnp.inf)
    polygon_vertices = fill_padding(polygon_vertices[jnp.argsort(vertex_angles)], degree)
    next_polygon_vertices = roll_padded(polygon_vertices, degree, -1)
    sides = next_polygon_vertices - polygon_vertices
    edges = fill_padding(jnp.stack([polygon_vertices, next_polygon_vertices], axis=1), degree)
    side_normals = jnp.stack([sides[:, 1], -sides[:, 0]], axis=1)
    side_normals /= jnp.linalg.norm(side_normals, axis=1, keepdims=True)
    mid_normal = side_normals[0] + side_normals[degree-1]
    mid_normal_angle = jnp.arctan2(mid_normal[1], mid_normal[0])
    side_lengths = fill_padding(jnp.linalg.norm(sides, axis=1), degree, 0.)
    x = vectors[:, 0]
    y = vectors[:, 1]
    neighbor_angles = jnp.arctan2(y, x)
    neighbor_angles = jnp.where(mask, (neighbor_angles - mid_normal_angle) % (2 * jnp.pi), jnp.inf)
    neighbors_ordering = jnp.argsort(neighbor_angles)
    neighbor_ids = fill_padding(neighbor_ids[neighbors_ordering], degree, jnp.iinfo(neighbor_ids.dtype).max)
    vectors = fill_padding(vectors[neighbors_ordering], degree)
    midpoints = fill_padding(center + vectors / 2, degree)
    normals = midpoints - center
    center_to_midpoints = fill_padding(jnp.linalg.norm(normals, axis=1), degree, 0.)
    inner_radius = jnp.min(center_to_midpoints)
    outer_radius = jnp.max(fill_padding(jnp.linalg.norm(polygon_vertices - center, axis=1), degree, -jnp.inf))
    normals /= center_to_midpoints[:, None]
    normals = fill_padding(normals, degree)
    area = 0.5 * jnp.sum(jnp.nan_to_num(center_to_midpoints * side_lengths) * mask)
    edge_num_polygon_vertices = fill_padding(jnp.full_like(neighbor_ids, 2, dtype=jnp.int8), degree, 0)
    if CHECKS:
        error = jnp.max(jnp.nan_to_num(jnp.linalg.norm(side_normals/jnp.linalg.norm(side_normals, axis=1, keepdims=True) - normals, axis=1)))
    else:
        error = 0.
    return {"polytope_vertices": polygon_vertices, "neighbors": neighbor_ids, "edge_polygon_vertices": edges,
            "edge_midpoints": midpoints, "normals": normals, "edge_center_distances": center_to_midpoints, 
            "edge_boundary_areas": side_lengths, "volumes": area, "edge_num_polygon_vertices": edge_num_polygon_vertices, 
            "inner_radii": inner_radius, "outer_radii": outer_radius}
          
@jit
def tesselation_2d(centers: Float[Array, "num_points 2"], neighbor_indices: UInt[Array, "num_points max_degree"], vectors: Float[Array, "num_points max_degree 2"], polytopes: Float[Array, "num_points max_degree 2"], degrees: Int[Array, "num_points"], edge_detection_threshold: float = 1e-5):
    n_cells = centers.shape[0]
    dim = 2
    centers = centers
    num_polytope_vertices = degrees
    degrees = degrees
    res = compute_voronoi_cells_2d(polytopes, centers, vectors, neighbor_indices, degrees)

    return Tesselation(n_cells=n_cells, dim=dim, centers=centers, num_polytope_vertices=num_polytope_vertices, degrees=degrees, **res)


@partial(jit, static_argnames=('max_polygon_vert'))
@partial(vmap, in_axes=(0, 0, 0, 0, None, None))
def compute_voronoi_cells_3d(center: Float[Array, "3"], vectors: Float[Array, "max_degree 3"], polytope_vertices: Float[Array, "max_num_vert 3"], degree: Int | None = None, max_polygon_vert : Int = 10, threshold: float = 1e-5):
    """returns: edge_polygon_vertices, edge_boundary_areas, edge_center_distances, edge_midpoints, normals"""
    if SHOW_COMPILE_MESSAGES:
        print("recompiling compute_voronoi_polytope_3d with max_degree =", vectors.shape[0], "and max_num_vert =", polytope_vertices.shape[0])
    if degree is None:
        degree = vectors.shape[0]

    midpoints = center + vectors / 2
    normal_vectors = vectors
    center_distances = jnp.linalg.norm(vectors, axis=1) / 2
    normal_vectors /= 2 * center_distances[:, None]
    distances = jnp.sum((polytope_vertices[None, :, :] - midpoints[:, None, :]) * normal_vectors[:, None, :], axis=-1)
    mask2d = jnp.abs(distances) < threshold
    polygon_vert_nums = mask2d.sum(axis=1)
    
    @partial(vmap, in_axes=(0, 0, 0))
    @jit
    def get_polygon_vertices(mask, normal, num_vert):
        polygon_vertices = polytope_vertices[jnp.nonzero(mask, size=max_polygon_vert)]
        return reorder_polygon_vertices_and_area_3d(polygon_vertices, normal, num_vert)

    polygon_vertices, boundary_areas = get_polygon_vertices(mask2d, normal_vectors, polygon_vert_nums)
    polygon_vertices = fill_padding(polygon_vertices, degree, jnp.nan)
    boundary_areas = fill_padding(boundary_areas, degree, 0.)
    volume = jnp.sum(boundary_areas * jnp.nan_to_num(center_distances)) / 3.
    inner_radius = jnp.min(fill_padding(center_distances, degree, jnp.inf))
    outer_radius = jnp.max(jnp.nan_to_num(jnp.linalg.norm(polytope_vertices - center, axis=1), nan=-jnp.inf))

    return {"edge_polygon_vertices": polygon_vertices, "edge_boundary_areas": boundary_areas, "edge_center_distances": center_distances, 
            "edge_midpoints": midpoints, "normals": normal_vectors, "edge_num_polygon_vertices": polygon_vert_nums, "inner_radii": inner_radius,
            "outer_radii": outer_radius, "edge_center_distances": center_distances, "volumes": volume}

@jit
def tesselation_3d(centers: Float[Array, "num_points 3"], neighbor_indices: UInt[Array, "num_points max_degree"], vectors: Float[Array, "num_points max_degree 3"], polytopes: Float[Array, "num_points max_num_vert 3"], num_polytope_vertices: Int[Array, "num_points"], degrees: Int[Array, "num_points"], edge_detection_threshold: float = 1e-5):
    n_cells = centers.shape[0]
    dim = 3
    centers = centers
    num_polytope_vertices = num_polytope_vertices
    degrees = degrees
    
    res = compute_voronoi_cells_3d(centers, vectors, polytopes, degrees, 10, edge_detection_threshold)
    
    return Tesselation(
        n_cells=n_cells,
        dim=dim,
        centers=centers,
        polytope_vertices=polytopes,
        neighbors=neighbor_indices,
        num_polytope_vertices=num_polytope_vertices,
        degrees=degrees,
        **res
    )