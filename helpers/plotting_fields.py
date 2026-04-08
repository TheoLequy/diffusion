from IPython.display import display
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from cycler import cycler
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.axes import Axes
import scienceplots
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
import zarr
import scipy.sparse
import networkx as nx
from skimage.measure import marching_cubes
from jaxtyping import Float, Array, Int, Bool, PyTree
from typing import Any, Callable, Collection, Iterable, List, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
	from coarse_graining.fields import Field
	from coarse_graining.simulations import Simulator
	from helpers.geometry_helpers import Tesselation
plt.style.use(['science'])
rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 10})
# rc('figure.constrained_layout', use=True)
rc('text', usetex=True)
rc('lines', linewidth=2)
rc('axes.formatter', use_mathtext=True)
plt.rcParams.update({'figure.dpi': '100'})
prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = prop_cycle.by_key()['color']
COLORS.append('tab:pink')
plt.rcParams['axes.prop_cycle'] =  cycler(color=COLORS)
comp_cycler = (cycler(color=COLORS[:4]) + cycler(lw=[2, 2, 2, 2]) + cycler(linestyle=['-', '--', '-.', ':']))


def _nearest_index(values: Array | np.ndarray, value: float) -> int:
	"""Internal helper to nearest index.
	
	Args:
		values: Input parameter.
		value: Input parameter.
	
	Returns:
		Output value computed by this function.
	"""
	values_np = np.asarray(values)
	if values_np.size == 0:
		return 0
	return int(np.argmin(np.abs(values_np - value)))


def plot_tesselation(
	tesselation: "Tesselation",
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
	annotation_size: float = 10.0,
	z_level: Float | None = None,
	size_decay_length: Float = 0.5,
	**kwargs
) -> List:
	"""Plot tesselation cells.

	Returns a list of matplotlib artists created on the provided axes.
	"""
	if show_faces is None:
		show_faces = not isinstance(ax, Axes3D)

	if ax is None:
		fig, ax = plt.subplots(subplot_kw={'projection': '3d'} if tesselation.dim == 3 else {})
	else:
		fig = ax.get_figure()
	if cell_ids is None:
		cell_ids = jnp.arange(tesselation.n_cells)
	if tesselation.dim <= 2:
		assert isinstance(ax, Axes), f"Expected 2D axes for plotting 2D or 1D cell. Got {type(ax)}"
	if tesselation.dim > 3:
		raise NotImplementedError(f"Plotting not implemented for dimension {tesselation.dim} greater than 3")
	if interior_alpha is None:
		interior_alpha = face_alpha if tesselation.dim <= 2 else 0.0
	if colors is None:
		colors = [plt.get_cmap(cmap)(float(jr.uniform(key=jr.key(int(cell_id))))) for cell_id in cell_ids]

	if not isinstance(face_alpha, Array):
		face_alpha = jnp.full(len(cell_ids), face_alpha)
	if not isinstance(edge_alpha, Array):
		edge_alpha = jnp.full(len(cell_ids), edge_alpha)
	if not isinstance(edge_color, list):
		edge_color = [edge_color] * len(cell_ids)

	facecolors = [mcolors.to_rgba(c, alpha=float(fa)) for c, fa in zip(colors, face_alpha)]
	edgecolors = [mcolors.to_rgba(ec, alpha=float(ea)) for ec, ea in zip(edge_color, edge_alpha)]

	if shift_vectors is None:
		shift_vectors = jnp.zeros_like(tesselation.centers[cell_ids])

	centers_shifted = tesselation.centers[cell_ids] + shift_vectors
	polytope_vertices_shifted = tesselation.polytope_vertices[cell_ids, :, :] + shift_vectors[:, None, :]
	polytope_vertices_shifted_ragged = [
		polytope_vertices_shifted[i, :tesselation.num_polytope_vertices[cell_ids[i]], :]
		for i in range(len(cell_ids))
	]
	x_offsets = 0.1 * tesselation.inner_radii[cell_ids]

	if tesselation.dim == 1:
		assert batch_potential_fun is not None, "batch_potential_fun must be provided for plotting 1D cells"
		center_potentials = batch_potential_fun(centers_shifted)
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		mask = (
			(centers_shifted[:, 0] >= xlim[0])
			& (centers_shifted[:, 0] <= xlim[1])
			& (center_potentials >= ylim[0])
			& (center_potentials <= ylim[1])
		)
		centers_masked = centers_shifted[mask]
		center_potentials_masked = center_potentials[mask]
		cell_ids_masked = cell_ids[mask]
		colors_masked = [c for c, m in zip(colors, mask) if bool(m)]
		x_offsets_masked = x_offsets[mask]
		center_points = ax.scatter(
			centers_masked[:, 0],
			center_potentials_masked,
			color=colors_masked,
			s=center_marker_size,
			alpha=float(0.5 * (jnp.mean(interior_alpha) + 1))
		)
		center_labels = [
			ax.text(
				x=c + x_off,
				y=p,
				s=f"{cell_id}",
				fontsize=annotation_size,
				ha='left',
				va='top',
				clip_on=True
			)
			for c, p, cell_id, x_off in zip(
				centers_masked[:, 0],
				center_potentials_masked,
				cell_ids_masked,
				x_offsets_masked
			)
		]
		if not show_faces:
			return [center_points] + center_labels

		y_bottom, y_top = ax.get_ylim()
		verts = [
			jnp.array([[v[0], y_bottom], [v[1], y_bottom], [v[1], y_top], [v[0], y_top]])
			for v in polytope_vertices_shifted[:, :, 0]
		]
		polygons = mcollections.PolyCollection(verts, facecolors=facecolors, edgecolors=edgecolors, linewidths=edge_width)
		ax.add_collection(polygons)
		return [center_points] + center_labels + [polygons]

	if tesselation.dim == 2:
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		mask = (
			(centers_shifted[:, 0] >= xlim[0])
			& (centers_shifted[:, 0] <= xlim[1])
			& (centers_shifted[:, 1] >= ylim[0])
			& (centers_shifted[:, 1] <= ylim[1])
		)
		centers_masked = centers_shifted[mask]
		cell_ids_masked = cell_ids[mask]
		colors_masked = [c for c, m in zip(colors, mask) if bool(m)]
		edgecolors_masked = [ec for ec, m in zip(edgecolors, mask) if bool(m)]
		x_offsets_masked = x_offsets[mask]
		center_points = ax.scatter(
			*centers_masked.T,
			color=colors_masked,
			s=center_marker_size,
			alpha=float(0.5 * (jnp.mean(interior_alpha) + 1)),
			edgecolors=edgecolors_masked,
			linewidths=edge_width
		)
		center_labels = [
			ax.text(
				x=x + x_off,
				y=y,
				s=f"{cell_id}",
				fontsize=annotation_size,
				ha='left',
				va='top',
				clip_on=True
			)
			for (x, y), cell_id, x_off in zip(centers_masked, cell_ids_masked, x_offsets_masked)
		]

		if not show_faces:
			return [center_points] + center_labels

		polygons = mcollections.PolyCollection(polytope_vertices_shifted_ragged, facecolors=facecolors, edgecolors=edgecolors, linewidths=edge_width)
		ax.add_collection(polygons)
		return [center_points] + center_labels + [polygons]

	if tesselation.dim == 3:
		if isinstance(ax, Axes3D):
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			zlim = ax.get_zlim()
			mask = (
				(centers_shifted[:, 0] >= xlim[0])
				& (centers_shifted[:, 0] <= xlim[1])
				& (centers_shifted[:, 1] >= ylim[0])
				& (centers_shifted[:, 1] <= ylim[1])
				& (centers_shifted[:, 2] >= zlim[0])
				& (centers_shifted[:, 2] <= zlim[1])
			)
			centers_masked = centers_shifted[mask]
			cell_ids_masked = cell_ids[mask]
			colors_masked = [c for c, m in zip(colors, mask) if bool(m)]
			edgecolors_masked = [ec for ec, m in zip(edgecolors, mask) if bool(m)]
			x_offsets_masked = x_offsets[mask]
			center_points = ax.scatter(
				*centers_masked.T,
				color=colors_masked,
				s=center_marker_size,
				edgecolors=edgecolors_masked,
				linewidths=edge_width
			)
			center_labels = [
				ax.text(
					x=x + x_off,
					y=y,
					z=z,
					s=f"{cell_id}",
					fontsize=annotation_size,
					ha='left',
					va='top',
					clip_on=True
				)
				for (x, y, z), cell_id, x_off in zip(centers_masked, cell_ids_masked, x_offsets_masked)
			]
			if not show_faces:
				return [center_points] + center_labels
			
			verts = [
				tesselation.edge_polygon_vertices[i, j, :tesselation.edge_num_polygon_vertices[i, j], :] + shift_vectors[i]
				for i in cell_ids
				for j in range(tesselation.degrees[i])
			]
			edge_colors = [ec for ec, i in zip(edgecolors, cell_ids) for _ in range(tesselation.degrees[i])]
			face_colors = [fc for fc, i in zip(facecolors, cell_ids) for _ in range(tesselation.degrees[i])]
			print(face_colors, edge_colors)
			edge_polygons = Poly3DCollection(verts, facecolors=face_colors, edgecolors=edge_colors, linewidths=edge_width)
			ax.add_collection3d(edge_polygons)

			return [center_points] + center_labels + [edge_polygons]
		
		assert z_level is not None, "z_level must be provided for plotting 3D cell on 2D axes"
		intersected = jnp.array([
			jnp.any(pv[:, 2] < z_level) and jnp.any(pv[:, 2] > z_level)
			for pv in polytope_vertices_shifted_ragged
		])
		centers_subset = centers_shifted[intersected]

		delta_zs = centers_subset[:, 2] - z_level
		decay_factors = 1 / (1 + jnp.abs(delta_zs / size_decay_length))
		masked_colors = [c for c, m in zip(colors, intersected) if bool(m)]

		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		centers_subset_xy = centers_subset[:, :2]
		mask = (
			(centers_subset_xy[:, 0] >= xlim[0])
			& (centers_subset_xy[:, 0] <= xlim[1])
			& (centers_subset_xy[:, 1] >= ylim[0])
			& (centers_subset_xy[:, 1] <= ylim[1])
		)
		centers_subset_masked = centers_subset[mask]
		cell_ids_subset = cell_ids[intersected]
		cell_ids_masked = cell_ids_subset[mask]
		decay_factors_masked = decay_factors[mask]
		colors_masked = [c for c, m in zip(masked_colors, mask) if bool(m)]
		edgecolors_masked = [ec for ec, m in zip(edgecolors, intersected) if bool(m)]
		edgecolors_masked = [ec for ec, m in zip(edgecolors_masked, mask) if bool(m)]
		x_offsets_masked = x_offsets[intersected][mask]
		center_points = ax.scatter(
			*centers_subset_masked[:, :2].T,
			color=colors_masked,
			s=center_marker_size * decay_factors_masked,
			edgecolors=edgecolors_masked,
			linewidths=list(edge_width * decay_factors_masked)
		)
		center_labels = [
			ax.text(
				x=x + x_off,
				y=y,
				s=f"{cell_id}",
				fontsize=annotation_size * df,
				ha='left',
				va='top',
				clip_on=True
			)
			for (x, y, _z), cell_id, df, x_off in zip(centers_subset_masked, cell_ids_masked, decay_factors_masked, x_offsets_masked)
		]
		if not show_faces:
			return [center_points] + center_labels

		cell_ids_subset = cell_ids[intersected]
		shift_vectors_subset = shift_vectors[intersected]
		facecolors_subset = [fc for fc, m in zip(facecolors, intersected) if bool(m)]
		edgecolors_subset = [ec for ec, m in zip(edgecolors, intersected) if bool(m)]
		verts = [
			[
				tesselation.edge_polygon_vertices[int(cell_id), j, :tesselation.edge_num_polygon_vertices[int(cell_id), j], :]
				+ shift_vectors_subset[idx]
				for j in range(tesselation.degrees[int(cell_id)])
			]
			for idx, cell_id in enumerate(cell_ids_subset)
		]

		def get_cut_polygon(edge_polygon_vertices: List[Float[Array, "num_edge_polygon_vert 3"]]):
			"""Return cut polygon.
			
			Args:
				edge_polygon_vertices: Input parameter.
			
			Returns:
			Output value computed by this function.
			"""
			face_is_intersected = jnp.array([
				jnp.any(pv[:, 2] < z_level) and jnp.any(pv[:, 2] > z_level)
				for pv in edge_polygon_vertices
			])
			if not any(face_is_intersected):
				return jnp.array([]).reshape(0, 2)
			cut_verts = []
			for i, pvs in enumerate(edge_polygon_vertices):
				if not face_is_intersected[i]:
					continue
				below_mask = pvs[:, 2] < z_level
				edge_is_intersected = below_mask != jnp.roll(below_mask, -1)
				intersecting_indices = jnp.nonzero(edge_is_intersected)[0]
				points1 = pvs[intersecting_indices]
				points2 = pvs[(intersecting_indices + 1) % pvs.shape[0]]
				ratios = (z_level - points1[:, 2]) / (points2[:, 2] - points1[:, 2])
				new_verts = points1[:, :2] + ratios[:, None] * (points2[:, :2] - points1[:, :2])
				cut_verts.extend(new_verts)

			cut_verts = jnp.unique(jnp.asarray(cut_verts), axis=0)
			midpoint = jnp.mean(cut_verts, axis=0)
			angles = jnp.arctan2(cut_verts[:, 1] - midpoint[1], cut_verts[:, 0] - midpoint[0])
			cut_verts = cut_verts[jnp.argsort(angles)]
			return cut_verts

		cut_polygons = [get_cut_polygon(edge_polygon_vertices) for edge_polygon_vertices in verts]
		polygons = mcollections.PolyCollection(cut_polygons, facecolors=facecolors_subset, edgecolors=edgecolors_subset, linewidths=edge_width)
		ax.add_collection(polygons)
		return [center_points] + center_labels + [polygons]

	raise NotImplementedError("Plotting 3D cell on 2D axes not implemented yet.")


def plot_field(
	field: "Field",
	ax_x: Axes | None = None,
	ax_k: Axes | None = None,
	res: Float | None = None,
	plot_dim: Int | None = None,
	threshold: Float = 0.0003,
	n_levels: Int = 5,
	show_cells: Bool | None = None,
	cell_plot_args: PyTree = {},
	verbose: Bool = True,
	flip_potential_axes: Bool = False,
	bounds: Float[Array, "dim 2"] | None = None,
	pad: Float = 0.04,
	slider_k_init: float = 0.0,
	slider_x_init: float = 0.0,
	slider_y_init: float = 0.0,
	slider_z_init: float = 0.0,
	slider_mag_init: float | None = None,
	sliders: bool = True,
	transformation: str | None = None,
	bw: bool = False,
) -> tuple[object | None, tuple]:
	"""Plot field in k- and x-space.

	Returns the figure and any widget handles (empty tuple when no widgets).
	"""
	if bounds is None:
		bounds = jnp.array([[-s / 2, s / 2] for s in field.size])
	else:
		assert ax_x is not None, "must provide ax_x if custom bounds are provided"

	if plot_dim is None:
		plot_dim = int(min(field.dim, 2.0))
	if res is None:
		res = 50.0 / field.lambda0
	if field.dim == 2 and plot_dim == 1:
		print("WARNING: plot_dim=1 for a 2D field is not completely implemented yet.")

	if show_cells is None:
		show_cells = field.tesselation is not None

	assert 1 <= plot_dim <= field.dim <= 3, f"plot_dim must be between 1 and self.dim (inclusive). Got plot_dim={plot_dim}, self.dim={field.dim}"
	assert field.dim - plot_dim <= 1, f"can only show slices of one dimension less than the full dimension. Got plot_dim={plot_dim}, self.dim={field.dim}"

	xs = [jnp.linspace(b[0], b[1], int(res * (b[1] - b[0]) + 1)) for b in bounds]

	absA = jnp.abs(field.A)
	create_fig = ax_k is None and ax_x is None
	if create_fig:
		if plot_dim == 3:
			fig, (ax_k, ax_x) = plt.subplots(1, 2, figsize=(7, 3), subplot_kw={'projection': '3d'}, constrained_layout=True)
		else:
			fig, (ax_k, ax_x) = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
	else:
		fig = None

	k_slider_ids = ()
	x_slider_ids = ()
	k_ev, k_abs, k_real, k_im, k_plot, slider_x, slider_k, x_plot = (None,) * 8

	allow_widgets = sliders and create_fig and (plot_dim != field.dim or plot_dim == 3)
	if plot_dim != field.dim and not allow_widgets:
		k_slider_ids = (_nearest_index(field.ks[-1], slider_k_init),)
		init = slider_y_init if plot_dim == 1 else slider_z_init
		x_slider_ids = (_nearest_index(xs[-1], init),)

	if plot_dim != field.dim and allow_widgets:
		ax_k_slider, ax_x_slider = fig.add_axes((0.05, 0.02, 0.4, 0.06)), fig.add_axes((0.55, 0.02, 0.4, 0.06))
		slider_k = Slider(ax_k_slider, r'$k_i$', float(field.ks[-1][0]), float(field.ks[-1][-1]), valinit=slider_k_init, valstep=field.ks[-1])
		init = slider_y_init if plot_dim == 1 else slider_z_init
		slider_x = Slider(ax_x_slider, r'$x_i$', float(xs[-1][0]), float(xs[-1][-1]), valinit=init, valstep=xs[-1])
		k_slider_ids = (_nearest_index(field.ks[-1], float(slider_k.val)),)
		x_slider_ids = (_nearest_index(xs[-1], float(slider_x.val)),)

	if verbose:
		print(f"Evaluating field on grid of size {tuple(xis.size for xis in xs)} for plotting...")
	if transformation == "qm":
		Phi = field.qm_potential_on_grid(xs)
	else:
		Phi = field.potential_on_grid(xs)
	if verbose:
		print("Done evaluating field on grid, now plotting...")

	if plot_dim == 1:
		if ax_k is not None:
			if field.EA is not None:
				k_ev, = ax_k.plot(field.ks[0], field.EA[:, *k_slider_ids], label=r'$\sqrt{E[|A|^2]}$', lw=1)
			k_abs = ax_k.scatter(field.ks[0], absA[:, *k_slider_ids], label=r'$|A|$', s=20)
			k_real = ax_k.scatter(field.ks[0], jnp.real(field.A[:, *k_slider_ids]), label=r'$\Re(A)$', s=20)
			k_im = ax_k.scatter(field.ks[0], jnp.imag(field.A[:, *k_slider_ids]), label=r'$\Im(A)$', s=20)

			ax_k.legend(loc='lower center')
			ax_k.set_xlabel(r'$k$')
			ax_k.set_ylabel(r'$A$')

		if ax_x is not None:
			if flip_potential_axes:
				x_plot, = ax_x.plot(Phi[:, *x_slider_ids], xs[0], label="potential")
				ax_x.set_ylabel(r'$x$')
				ax_x.set_xlabel(r'$\Phi$')
			else:
				x_plot, = ax_x.plot(xs[0], Phi[:, *x_slider_ids], label="potential")
				ax_x.set_xlabel(r'$x$')
				ax_x.set_ylabel(r'$\Phi$')

			ax_x.legend(loc='lower left')

	if plot_dim == 2:
		colorbar_pad = max(pad, 0.08)
		if ax_k is not None:
			extent = (float(field.ks[0][0]), float(field.ks[0][-1]), float(field.ks[1][0]), float(field.ks[1][-1]))
			k_plot = ax_k.imshow(absA[:, :, *(k_slider_ids)].T, cmap='viridis', origin='lower', aspect='equal', extent=extent, vmin=0, vmax=float(jnp.max(absA)))
			k_colorbar = plt.colorbar(k_plot, ax=ax_k, fraction=0.046, pad=colorbar_pad, label=r'$|A|$')
			k_colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
			k_colorbar.update_ticks()
			ax_k.set_xlabel(r'$k_x$')
			ax_k.set_ylabel(r'$k_y$')

		if ax_x is not None:
			if bw:
				phi_min = float(jnp.min(Phi))
				phi_max = float(jnp.max(Phi))
				if phi_min == phi_max:
					phi_min -= 0.5
					phi_max += 0.5
				cmap_x = 'Greys'
				vmin_x, vmax_x = phi_min, phi_max
			else:
				phi_abs_max = float(jnp.max(jnp.abs(Phi)))
				cmap_x = 'RdBu_r'
				vmin_x, vmax_x = -phi_abs_max, phi_abs_max
			extent = (float(xs[0][0]), float(xs[0][-1]), float(xs[1][0]), float(xs[1][-1]))
			x_plot = ax_x.imshow(Phi[:, :, *(x_slider_ids)].T, cmap=cmap_x, origin='lower', aspect='equal', extent=extent, vmin=vmin_x, vmax=vmax_x)

			x_colorbar = plt.colorbar(x_plot, ax=ax_x, fraction=0.046, pad=colorbar_pad, label=r'$\Phi$')
			x_colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
			x_colorbar.update_ticks()
			ax_x.set_xlabel(r'$x$')
			ax_x.set_ylabel(r'$y$')

	if plot_dim == 3:
		assert ax_x is not None and isinstance(ax_x, Axes3D)
		if ax_k is not None:
			assert isinstance(ax_k, Axes3D)
		if create_fig:
			fig.subplots_adjust(right=0.9, bottom=0.1)
		ax_x_slider = ax_y_slider = ax_z_slider = ax_mag_slider = None
		slider_y = slider_z = slider_mag = None
		mag_init = 0.5 if slider_mag_init is None else float(slider_mag_init)
		if allow_widgets:
			ax_x_slider = fig.add_axes((0.55, 0.02, 0.35, 0.06))
			ax_y_slider = fig.add_axes((0.91, 0.2, 0.03, 0.6))
			ax_z_slider = fig.add_axes((0.94, 0.2, 0.03, 0.6))
			ax_mag_slider = fig.add_axes((0.97, 0.2, 0.03, 0.6))
			slider_x = Slider(ax_x_slider, orientation='horizontal', label=r'$x$', valmin=float(xs[0][0]), valmax=float(xs[0][-1]), valinit=slider_x_init, valstep=xs[0])
			slider_y = Slider(ax_y_slider, orientation='vertical', label=r'$y$', valmin=float(xs[1][0]), valmax=float(xs[1][-1]), valinit=slider_y_init, valstep=xs[1])
			slider_z = Slider(ax_z_slider, orientation='vertical', label=r'$z$', valmin=float(xs[2][0]), valmax=float(xs[2][-1]), valinit=slider_z_init, valstep=xs[2])
			slider_mag = Slider(ax_mag_slider, orientation='vertical', label='mag', valmin=float(1.0 / jnp.min(field.size)), valmax=3 * mag_init, valinit=mag_init, color='C1')

		id_sorted = jnp.argsort(jnp.ravel(absA), descending=True)
		max_absA = absA[jnp.unravel_index(id_sorted[0], absA.shape)]
		ks_id_sorted = jnp.unravel_index(id_sorted, absA.shape)
		n_plot = jnp.searchsorted(-jnp.ravel(absA[ks_id_sorted]), -threshold, side='right')
		print(f"Plotting {n_plot} / {absA.size} k-modes above threshold |A| = {threshold:3f}")
		ks_id_sorted = tuple(k_id[:n_plot] for k_id in ks_id_sorted)
		ks_sorted = [k[ks_id_sorted[i]] for i, k in enumerate(field.ks)]
		absA_sorted = absA[ks_id_sorted]
		if ax_k is not None:
			k_scatter = ax_k.scatter(*ks_sorted, s=jnp.sqrt(absA_sorted) * 30, c=ks_sorted[2], cmap='magma')
			ax_k.figure.colorbar(k_scatter, ax=ax_k, fraction=0.046, pad=pad, location='left', label=r'$k_z$', shrink=0.6)
			ax_k.set_xlabel(r'$k_x$')
			ax_k.set_ylabel(r'$k_y$')
			ax_k.set_zlabel(r'$k_z$')
			ax_k.set_aspect('equal')
			for a in jnp.logspace(-2, 0, 4) * max_absA:
				a_round = jnp.round(a, decimals=2 - int(jnp.floor(jnp.log10(a))))
				ax_k.scatter([], [], s=int(jnp.sqrt(a_round) * 30), label=f'{a_round:1.0e}', color='gray')

			ax_k.legend(title=r'$|A| =$', loc='upper right', frameon=True, framealpha=0.5)

		spacings = tuple(xs[i][1] - xs[i][0] for i in range(3))
		levels = jnp.linspace(-jnp.max(jnp.abs(Phi)), jnp.max(jnp.abs(Phi)), n_levels + 2)[1:-1]
		cmap = plt.get_cmap("RdBu_r")
		colors = cmap((levels + 0.7 * jnp.max(jnp.abs(Phi))) / (1.4 * jnp.max(jnp.abs(Phi))))

		def plot_cropped_region(x_center, y_center, z_center, magnification):
			"""Plot cropped region.
			
			Args:
				x_center: Input parameter.
				y_center: Input parameter.
				z_center: Input parameter.
				magnification: Input parameter.
			"""
			w = int(res / magnification)
			idx = jnp.clip(jnp.searchsorted(xs[0], x_center) - w // 2, 0, Phi.shape[0] - w)
			idy = jnp.clip(jnp.searchsorted(xs[1], y_center) - w // 2, 0, Phi.shape[1] - w)
			idz = jnp.clip(jnp.searchsorted(xs[2], z_center) - w // 2, 0, Phi.shape[2] - w)
			Phi_cropped = Phi[idx:idx + w, idy:idy + w, idz:idz + w]

			x_min, x_max = xs[0][idx], xs[0][idx + w - 1]
			y_min, y_max = xs[1][idy], xs[1][idy + w - 1]
			z_min, z_max = xs[2][idz], xs[2][idz + w - 1]

			ax_x.set_xlim(x_min, x_max)
			ax_x.set_ylim(y_min, y_max)
			ax_x.set_zlim(z_min, z_max)

			for ts in list(ax_x.collections):
				ts.remove()
			for ts in list(ax_x.texts):
				ts.remove()

			phi_q3 = jnp.quantile(Phi_cropped, 0.03)
			phi_q97 = jnp.quantile(Phi_cropped, 0.97)
			for level, c in zip(levels, colors):
				if level < phi_q3 or level > phi_q97:
					continue
				try:
					verts, faces, _normals, _values_on_surface = marching_cubes(np.array(Phi_cropped), level=level, spacing=spacings, step_size=int(w / 20) + 1)
				except Exception as e:
					print(f"Error in marching_cubes: {e} for level {level}, skipping this level.")
					continue
				ax_x.plot_trisurf(verts[:, 0] + x_min, verts[:, 1] + y_min, faces, verts[:, 2] + z_min, lw=1, alpha=0.6, color=c)
			if not show_cells:
				return

			assert field.tesselation is not None, "Tesselation not constructed, cannot plot cells."
			inside_vertex_mask = jnp.all((jnp.array([x_min, y_min, z_min]) <= field.discr_points) & (field.discr_points <= jnp.array([x_max, y_max, z_max])), axis=-1)
			cell_ids = jnp.arange(field.num_vert)[inside_vertex_mask]
			plot_tesselation(field.tesselation, ax=ax_x, cell_ids=cell_ids, batch_potential_fun=field.batch_potential, **cell_plot_args)

		if allow_widgets:
			plot_cropped_region(slider_x.val, slider_y.val, slider_z.val, slider_mag.val)
		else:
			plot_cropped_region(slider_x_init, slider_y_init, slider_z_init, mag_init)

		dummies = [ax_x.plot_trisurf([0, 1, 0], [0, 0, 1], [0, 1, 2], [1, 0, 0], color=c, lw=1, alpha=0.6, label=fr'${level: .2f}$') for level, c in zip(levels, colors)]
		ax_x.legend(loc='upper right', title=r'Isosurfaces $\Phi=$', frameon=True, framealpha=0.5)
		for ts in dummies:
			ts.remove()

		ax_x.set_xlabel(r'$x$')
		ax_x.set_ylabel(r'$y$')
		ax_x.set_zlabel(r'$z$')
		ax_x.set_aspect('equal')

		if allow_widgets:
			def update_3d(_val):
				"""Update 3d.
				
				Args:
					_val: Input parameter.
				"""
				plot_cropped_region(slider_x.val, slider_y.val, slider_z.val, slider_mag.val)
				fig.canvas.draw_idle()

			slider_x.on_changed(update_3d)
			slider_y.on_changed(update_3d)
			slider_z.on_changed(update_3d)
			slider_mag.on_changed(update_3d)

			plt.plot()
			return fig, (slider_x, slider_y, slider_z, slider_mag)

	cell_collections = []
	shift_vectors = field.discr_points_extended - field.discr_points[field.discr_indices_extended]
	if plot_dim < 3 and show_cells:
		assert field.tesselation is not None, "Tesselation not constructed, cannot plot cells."
		if ax_x is not None:
			ax_x.set_xlim(*ax_x.get_xlim())
			ax_x.set_ylim(*ax_x.get_ylim())
			z_level = None
			if slider_x is not None:
				z_level = slider_x.val
			elif plot_dim != field.dim:
				z_level = slider_z_init
			cell_ids = field.discr_indices_extended
			cell_collections = plot_tesselation(
				field.tesselation,
				ax=ax_x,
				cell_ids=cell_ids,
				shift_vectors=shift_vectors,
				batch_potential_fun=field.batch_potential,
				z_level=z_level,
				size_decay_length=0.5 * field.lambda0,
				**cell_plot_args
			)

	if ax_k is not None:
		if plot_dim < 3:
			ax_k.xaxis.set_major_locator(MaxNLocator(nbins=5))
			ax_k.yaxis.set_major_locator(MaxNLocator(nbins=5))
	if ax_x is not None:
		if plot_dim < 3:
			ax_x.xaxis.set_major_locator(MaxNLocator(nbins=5))
			ax_x.yaxis.set_major_locator(MaxNLocator(nbins=5))

	if ax_k is not None and ax_x is not None:
		title_color = 'white' if plot_dim == 2 else 'black'
		ax_k.set_title('(a)', loc='left', x=0.05, y=1, pad=-15, color=title_color)	
		ax_x.set_title('(b)', loc='left', x=0.05, y=1, pad=-15)


	if plot_dim != field.dim and allow_widgets:
		assert fig is not None
		delta_k = field.ks[-1][1] - field.ks[-1][0]
		k_num = len(field.ks[-1])
		delta_x = xs[-1][1] - xs[-1][0]
		x_num = len(xs[-1])

		def update_k(_val):
			"""Update k.
			
			Args:
				_val: Input parameter.
			"""
			k_slider_ids = (round(slider_k.val / delta_k) + k_num // 2,)
			if plot_dim == 1:
				assert k_abs is not None and k_real is not None and k_im is not None
				if field.EA is not None:
					assert k_ev is not None
					k_ev.set_ydata(field.EA[:, *k_slider_ids])
				k_abs.set_offsets(jnp.stack([field.ks[0], absA[:, *k_slider_ids]], axis=1))
				k_real.set_offsets(jnp.stack([field.ks[0], jnp.real(field.A[:, *k_slider_ids])], axis=1))
				k_im.set_offsets(jnp.stack([field.ks[0], jnp.imag(field.A[:, *k_slider_ids])], axis=1))
			if plot_dim == 2:
				assert k_plot is not None
				k_plot.set_data(absA[:, :, *(k_slider_ids)].T)
			fig.canvas.draw_idle()

		def update_x(_val):
			"""Update x.
			
			Args:
				_val: Input parameter.
			"""
			assert slider_x is not None and x_plot is not None
			x_slider_ids = (round(slider_x.val / delta_x) + x_num // 2,)
			if plot_dim == 1:
				assert isinstance(x_plot, Line2D)
				x_plot.set_ydata(Phi[:, *x_slider_ids])
			if plot_dim == 2:
				x_plot.set_data(Phi[:, :, *(x_slider_ids)].T)
			if show_cells:
				assert field.tesselation is not None and ax_x is not None
				for ts in cell_collections:
					ts.remove()
				z_level = slider_x.val
				object_list = plot_tesselation(
					field.tesselation,
					ax=ax_x,
					cell_ids=field.discr_indices_extended,
					shift_vectors=shift_vectors,
					batch_potential_fun=field.batch_potential,
					z_level=z_level,
					size_decay_length=0.5 * field.lambda0,
					**cell_plot_args
				)
				cell_collections.clear()
				cell_collections.extend(object_list)

			fig.canvas.draw_idle()

		assert slider_x is not None and slider_k is not None
		slider_k.on_changed(update_k)
		slider_x.on_changed(update_x)

		plt.show()
		return fig, (slider_k, slider_x)

	return fig, ()


def plot_stationary_distribution_test(
	hist: Array,
	p_theory: Array,
	n_samples: Int,
	alpha: float = 0.2,
	ax: Axes | None = None
) -> tuple[object, Axes]:
	"""Plot observed vs expected stationary distribution counts."""
	if ax is None:
		fig, ax = plt.subplots(figsize=(7, 5))
	else:
		fig = ax.get_figure()
	ax.scatter(jnp.ravel(p_theory * n_samples), jnp.ravel(hist), alpha=alpha, label="data")
	ax.set_xlabel("Expected counts")
	ax.set_ylabel("Observed counts")
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', label="y=x")
	xs = jnp.linspace(*ax.get_xlim(), 100)
	ax.plot(xs, xs + 1.96 * jnp.sqrt(xs * (1 - xs / n_samples)), 'k:', label="95% CI")
	ax.plot(xs, xs - 1.96 * jnp.sqrt(xs * (1 - xs / n_samples)), 'k:')
	ax.legend()
	return fig, ax


def plot_stationary_distribution_sample(
	field: "Field",
	n_points: Int = 10000,
	scatter_args: PyTree = {"c": "C5", "alpha": 0.2, "s": 2},
	beta: Float = 1.0,
	res: Int = 30,
	seed: Int = 42,
	ax: Axes | None = None
) -> tuple[object, Axes]:
	"""Plot sampled points over the field background."""
	assert field.dim == 2, NotImplementedError("plot_stationary_distribution_sample is only implemented for 2D fields.")
	if ax is None:
		fig, ax = plt.subplots(figsize=(7, 5))
	else:
		fig = ax.get_figure()
	key = jr.key(seed)
	samples = field.sample_stationary_distribution(key=key, n_samples=n_points, beta=beta)
	plot_field(field, ax_x=ax, ax_k=None, res=res, plot_dim=2, show_cells=False, verbose=False)
	ax.scatter(samples[:, 0], samples[:, 1], **scatter_args, label="sampled points")
	ax.legend(frameon=True, framealpha=0.8, loc='upper right')
	return fig, ax
