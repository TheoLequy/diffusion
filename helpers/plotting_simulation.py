from matplotlib.widgets import Slider, RadioButtons, CheckButtons
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import os
import shutil
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import zarr
from time import perf_counter
from jaxtyping import Float, Array, Int
from typing import Any, Collection, TYPE_CHECKING, cast

from helpers.plotting_fields import plot_field, plot_tesselation, COLORS
from helpers.plotting_graphs import (
    _get_vertex_position,
    _darken_color,
    _lighten_color,
    _build_labeled_digraph,
)
from helpers.fullgraph_inference import propagator_from_laplacian, dehermitianize

if TYPE_CHECKING:
	from coarse_graining.simulations import Simulator


def plot_simulation_sample(
	simulator: "Simulator",
	ax: Axes | None = None,
	ax_pot: Axes | None = None,
	n_sample: Int = 3,
	ids: Int[Array, "n_sample"] | Collection[Int] | None = None,
	discrete: bool | None = None,
	seed: Int = 42,
	colors: list[str] = COLORS,
	line_styles: list[str] = ["-", "--", "-.", ":"],
	plot_args: dict = {"zorder": 10},
	plot_field_flag: bool | None = None,
	field_args: dict = {},
	res: Int | None = None,
	linewidth: float = 1.0,
	decay_constant: float = 0.2,
	darken: float = 0.2,
	arrow_darken: float = 0.4,
	video_dir: str | None = None,
) -> object | None:
	"""Plot sample trajectories with optional field background."""
	if discrete is None:
		discrete = not ("traj_cont" in simulator.zarr_root.array_keys())
	if not discrete and "traj_cont" not in simulator.zarr_root.array_keys():
		raise ValueError("No continuous trajectory data found in the zarr store.")
	if discrete and "traj_disc" not in simulator.zarr_root.array_keys():
		raise ValueError("No discretized trajectory data found in the zarr store.")

	if res is None:
		res = float(50.0 / simulator.field.lambda0)

	if plot_field_flag is None:
		plot_field_flag = (simulator.field.dim <= 2)

	if not (0.0 <= darken < 1.0):
		raise ValueError("darken must be in [0, 1).")
	if not (0.0 <= arrow_darken < 1.0):
		raise ValueError("arrow_darken must be in [0, 1).")

	def _plot_sample_1d(ax_traj: Axes) -> list:
		"""Internal helper to plot sample 1d.
		
		Args:
			ax_traj: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		if ids is None:
			chosen_ids = jr.choice(key=jr.key(seed), a=jnp.arange(simulator.n_particles), shape=(n_sample,), replace=False)
		else:
			chosen_ids = ids
		lines = []
		if discrete:
			traj_disc = simulator.zarr_root["traj_disc"]
			assert isinstance(traj_disc, zarr.Array)
			trajs = [simulator.field.discr_points[jnp.array(traj_disc[int(i)], dtype=int)] for i in chosen_ids]
		else:
			traj_cont = simulator.zarr_root["traj_cont"]
			assert isinstance(traj_cont, zarr.Array)
			trajs = [jnp.array(traj_cont[int(i), :]) for i in chosen_ids]

		for i, (pid, traj) in enumerate(zip(chosen_ids, trajs)):
			c = colors[i % len(colors)]
			ls = line_styles[i % len(line_styles)]
			trajectory = jnp.array(traj)
			grid_ids = ((trajectory + simulator.field.size / 2) // simulator.field.size).astype(int)
			first = True
			for grid_id in jnp.unique(grid_ids, axis=0):
				mask = (grid_ids[:, 0] == grid_id[0])
				mask = mask | jnp.roll(mask, axis=0, shift=1) | jnp.roll(mask, axis=0, shift=-1)
				masked_traj = trajectory.at[~mask, 0].set(jnp.nan) - grid_id * simulator.field.size
				if first:
					first = False
					line, = ax_traj.plot(simulator.ts, masked_traj, lw=2, label=f"particle {pid}", c=c, ls=ls)
				else:
					line, = ax_traj.plot(simulator.ts, masked_traj, lw=2, c=c, ls=ls)
				lines.append(line)
		ax_traj.legend(frameon=True, framealpha=0.5, loc="lower right")
		ax_traj.grid(True)
		return lines

	def _plot_sample_2d(ax_traj: Axes, ax_bfield: Axes | None = None, ax_slider: Axes | None = None) -> object:
		"""Internal helper to plot sample 2d.
		
		Args:
			ax_traj: Input parameter.
			ax_bfield: Input parameter.
			ax_slider: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		x_min = float(-simulator.field.size[0] / 2)
		x_max = float(simulator.field.size[0] / 2)
		y_min = float(-simulator.field.size[1] / 2)
		y_max = float(simulator.field.size[1] / 2)

		def _lock_xy_limits() -> None:
			"""Internal helper to lock xy limits.
			"""
			ax_traj.set_xlim(x_min, x_max)
			ax_traj.set_ylim(y_min, y_max)
			ax_traj.set_aspect("equal", adjustable="box")
			ax_traj.margins(x=0.0, y=0.0)
			ax_traj.set_autoscale_on(False)

		if ids is None:
			chosen_ids = jr.choice(key=jr.key(seed), a=jnp.arange(simulator.n_particles), shape=(n_sample,), replace=False)
		else:
			chosen_ids = ids

		_lock_xy_limits()

		has_bfield_phase = (not discrete) and (simulator.Bfield is not None)
		if has_bfield_phase:
			traj_cont = simulator.zarr_root["traj_cont"]
			assert isinstance(traj_cont, zarr.Array)
			if traj_cont.shape[-1] < 3:
				has_bfield_phase = False

		if has_bfield_phase:
			traj_cont = simulator.zarr_root["traj_cont"]
			assert isinstance(traj_cont, zarr.Array)
			times = np.asarray(simulator.ts, dtype=float)
			n_times = int(traj_cont.shape[1])
			if n_times != len(times):
				times = np.linspace(0.0, float(simulator.t_end), n_times)

			trajs_pos = [np.asarray(traj_cont[int(i), :, :2], dtype=float) for i in chosen_ids]
			trajs_phi = [np.asarray(traj_cont[int(i), :, 2], dtype=float) for i in chosen_ids]

			_lock_xy_limits()
			ax_traj.set_xlabel(r"$x$")
			ax_traj.set_ylabel(r"$y$")

			if plot_field_flag:
				field_args_local = dict(field_args)
				field_args_local.setdefault("show_cells", False)
				field_args_local.setdefault("bw", True)
				plot_field(simulator.field, ax_x=ax_traj, plot_dim=2, **field_args_local)
				_lock_xy_limits()
			if ax_bfield is not None:
				ax_traj.set_ylabel("")
				ax_traj.tick_params(axis="y", which="both", left=False, labelleft=False)

			bfield_artist: tuple[Any, np.ndarray] | None = None
			bfield_xlim: tuple[float, float] | None = None
			b_vals_all_np: np.ndarray | None = None
			ys_np: np.ndarray | None = None
			if ax_bfield is not None:
				bfield_callable = simulator.Bfield
				assert bfield_callable is not None
				n_y = int(max(40, min(120, res)))
				ys = jnp.linspace(float(-simulator.field.size[1] / 2), float(simulator.field.size[1] / 2), n_y)
				times_jax = jnp.asarray(times, dtype=ys.dtype)

				def _b_eval_single(t_val, y_val):
					"""Internal helper to b eval single.
					
					Args:
						t_val: Input parameter.
						y_val: Input parameter.
					
					Returns:
					Output value computed by this function.
					"""
					return jnp.asarray(bfield_callable(t_val, 0.0, y_val), dtype=ys.dtype)

				b_vals_all = jax.vmap(lambda t_val: jax.vmap(lambda y_val: _b_eval_single(t_val, y_val))(ys))(times_jax)
				b_vals_all_np = np.asarray(b_vals_all, dtype=float)
				ys_np = np.asarray(ys, dtype=float)

				line, = ax_bfield.plot(b_vals_all_np[0], ys_np, lw=1.0)
				b_min = float(np.min(b_vals_all_np))
				b_max = float(np.max(b_vals_all_np))
				if b_min == b_max:
					delta = max(1e-8, abs(b_min) * 0.1 + 1e-8)
					b_min, b_max = b_min - delta, b_max + delta
				pad = 0.05 * (b_max - b_min)
				bfield_xlim = (b_min - pad, b_max + pad)
				ax_bfield.axvline(0.0, color="0.7", lw=0.8, zorder=0)
				ax_bfield.set_xlabel(r"$B$")
				ax_bfield.set_ylabel("")
				ax_bfield.set_box_aspect(3)
				ax_bfield.set_xlim(*bfield_xlim)
				bfield_artist = (line, ys_np)

			dynamic_artists: list[Any] = []
			hsv_cmap = plt.get_cmap("hsv")
			arrow_length = 1.2 * 0.04 * float(jnp.min(simulator.field.size))
			plot_args_local = dict(plot_args)
			plot_args_local.pop("c", None)
			plot_args_local.pop("color", None)
			plot_args_local.pop("ls", None)

			def _update(idx: int) -> None:
				"""Internal helper to update.
				
				Args:
					idx: Input parameter.
				"""
				nonlocal dynamic_artists
				idx = int(np.clip(idx, 0, n_times - 1))
				t_now = float(times[idx])
				for artist in dynamic_artists:
					try:
						artist.remove()
					except ValueError:
						pass
				dynamic_artists = []

				for pos, phi in zip(trajs_pos, trajs_phi):
					if idx == 0:
						phi_now = float(phi[0])
						hue = (phi_now % (2 * np.pi)) / (2 * np.pi)
						rgba = np.asarray(hsv_cmap(hue))
						rgba[:3] *= (1.0 - darken)
						sc = ax_traj.scatter([pos[0, 0]], [pos[0, 1]], c=[rgba], s=30, zorder=20)
						dynamic_artists.append(sc)
						rgba_arrow = rgba.copy()
						rgba_arrow[:3] *= (1.0 - arrow_darken)
						vec = np.array([-np.sin(phi_now), np.cos(phi_now)], dtype=float)
						qu = ax_traj.quiver([pos[0, 0]], [pos[0, 1]], [arrow_length * vec[0]], [arrow_length * vec[1]], angles="xy", scale_units="xy", scale=1.0, pivot="mid", width=0.004, zorder=21, color=[rgba_arrow])
						dynamic_artists.append(qu)
						continue

					pos_hist = pos[:idx + 1]
					phi_hist = phi[:idx + 1]
					t_hist = times[:idx + 1]

					p0 = pos_hist[:-1]
					p1 = pos_hist[1:]
					dp = np.abs(p1 - p0)
					jump_mask = (dp[:, 0] > float(simulator.field.size[0]) / 2) | (dp[:, 1] > float(simulator.field.size[1]) / 2)
					valid = ~jump_mask
					if np.any(valid):
						segments = np.stack([p0[valid], p1[valid]], axis=1)
						phi_mid = 0.5 * (phi_hist[:-1] + phi_hist[1:])[valid]
						t_mid = 0.5 * (t_hist[:-1] + t_hist[1:])[valid]
						hue = np.mod(phi_mid, 2 * np.pi) / (2 * np.pi)
						rgba = np.asarray(hsv_cmap(hue))
						rgba[:, :3] *= (1.0 - darken)
						rgba[:, 3] = np.clip(np.exp(decay_constant * (t_mid - t_now)), 0.0, 1.0)
						lc = mcollections.LineCollection(list(segments), colors=rgba, linewidths=linewidth, **plot_args_local)
						ax_traj.add_collection(lc)
						dynamic_artists.append(lc)

					p_now = pos_hist[-1]
					phi_now = float(phi_hist[-1])
					hue_now = (phi_now % (2 * np.pi)) / (2 * np.pi)
					rgba_now = np.asarray(hsv_cmap(hue_now))
					rgba_now[:3] *= (1.0 - darken)
					sc = ax_traj.scatter([p_now[0]], [p_now[1]], c=[rgba_now], s=30, zorder=20)
					dynamic_artists.append(sc)
					rgba_arrow_now = rgba_now.copy()
					rgba_arrow_now[:3] *= (1.0 - arrow_darken)
					vec = np.array([-np.sin(phi_now), np.cos(phi_now)], dtype=float)
					qu = ax_traj.quiver([p_now[0]], [p_now[1]], [arrow_length * vec[0]], [arrow_length * vec[1]], angles="xy", scale_units="xy", scale=1.0, pivot="mid", width=0.004, zorder=21, color=[rgba_arrow_now])
					dynamic_artists.append(qu)

				if bfield_artist is not None:
					line, ys = bfield_artist
					if b_vals_all_np is not None:
						line.set_data(b_vals_all_np[idx], ys)
					if ax_bfield is not None and bfield_xlim is not None:
						ax_bfield.set_xlim(*bfield_xlim)

				if ax_slider is not None:
					ax_slider.set_title(rf"$t={t_now:.3g}$", pad=-10)

				canvas = ax_traj.figure.canvas
				if canvas is not None:
					canvas.draw_idle()

			t_slider = None
			if ax_slider is not None:
				t_slider = Slider(ax_slider, r"$t$", valmin=0, valmax=n_times - 1, valinit=0, valstep=1)

				def _on_slider(val: float) -> None:
					"""Handle the slider event.
					
					Args:
						val: Input parameter.
					"""
					_update(int(val))

				t_slider.on_changed(_on_slider)
				ax_slider.set_title(r"$t=0$", pad=-10)
				fig_any = cast(Any, ax_traj.figure)
				widget_refs = getattr(fig_any, "_diffsim_widget_refs", [])
				widget_refs.extend([t_slider, _on_slider])
				setattr(fig_any, "_diffsim_widget_refs", widget_refs)

			_update(0)
			_lock_xy_limits()

			video_path: str | None = None
			if video_dir is not None:
				os.makedirs(video_dir, exist_ok=True)
				file_stem = f"{simulator.name}_spin_sample"
				video_mp4 = os.path.join(video_dir, f"{file_stem}.mp4")

				def _animate(frame_idx: int):
					"""Internal helper to animate.
					
					Args:
						frame_idx: Input parameter.
					
					Returns:
					Output value computed by this function.
					"""
					_update(int(frame_idx))
					return tuple(dynamic_artists)

				fig_anim = ax_traj.figure
				if not isinstance(fig_anim, Figure):
					raise ValueError("Video export requires a matplotlib Figure.")
				animation = FuncAnimation(fig_anim, _animate, frames=n_times, interval=80, blit=False, repeat=False)
				ffmpeg_exe = shutil.which("ffmpeg")
				if ffmpeg_exe is None:
					try:
						import imageio_ffmpeg
						ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
					except Exception:
						ffmpeg_exe = None
				if ffmpeg_exe is None:
					raise RuntimeError("Failed to save MP4 video. Install system ffmpeg or run `pip install imageio-ffmpeg` in the active environment.")
				prev_ffmpeg_path = mpl.rcParams.get("animation.ffmpeg_path", "ffmpeg")
				mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
				try:
					animation.save(
						video_mp4,
						writer=FFMpegWriter(fps=8, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]),
						dpi=110,
					)
				except Exception as exc:
					raise RuntimeError("Failed to save MP4 video even after locating ffmpeg executable.") from exc
				finally:
					mpl.rcParams["animation.ffmpeg_path"] = prev_ffmpeg_path
				video_path = video_mp4

			return {
				"ax": ax_traj,
				"ax_bfield": ax_bfield,
				"ax_slider": ax_slider,
				"slider": t_slider,
				"video_path": video_path,
				"artists": dynamic_artists,
			}

		if discrete:
			traj_disc = simulator.zarr_root["traj_disc"]
			assert isinstance(traj_disc, zarr.Array)
			trajs = [simulator.field.discr_points[jnp.array(traj_disc[int(i)], dtype=int)] for i in chosen_ids]
		else:
			traj_cont = simulator.zarr_root["traj_cont"]
			assert isinstance(traj_cont, zarr.Array)
			trajs = [jnp.array(traj_cont[int(i), :]) for i in chosen_ids]

		lines = []
		for i, (pid, traj) in enumerate(zip(chosen_ids, trajs)):
			c = colors[i % len(colors)]
			ls = line_styles[i % len(line_styles)]
			trajectory = jnp.array(traj)
			grid_ids = ((trajectory + simulator.field.size / 2) // simulator.field.size).astype(int)
			first = True
			for grid_id in jnp.unique(grid_ids, axis=0):
				mask = (grid_ids[:, 0] == grid_id[0]) & (grid_ids[:, 1] == grid_id[1])
				mask = mask | jnp.roll(mask, axis=0, shift=1) | jnp.roll(mask, axis=0, shift=-1)
				masked_traj = trajectory.at[~mask, 0].set(jnp.nan) - grid_id * simulator.field.size
				if first:
					first = False
					line, = ax_traj.plot(*masked_traj.T, lw=linewidth, label=f"particle {pid}", c=c, ls=ls, **plot_args)
				else:
					line, = ax_traj.plot(*masked_traj.T, lw=linewidth, c=c, ls=ls, **plot_args)
				lines.append(line)
		ax_traj.legend(frameon=True, framealpha=0.8, loc='lower left')
		_lock_xy_limits()
		return lines

	def _plot_sample_3d(ax_traj: Axes) -> list:
		"""Internal helper to plot sample 3d.
		
		Args:
			ax_traj: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		if ids is None:
			chosen_ids = jr.choice(key=jr.key(seed), a=jnp.arange(simulator.n_particles), shape=(n_sample,), replace=False)
		else:
			chosen_ids = ids

		if discrete:
			traj_disc = simulator.zarr_root["traj_disc"]
			assert isinstance(traj_disc, zarr.Array)
			trajs = [simulator.field.discr_points[jnp.array(traj_disc[int(i)], dtype=int)] for i in chosen_ids]
		else:
			traj_cont = simulator.zarr_root["traj_cont"]
			assert isinstance(traj_cont, zarr.Array)
			trajs = [jnp.array(traj_cont[int(i), :]) for i in chosen_ids]

		lines = []
		for i, (pid, traj) in enumerate(zip(chosen_ids, trajs)):
			c = colors[i % len(colors)]
			ls = line_styles[i % len(line_styles)]
			trajectory = jnp.array(traj)
			grid_ids = ((trajectory + simulator.field.size / 2) // simulator.field.size).astype(int)
			first = True
			for grid_id in jnp.unique(grid_ids, axis=0):
				mask = jnp.all(grid_ids == grid_id, axis=1)
				mask = mask | jnp.roll(mask, axis=0, shift=1) | jnp.roll(mask, axis=0, shift=-1)
				masked_traj = trajectory.at[~mask, 0].set(jnp.nan) - grid_id * simulator.field.size
				if first:
					first = False
					line, = ax_traj.plot(*masked_traj.T, lw=linewidth, label=f"particle {pid}", c=c, ls=ls, **plot_args)
				else:
					line, = ax_traj.plot(*masked_traj.T, lw=linewidth, c=c, ls=ls, **plot_args)
				lines.append(line)
		ax_traj.legend(frameon=True, framealpha=0.5)
		return lines

	if simulator.field.dim == 1:
		if ax is None and ax_pot is None:
			fig, (ax_pot, ax_traj) = plt.subplots(figsize=(10, 5), ncols=2, sharey=True, width_ratios=[0.1, 0.9])
		else:
			ax_traj = ax
		if ax_traj is not None:
			xs = jnp.linspace(-simulator.field.size[0] / 2, simulator.field.size[0] / 2, int(res * simulator.field.size[0] + 1))
			ax_traj.set_xlabel(r'$t$')
			ax_traj.set_ylim(float(xs[0]), float(xs[-1]))

		if plot_field_flag and ax_pot is not None:
			plot_field(simulator.field, ax_x=ax_pot, flip_potential_axes=True, **field_args, show_cells=False)
			legend = ax_pot.get_legend()
			if legend is not None:
				legend.remove()
			ax_pot.set_title('(a)', loc='left', x=0.05, y=1, pad=-15)
		if ax_traj is not None:
			if ax_pot is not None:
				ax_traj.set_title('(b)', loc='left', x=0.05, y=1, pad=-15)
			return _plot_sample_1d(ax_traj)

	elif simulator.field.dim == 2:
		use_bfield_layout = (not discrete) and (simulator.Bfield is not None) and (ax is None)
		if use_bfield_layout:
			size_x = float(simulator.field.size[0])
			size_y = float(simulator.field.size[1])
			field_aspect = max(0.3, min(3.0, size_y / size_x if size_x > 0 else 1.0))
			fig_width = 12.0
			fig_height = 9.
			fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
			gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[20, 1], figure=fig)
			ax_bfield = fig.add_subplot(gs[0, 0])
			ax = fig.add_subplot(gs[0, 1], sharey=ax_bfield)
			ax_slider = fig.add_subplot(gs[1, :])
			return _plot_sample_2d(ax, ax_bfield=ax_bfield, ax_slider=ax_slider)
		else:
			if ax is None:
				if ax_pot is None:
					fig, ax = plt.subplots(figsize=(6, 6))
				else:
					ax = ax_pot
			if plot_field_flag and ax is not None:
				plot_field(simulator.field, ax_x=ax, plot_dim=2, **field_args)
				legend = ax.get_legend()
				if legend is not None:
					legend.remove()
			if ax is not None:
				return _plot_sample_2d(ax)

	elif simulator.field.dim == 3:
		if ax is None:
			if ax_pot is None:
				fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
			else:
				ax = ax_pot
		if plot_field_flag and ax is not None:
			plot_field(simulator.field, ax_x=ax, plot_dim=3, **field_args)
		if ax is not None:
			ax.set_xlabel(r'$x$')
			ax.set_ylabel(r'$y$')
			ax.set_zlabel(r'$z$')
			return _plot_sample_3d(ax)
	else:
		raise NotImplementedError("Plotting only implemented for 1, 2 and 3 dimensional fields.")

	return None


def display_cell_scalar_field(
	ax: Axes,
	values: np.ndarray,
	show_cell_ids: bool = True,
	cmap: str = "viridis"
) -> mcollections.PolyCollection:
	"""Display scalar values on tessellation cells (including periodic images)."""
	field = getattr(ax, "_diffsim_field", None)
	if field is None:
		raise ValueError("Axis is missing '_diffsim_field' attribute with a Field instance.")
	if field.tesselation is None:
		raise ValueError("Field tesselation is required.")

	values_arr = np.asarray(values, dtype=float)
	num_cells = int(field.num_vert)
	if values_arr.shape[0] != num_cells:
		raise ValueError(f"values must have length {num_cells}, got {values_arr.shape[0]}.")

	cell_ids = field.discr_indices_extended
	shift_vectors = field.discr_points_extended - field.discr_points[field.discr_indices_extended]
	plot_artists = plot_tesselation(
		field.tesselation,
		cell_ids=cell_ids,
		shift_vectors=shift_vectors,
		ax=ax,
		edge_alpha=0.5,
		face_alpha=1.0,
		center_marker_size=0.0,
		annotation_size=0.0
	)
	if len(plot_artists) == 0 or not isinstance(plot_artists[-1], mcollections.PolyCollection):
		raise ValueError("Failed to create PolyCollection for tessellation plot.")
	poly_collection = plot_artists[-1]

	base_cell_ids = np.asarray(cell_ids, dtype=int)
	setattr(poly_collection, "_diffsim_base_cell_ids", base_cell_ids)
	setattr(poly_collection, "_diffsim_cmap", cmap)
	cmap_obj = plt.get_cmap(cmap)
	plot_values = values_arr[base_cell_ids]
	finite_vals = np.isfinite(plot_values)
	if np.any(finite_vals):
		vmin = float(np.min(plot_values[finite_vals]))
		vmax = float(np.max(plot_values[finite_vals]))
		if vmin == vmax:
			vmin -= 0.5
			vmax += 0.5
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		facecolors = np.asarray(cmap_obj(norm(np.where(finite_vals, plot_values, vmin))))
	else:
		facecolors = np.asarray(cmap_obj(np.zeros_like(plot_values, dtype=float)))
	facecolors[~finite_vals, 3] = 0.0
	poly_collection.set_facecolors(facecolors)  # type: ignore[attr-defined]

	if show_cell_ids:
		centers = np.asarray(field.tesselation.centers[cell_ids] + shift_vectors)
		for center, cell_id in zip(centers, base_cell_ids):
			ax.text(
				float(center[0]),
				float(center[1]),
				str(int(cell_id)),
				ha="center",
				va="center",
				fontsize=14,
				clip_on=True,
				color="k",
				zorder=20
			)

	return poly_collection


def update_cell_scalar_field(
	ax: Axes,
	poly_collection: mcollections.PolyCollection,
	values: np.ndarray,
	norm: mcolors.Normalize,
	cmap: mcolors.Colormap,
	blit_state: dict[str, object] | None = None,
	use_blit: bool = True,
	scale: str = "log"
) -> None:
	"""Update PolyCollection facecolors from scalar values using linear or log scaling."""
	base_cell_ids = getattr(poly_collection, "_diffsim_base_cell_ids", None)
	if base_cell_ids is None:
		raise ValueError("poly_collection is missing '_diffsim_base_cell_ids'.")

	values_arr = np.asarray(values, dtype=float)
	if values_arr.ndim != 1:
		raise ValueError("values must be a 1D array.")
	if int(np.max(base_cell_ids)) >= values_arr.shape[0]:
		raise ValueError("values length is incompatible with mapped cell ids.")

	plot_values = values_arr[np.asarray(base_cell_ids, dtype=int)]
	if scale == "log":
		with np.errstate(divide="ignore", invalid="ignore"):
			mapped_values = np.log10(plot_values)
	elif scale == "linear":
		mapped_values = plot_values
	else:
		raise ValueError(f"Unknown scale '{scale}'. Expected 'log' or 'linear'.")

	finite_mask = np.isfinite(mapped_values)
	fallback_value = float(norm.vmin) if norm.vmin is not None else 0.0
	colors = cmap(norm(np.where(finite_mask, mapped_values, fallback_value)))
	colors = np.asarray(colors)
	colors[~finite_mask, 3] = 0.0
	poly_collection.set_facecolors(colors)  # type: ignore[attr-defined]

	background = None if blit_state is None else blit_state.get("background")
	canvas = ax.figure.canvas
	if use_blit and background is not None and canvas is not None and canvas.supports_blit and hasattr(canvas, "restore_region"):
		canvas.restore_region(background)
		ax.draw_artist(poly_collection)
		canvas.blit(ax.bbox)
		canvas.flush_events()
		return

	ax.figure.canvas.draw_idle()


def plot_simulation_propagator(
	simulator: "Simulator",
	vert0_id: Int,
	n_subset_particles: Int | None = None,
	n_snapshots: Int = 2**10,
	ax: Axes | None = None,
	ax_slider: Axes | None = None,
	ax_check: Axes | None = None,
	t_index_init: int = 0,
	show_data: bool = True,
	show_model: bool = False,
	show_cell_ids: bool = True,
	posterior_init: bool = False,
	verbose: bool = True,
) -> tuple[object, Axes, Slider | None, CheckButtons | None]:
	"""Plot propagator slices from saved zarr data with fast interactive updates."""
	if verbose:
		print("Loading propagator data...")
	if simulator.field.tesselation is None:
		raise ValueError("Tesselation must be defined in the field to plot the propagator.")
	if simulator.field.dim != 2:
		raise NotImplementedError("plot_propagator is only implemented for 2D fields.")
	if "propagator" not in simulator.zarr_root.array_keys():
		raise ValueError("No propagator array found in the zarr store.")

	propagator_array = simulator.zarr_root["propagator"]
	assert isinstance(propagator_array, zarr.Array)
	propagator = np.asarray(propagator_array[:, :, :, 0], dtype=float)
	if propagator.ndim != 3:
		raise ValueError(f"Expected propagator with 3 dimensions after slicing [...,0], got shape {propagator.shape}.")

	n_t, n_in, n_out = propagator.shape
	if n_in != n_out:
		raise ValueError(f"Expected square in/out cell dimensions, got {n_in} and {n_out}.")

	num_cells = int(simulator.field.num_vert)
	if n_in != num_cells:
		raise ValueError(f"Propagator cell count {n_in} does not match field cell count {num_cells}.")

	if "delta_t" not in propagator_array.attrs:
		raise ValueError("propagator array is missing attribute 'delta_t'.")
	delta_t_steps = float(cast(float, propagator_array.attrs["delta_t"]))
	delta_t = delta_t_steps * float(simulator.dt)

	propagator_plot = np.clip(propagator, 0.0, np.inf)
	row_sums0 = np.sum(propagator_plot[0], axis=-1)
	has_t0_data = bool(np.any(np.isfinite(row_sums0) & (row_sums0 > 1e-12)))
	if has_t0_data:
		times_data = np.arange(n_t, dtype=float) * delta_t
		if verbose:
			print("Using stored propagator at t=0.")
	else:
		identity = np.eye(num_cells, dtype=float)
		propagator_plot = np.concatenate([identity[None, :, :], propagator_plot], axis=0)
		times_data = np.arange(propagator_plot.shape[0], dtype=float) * delta_t
		if verbose:
			print("No reliable t=0 data found; prepending delta initial condition.")

	n_t_plot = propagator_plot.shape[0]
	n_snapshots = int(min(max(n_snapshots, 1), n_t_plot))
	snapshot_selection = np.linspace(0, n_t_plot - 1, n_snapshots, dtype=int)
	snapshot_indices = np.unique(snapshot_selection)
	snapshot_times = times_data[snapshot_indices]
	t_index_init = int(np.clip(t_index_init, 0, len(snapshot_indices) - 1))
	if verbose:
		print(f"Prepared {len(snapshot_indices)} propagator snapshots.")

	vert0_id = int(vert0_id)
	if not (0 <= vert0_id < num_cells):
		raise ValueError(f"vert0_id must be in [0, {num_cells}).")

	if not show_data and not show_model:
		show_data = True

	use_widgets = ax is None and ax_slider is None and ax_check is None
	if use_widgets:
		fig = plt.figure(figsize=(10, 10), constrained_layout=True)
		gs = GridSpec(2, 2, height_ratios=[12, 1], width_ratios=[12, 2], figure=fig)
		ax = fig.add_subplot(gs[0, :])
		ax_slider = fig.add_subplot(gs[1, 0])
		ax_check = fig.add_subplot(gs[1, 1])
	else:
		if ax is None:
			raise ValueError("ax is required when custom widget axes are provided.")
		fig = ax.get_figure()

	assert ax is not None
	fig_any = cast(Any, fig)
	ax.set_xlim(float(-simulator.field.size[0] / 2), float(simulator.field.size[0] / 2))
	ax.set_ylim(float(-simulator.field.size[1] / 2), float(simulator.field.size[1] / 2))
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")

	setattr(ax, "_diffsim_field", simulator.field)

	stationary = None
	if "stationary" in simulator.zarr_root.array_keys():
		stationary_arr = simulator.zarr_root["stationary"]
		assert isinstance(stationary_arr, zarr.Array)
		stationary = np.asarray(stationary_arr[:, 0], dtype=float)
		if stationary.shape[0] != num_cells:
			raise ValueError(f"stationary has length {stationary.shape[0]}, expected {num_cells}.")
	else:
		stationary = np.asarray(simulator.field.get_discrete_stationary_distribution(), dtype=float)
		if stationary.shape[0] != num_cells:
			raise ValueError(f"fallback stationary has length {stationary.shape[0]}, expected {num_cells}.")

	model_available = False
	model_values_cache: dict[int, np.ndarray] = {}
	model_values_posterior_cache: dict[int, np.ndarray] = {}
	model_all: np.ndarray | None = None
	snapshot_times_model = jnp.asarray(snapshot_times, dtype=jnp.float32)

	def _ensure_model_available() -> None:
		"""Internal helper to ensure model available.
		
		Returns:
		Output value computed by this function.
		"""
		nonlocal model_all, model_available
		if model_available:
			return
		if verbose:
			print("Preparing model propagators on snapshot times...")
		t0_model = perf_counter()
		if stationary is None:
			return
		if "edge_weights" not in simulator.zarr_root.array_keys():
			simulator.fit_graph_model(plot=False, recompute=False)
		if "edge_weights" not in simulator.zarr_root.array_keys():
			return
		Lherm = jnp.asarray(np.asarray(simulator.zarr_root["edge_weights"], dtype=np.float32))
		pi_stat = jnp.asarray(stationary, dtype=jnp.float32)

		@jax.jit
		def _build_model_props(lap, ts, pi):
			"""Internal helper to build model props.
			
			Args:
				lap: Input parameter.
				ts: Input parameter.
				pi: Input parameter.
			
			Returns:
			Output value computed by this function.
			"""
			return dehermitianize(propagator_from_laplacian(lap, ts), pi)

		model_props = _build_model_props(Lherm, snapshot_times_model, pi_stat)
		_ = jax.block_until_ready(model_props)
		model_all = np.asarray(model_props, dtype=float)
		model_available = model_all.shape == (len(snapshot_indices), num_cells, num_cells)
		if verbose:
			print(f"Model propagators ready in {perf_counter() - t0_model:.2f}s.")

	if show_model:
		_ensure_model_available()

	def _compute_model_values(in_cell_id: int) -> np.ndarray:
		"""Internal helper to compute model values.
		
		Args:
			in_cell_id: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		_ensure_model_available()
		if not model_available:
			raise ValueError("Model propagator is not available.")
		if in_cell_id in model_values_cache:
			return model_values_cache[in_cell_id]

		assert model_all is not None
		values = np.clip(model_all[:, in_cell_id, :], 0.0, np.inf)
		sums = np.sum(values, axis=-1, keepdims=True)
		safe_values = np.where(sums > 0, values / sums, values)

		model_values_cache[in_cell_id] = safe_values
		return safe_values

	def _compute_model_values_posterior(in_cell_id: int) -> np.ndarray:
		"""Internal helper to compute model values posterior.
		
		Args:
			in_cell_id: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		if not model_available or stationary is None:
			raise ValueError("Posterior model propagator is not available.")
		if in_cell_id in model_values_posterior_cache:
			return model_values_posterior_cache[in_cell_id]
		vals = _compute_model_values(in_cell_id)
		with np.errstate(divide="ignore", invalid="ignore"):
			vals_post = vals * stationary[in_cell_id] / stationary[None, :]
		model_values_posterior_cache[in_cell_id] = vals_post
		return vals_post

	def _norm_from_logs(log_values: np.ndarray) -> mcolors.Normalize:
		"""Internal helper to norm from logs.
		
		Args:
			log_values: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		finite = np.isfinite(log_values)
		if np.any(finite):
			vmin_loc = float(np.min(log_values[finite]))
			vmax_loc = float(np.max(log_values[finite]))
		else:
			vmin_loc, vmax_loc = 0.0, 1.0
		if vmin_loc == vmax_loc:
			vmin_loc -= 0.5
			vmax_loc += 0.5
		return mcolors.Normalize(vmin=vmin_loc, vmax=vmax_loc)

	propagator_snapshots = propagator_plot[snapshot_indices, :, :]
	with np.errstate(divide="ignore", invalid="ignore"):
		all_logs_likelihood = np.log10(propagator_snapshots)
	norm_likelihood = _norm_from_logs(all_logs_likelihood)

	norm_posterior = norm_likelihood
	if stationary is not None:
		with np.errstate(divide="ignore", invalid="ignore"):
			posterior_snapshots = propagator_snapshots * stationary[None, :, None] / stationary[None, None, :]
		with np.errstate(divide="ignore", invalid="ignore"):
			all_logs_posterior = np.log10(posterior_snapshots)
		norm_posterior = _norm_from_logs(all_logs_posterior)
	if verbose:
		print("Computed color normalization ranges.")

	cmap_name = "viridis"
	cmap_obj = plt.get_cmap(cmap_name)

	poly_collection = display_cell_scalar_field(
		ax,
		values=propagator_plot[snapshot_indices[t_index_init], vert0_id, :],
		show_cell_ids=show_cell_ids,
		cmap=cmap_name
	)

	blit_state: dict[str, object] = {"background": None}

	state = {
		"in_cell_id": vert0_id,
		"snapshot_idx": t_index_init,
		"model": bool(show_model and model_available),
		"posterior": bool(posterior_init),
		"lock": False
	}
	if show_model and not model_available:
		state["model"] = False
	if state["posterior"] and stationary is None:
		state["posterior"] = False

	def _active_norm() -> mcolors.Normalize:
		"""Internal helper to active norm.
		
		Returns:
		Output value computed by this function.
		"""
		return norm_posterior if state["posterior"] else norm_likelihood

	def _colorbar_label() -> str:
		"""Internal helper to colorbar label.
		
		Returns:
		Output value computed by this function.
		"""
		in_id = state["in_cell_id"]
		if state["posterior"]:
			return rf"$\log_{{10}}\,P^t_{{i\rightarrow j}}(i={in_id}\mid j)$"
		return rf"$\log_{{10}}\,P^t_{{i\rightarrow j}}(j\mid i={in_id})$"

	vmin_like = float(norm_likelihood.vmin) if norm_likelihood.vmin is not None else -12.0
	vmin_post = float(norm_posterior.vmin) if norm_posterior.vmin is not None else -12.0
	min_like = float(10 ** vmin_like)
	min_post = float(10 ** vmin_post)

	def _apply_update(use_blit: bool = True) -> None:
		"""Internal helper to apply update.
		
		Args:
			use_blit: Input parameter.
		"""
		t_snapshot = int(state["snapshot_idx"])
		t_global = snapshot_indices[t_snapshot]
		if state["model"]:
			if state["posterior"] and stationary is not None:
				values = _compute_model_values_posterior(int(state["in_cell_id"]))[t_snapshot, :]
				values = np.maximum(values, min_post)
			else:
				values = _compute_model_values(int(state["in_cell_id"]))[t_snapshot, :]
				values = np.maximum(values, min_like)
		else:
			if state["posterior"] and stationary is not None:
				with np.errstate(divide="ignore", invalid="ignore"):
					values = propagator_plot[t_global, state["in_cell_id"], :] * stationary[state["in_cell_id"]] / stationary
			else:
				values = propagator_plot[t_global, state["in_cell_id"], :]
		update_cell_scalar_field(
			ax=ax,
			poly_collection=poly_collection,
			values=values,
			norm=_active_norm(),
			cmap=cmap_obj,
			blit_state=blit_state,
			use_blit=use_blit
		)
		if not use_blit:
			sm.set_norm(_active_norm())
			cbar.update_normal(sm)
			cbar.set_label(_colorbar_label())

	posterior_check = None
	t_slider: Slider | None = None
	if use_widgets:
		assert ax_slider is not None
		t_slider = Slider(ax_slider, r"$t$", valmin=0, valmax=len(snapshot_indices) - 1, valinit=t_index_init, valstep=1)
		t_slider.valtext.set_text(f"{snapshot_times[t_index_init]:.3g}")

		if ax_check is not None:
			ax_check.set_title("options", pad=-10)
			ax_check.set_xticks([])
			ax_check.set_yticks([])
			posterior_check = CheckButtons(ax_check, ["posterior", "model"], [state["posterior"], state["model"]])

		def on_slider(val: float) -> None:
			"""Handle the slider event.
			
			Args:
				val: Input parameter.
			"""
			state["snapshot_idx"] = int(np.clip(int(val), 0, len(snapshot_indices) - 1))
			t_slider.valtext.set_text(f"{snapshot_times[state['snapshot_idx']]:.3g}")
			_apply_update(use_blit=True)

		def on_posterior(_label: str | None) -> None:
			"""Handle the posterior event.
			
			Args:
				_label: Input parameter.
			"""
			if state["lock"] or posterior_check is None:
				return
			prev_posterior = state["posterior"]
			new_posterior = bool(posterior_check.get_status()[0])
			new_model = bool(posterior_check.get_status()[1])
			if new_posterior and stationary is None:
				state["lock"] = True
				posterior_check.set_active(0)
				state["lock"] = False
				new_posterior = False
			if new_model and not model_available:
				_ensure_model_available()
				if not model_available:
					state["lock"] = True
					posterior_check.set_active(1)
					state["lock"] = False
					new_model = False
			state["posterior"] = new_posterior
			state["model"] = new_model
			_apply_update(use_blit=prev_posterior == new_posterior)

		t_slider.on_changed(on_slider)
		if posterior_check is not None:
			posterior_check.on_clicked(on_posterior)

		widget_refs = getattr(fig_any, "_diffsim_widget_refs", [])
		widget_refs.extend([t_slider, on_slider])
		if posterior_check is not None:
			widget_refs.extend([posterior_check, on_posterior])
		setattr(fig_any, "_diffsim_widget_refs", widget_refs)
		if verbose:
			print("Interactive slider and checkbox initialized.")
	def on_click(event) -> None:
		"""Handle the click event.
		
		Args:
			event: Input parameter.
		"""
		if event.inaxes != ax or event.xdata is None or event.ydata is None:
			return
		xy = jnp.array([[float(event.xdata), float(event.ydata)]], dtype=jnp.float32)
		cell_id = int(np.asarray(simulator.field.discretize(xy))[0])
		if cell_id == state["in_cell_id"]:
			return
		state["in_cell_id"] = cell_id
		_apply_update(use_blit=False)

	canvas = cast(Any, fig_any.canvas)
	if canvas is not None:
		canvas.mpl_connect("button_press_event", on_click)

	sm = plt.cm.ScalarMappable(norm=_active_norm(), cmap=cmap_obj)
	sm.set_array([])
	cbar = fig_any.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(_colorbar_label())

	_apply_update(use_blit=False)
	if canvas is not None and canvas.supports_blit and hasattr(canvas, "copy_from_bbox"):
		canvas.draw()
		blit_state["background"] = canvas.copy_from_bbox(ax.bbox)

	_ = n_subset_particles, show_data
	if verbose:
		print("Propagator plot ready.")
	return fig, ax, t_slider, posterior_check


def plot_simulation_mfpt(
	simulator: "Simulator",
	in_cell_id: Int,
	ax: Axes | None = None,
	ax_check: Axes | None = None,
	show_cell_ids: bool = True
) -> tuple[object, Axes]:
	"""Plot MFPT map from one in-cell using saved zarr mfpts data."""
	if simulator.field.tesselation is None:
		raise ValueError("Tesselation must be defined in the field to plot MFPT.")
	if simulator.field.dim != 2:
		raise NotImplementedError("plot_mfpt is only implemented for 2D fields.")
	if "mfpts" not in simulator.zarr_root.array_keys():
		raise ValueError("No mfpts array found in the zarr store.")

	mfpt_array = simulator.zarr_root["mfpts"]
	assert isinstance(mfpt_array, zarr.Array)
	mfpt_mean = np.asarray(mfpt_array[:, :, 0], dtype=float)
	if mfpt_mean.ndim != 2:
		raise ValueError(f"Expected mfpts[...,0] to be 2D, got shape {mfpt_mean.shape}.")

	num_cells = int(simulator.field.num_vert)
	if mfpt_mean.shape != (num_cells, num_cells):
		raise ValueError(
			f"mfpts shape {mfpt_mean.shape} is incompatible with field cell count {num_cells}."
		)

	in_cell_id = int(in_cell_id)
	if not (0 <= in_cell_id < num_cells):
		raise ValueError(f"in_cell_id must be in [0, {num_cells}).")

	use_widgets = ax is None and ax_check is None
	if use_widgets:
		fig = plt.figure(figsize=(10, 10), constrained_layout=True)
		gs = GridSpec(1, 2, width_ratios=[12, 2], figure=fig)
		ax = fig.add_subplot(gs[0, 0])
		ax_check = fig.add_subplot(gs[0, 1])
	else:
		if ax is None:
			raise ValueError("ax is required when custom ax_check is provided.")
		fig = ax.get_figure()

	assert ax is not None
	fig_any = cast(Any, fig)
	ax.set_xlim(float(-simulator.field.size[0] / 2), float(simulator.field.size[0] / 2))
	ax.set_ylim(float(-simulator.field.size[1] / 2), float(simulator.field.size[1] / 2))
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")

	setattr(ax, "_diffsim_field", simulator.field)

	with np.errstate(divide="ignore", invalid="ignore"):
		all_mapped = mfpt_mean
	finite_mask = np.isfinite(all_mapped)
	if np.any(finite_mask):
		vmin = float(np.min(all_mapped[finite_mask]))
		vmax = float(np.max(all_mapped[finite_mask]))
	else:
		vmin, vmax = 0.0, 1.0
	if vmin == vmax:
		vmin -= 0.5
		vmax += 0.5
	norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
	cmap_name = "viridis"
	cmap_obj = plt.get_cmap(cmap_name)

	poly_collection = display_cell_scalar_field(
		ax,
		values=mfpt_mean[in_cell_id, :],
		show_cell_ids=show_cell_ids,
		cmap=cmap_name
	)

	blit_state: dict[str, object] = {"background": None}
	state = {"in_cell_id": in_cell_id}

	def _apply_update(use_blit: bool = True) -> None:
		"""Internal helper to apply update.
		
		Args:
			use_blit: Input parameter.
		"""
		values = mfpt_mean[state["in_cell_id"], :]
		update_cell_scalar_field(
			ax=ax,
			poly_collection=poly_collection,
			values=values,
			norm=norm,
			cmap=cmap_obj,
			blit_state=blit_state,
			use_blit=use_blit,
			scale="linear"
		)

	status_text = None
	if ax_check is not None:
		ax_check.axis("off")
		status_text = ax_check.text(0.0, 0.5, f"in-cell: {state['in_cell_id']}\nclick in plot to change", va="center", ha="left")

	def on_click(event) -> None:
		"""Handle the click event.
		
		Args:
			event: Input parameter.
		"""
		if event.inaxes != ax or event.xdata is None or event.ydata is None:
			return
		xy = jnp.array([[float(event.xdata), float(event.ydata)]], dtype=jnp.float32)
		new_in_cell_id = int(np.asarray(simulator.field.discretize(xy))[0])
		if new_in_cell_id == state["in_cell_id"]:
			return
		state["in_cell_id"] = new_in_cell_id
		if status_text is not None:
			status_text.set_text(f"in-cell: {state['in_cell_id']}\nclick in plot to change")
		_apply_update(use_blit=False)

	canvas = cast(Any, fig_any.canvas)
	if canvas is not None:
		canvas.mpl_connect("button_press_event", on_click)

	sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
	sm.set_array([])
	cbar = fig_any.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(r"$\mathrm{MFPT}$")

	_apply_update(use_blit=False)
	if canvas is not None and canvas.supports_blit and hasattr(canvas, "copy_from_bbox"):
		canvas.draw()
		blit_state["background"] = canvas.copy_from_bbox(ax.bbox)

	return fig, ax


def plot_simulation_disc_stationary_distribution(
	simulator: "Simulator",
	ax: Axes | None = None,
	ax_radio: Axes | None = None,
	show_cell_ids: bool = True,
	res: float = 50.0,
	source_init: str = "data"
) -> tuple[object, Axes, Axes | None]:
	"""Plot discrete stationary distribution with source toggle (data vs Boltzmann)."""
	if simulator.field.tesselation is None:
		raise ValueError("Tesselation must be defined in the field to plot discrete stationary distribution.")
	if simulator.field.dim != 2:
		raise NotImplementedError("plot_disc_stationary_distribution is only implemented for 2D fields.")

	num_cells = int(simulator.field.num_vert)

	if "stationary" not in simulator.zarr_root.array_keys():
		raise ValueError("No stationary array found in zarr store. Run simulation with save_statistics=True first.")

	stationary_array = simulator.zarr_root["stationary"]
	assert isinstance(stationary_array, zarr.Array)
	p_data = np.asarray(stationary_array[:, 0], dtype=float)
	if p_data.shape[0] != num_cells:
		raise ValueError(f"stationary array length {p_data.shape[0]} does not match number of cells {num_cells}.")

	p_boltzmann = np.asarray(simulator.field.get_discrete_stationary_distribution(res=float(res)), dtype=float)
	if p_boltzmann.shape[0] != num_cells:
		raise ValueError(f"Boltzmann stationary length {p_boltzmann.shape[0]} does not match number of cells {num_cells}.")

	ax_cbar: Axes | None = None
	use_widgets = ax is None and ax_radio is None
	if use_widgets:
		fig = plt.figure(figsize=(12, 10), constrained_layout=True)
		gs = GridSpec(
			2,
			2,
			width_ratios=[11, 1],
			height_ratios=[9, 1],
			wspace=0.08,
			hspace=0.08,
			figure=fig,
		)
		ax = fig.add_subplot(gs[:, 0])
		ax_cbar = fig.add_subplot(gs[0, 1])
		ax_radio = fig.add_subplot(gs[1, 1])
	else:
		if ax is None:
			raise ValueError("ax is required when custom ax_radio is provided.")
		fig = ax.get_figure()

	assert ax is not None
	fig_any = cast(Any, fig)
	ax.set_xlim(float(-simulator.field.size[0] / 2), float(simulator.field.size[0] / 2))
	ax.set_ylim(float(-simulator.field.size[1] / 2), float(simulator.field.size[1] / 2))
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")

	setattr(ax, "_diffsim_field", simulator.field)

	def _norm_from_values(values: np.ndarray) -> mcolors.Normalize:
		"""Internal helper to norm from values.
		
		Args:
			values: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		finite = np.isfinite(values)
		if np.any(finite):
			vmin = float(np.min(values[finite]))
			vmax = float(np.max(values[finite]))
		else:
			vmin, vmax = 0.0, 1.0
		if vmin == vmax:
			vmin -= 0.5
			vmax += 0.5
		return mcolors.Normalize(vmin=vmin, vmax=vmax)

	norm_data = _norm_from_values(p_data)
	norm_boltzmann = _norm_from_values(p_boltzmann)
	cmap_name = "viridis"
	cmap_obj = plt.get_cmap(cmap_name)

	source_init = source_init.lower()
	if source_init not in {"data", "boltzmann"}:
		source_init = "data"
	state = {"source": source_init}

	def _current_values() -> np.ndarray:
		"""Internal helper to current values.
		
		Returns:
		Output value computed by this function.
		"""
		return p_boltzmann if state["source"] == "boltzmann" else p_data

	def _current_norm() -> mcolors.Normalize:
		"""Internal helper to current norm.
		
		Returns:
		Output value computed by this function.
		"""
		return norm_boltzmann if state["source"] == "boltzmann" else norm_data

	poly_collection = display_cell_scalar_field(
		ax,
		values=_current_values(),
		show_cell_ids=show_cell_ids,
		cmap=cmap_name
	)

	sm = plt.cm.ScalarMappable(norm=_current_norm(), cmap=cmap_obj)
	sm.set_array([])
	if ax_cbar is not None:
		cbar = fig_any.colorbar(sm, cax=ax_cbar)
	else:
		cbar = fig_any.colorbar(sm, ax=ax, fraction=0.015, pad=0.04)
	cbar.set_label(r"$\pi_i$" + f" ({state['source']})")

	def _apply_update() -> None:
		"""Internal helper to apply update.
		"""
		update_cell_scalar_field(
			ax=ax,
			poly_collection=poly_collection,
			values=_current_values(),
			norm=_current_norm(),
			cmap=cmap_obj,
			use_blit=False,
			scale="linear"
		)
		sm.set_norm(_current_norm())
		cbar.update_normal(sm)
		cbar.set_label(r"$\pi_i$" + f" ({state['source']})")
		canvas = cast(Any, fig_any.canvas)
		if canvas is not None:
			canvas.draw_idle()

	if ax_radio is not None:
		ax_radio.set_xticks([])
		ax_radio.set_yticks([])
		for spine in ax_radio.spines.values():
			spine.set_visible(False)
		radio = RadioButtons(ax_radio, ["data", "boltzmann"], active=0 if state["source"] == "data" else 1)
		ax_radio.set_title("source", fontsize=9, pad=1)
		for txt in radio.labels:
			txt.set_fontsize(9)

		def on_radio(label: str | None) -> None:
			"""Handle the radio event.
			
			Args:
				label: Input parameter.
			"""
			if label is None:
				return
			state["source"] = label
			_apply_update()

		cid = radio.on_clicked(on_radio)
		widget_refs = getattr(fig_any, "_diffsim_widget_refs", [])
		widget_refs.extend([radio, on_radio, cid])
		setattr(fig_any, "_diffsim_widget_refs", widget_refs)

	_apply_update()
	return fig, ax, ax_radio


def plot_simulation_final_magnetization(
	simulator: "Simulator",
	ax: Axes | None = None,
	cell_averages: bool = False,
	show_cell_ids: bool = False,
	marker_size: float = 8.0,
	alpha: float = 0.6,
	field_args: dict | None = None,
) -> tuple[object, Axes]:
	"""Show final magnetization on BW potential as scatter or cell-averaged map."""
	if simulator.field.dim != 2:
		raise NotImplementedError("show_final_magnetization is only implemented for 2D fields.")

	if "traj_final" in simulator.zarr_root.array_keys():
		traj_final = simulator.zarr_root["traj_final"]
		assert isinstance(traj_final, zarr.Array)
		final_data = np.asarray(traj_final, dtype=float)
	elif "traj_cont" in simulator.zarr_root.array_keys():
		traj_cont = simulator.zarr_root["traj_cont"]
		assert isinstance(traj_cont, zarr.Array)
		final_data = np.asarray(traj_cont[:, -1, :], dtype=float)
	else:
		raise ValueError("No final-state data found. Run simulation with save_final=True or save_continuous=True.")

	if final_data.ndim != 2 or final_data.shape[1] < 3:
		raise ValueError("Final-state data does not contain spin phase. Run with a Bfield-enabled simulation that stores phase.")

	positions = final_data[:, :2]
	phi = final_data[:, 2]
	magnetization = np.cos(phi)

	if ax is None:
		fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
	else:
		fig = ax.get_figure()

	kwargs = {} if field_args is None else dict(field_args)
	kwargs.setdefault("show_cells", False)
	kwargs.setdefault("bw", True)
	plot_field(simulator.field, ax_x=ax, plot_dim=2, **kwargs)
	setattr(ax, "_diffsim_field", simulator.field)

	if simulator.field.size[0] > 0 and simulator.field.size[1] > 0:
		ax.set_xlim(float(-simulator.field.size[0] / 2), float(simulator.field.size[0] / 2))
		ax.set_ylim(float(-simulator.field.size[1] / 2), float(simulator.field.size[1] / 2))
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")

	norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
	cmap_obj = plt.get_cmap("coolwarm")

	if cell_averages:
		if simulator.field.tesselation is None:
			raise ValueError("Field tesselation is required when cell_averages=True.")
		cell_ids = np.asarray(simulator.field.discretize(jnp.asarray(positions)), dtype=int)
		num_cells = int(simulator.field.num_vert)
		if cell_ids.shape[0] != magnetization.shape[0]:
			raise ValueError("Discretized indices length does not match number of particles.")

		counts = np.bincount(cell_ids, minlength=num_cells)
		sums = np.bincount(cell_ids, weights=magnetization, minlength=num_cells)
		cell_values = np.divide(
			sums,
			counts,
			out=np.full(num_cells, np.nan, dtype=float),
			where=counts > 0,
		)

		poly_collection = display_cell_scalar_field(
			ax,
			values=cell_values,
			show_cell_ids=show_cell_ids,
			cmap="coolwarm",
		)
		update_cell_scalar_field(
			ax,
			poly_collection=poly_collection,
			values=cell_values,
			norm=norm,
			cmap=cmap_obj,
			use_blit=False,
			scale="linear",
		)
		sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
		sm.set_array([])
		sc = sm
	else:
		sc = ax.scatter(
			positions[:, 0],
			positions[:, 1],
			c=magnetization,
			cmap="coolwarm",
			norm=norm,
			s=marker_size,
			alpha=alpha,
			linewidths=0.0,
		)
	fig_any = cast(Any, fig)
	cbar = fig_any.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(r"$m=\cos(\phi)$")

	return fig, ax


def plot_simulation_model_mri_response(
	simulator: "Simulator",
	times: np.ndarray,
	magnetization: np.ndarray,
	bfield_values: np.ndarray,
	bfield_y: np.ndarray | None = None,
	ax: Axes | None = None,
	ax_bfield: Axes | None = None,
	ax_slider: Axes | None = None,
	show_cell_ids: bool = False,
	show_complex: bool = True,
	show_field: bool = False,
	show_slider: bool = True,
	t_index_init: int | None = None,
	video_dir: str | None = None,
) -> tuple[object, Axes, Axes | None, Axes | None, Slider | None]:
	"""Visualize model MRI response with cell phase (HSV hue) and magnitude (alpha)."""
	if simulator.field.dim != 2:
		raise NotImplementedError("model_mri_response plotting is only implemented for 2D fields.")
	if simulator.field.tesselation is None:
		raise ValueError("Field tesselation is required for cell visualization.")

	times_arr = np.asarray(times, dtype=float)
	m_arr = np.asarray(magnetization, dtype=np.complex64)
	b_arr = np.asarray(bfield_values, dtype=float)
	n_times, n_cells = m_arr.shape
	if times_arr.shape[0] != n_times:
		raise ValueError("times and magnetization length mismatch.")
	if b_arr.shape[0] != n_times:
		raise ValueError("bfield_values and magnetization length mismatch.")

	if t_index_init is None:
		t_index_init = n_times - 1
	assert t_index_init is not None
	t_index_init = int(np.clip(int(t_index_init), 0, n_times - 1))

	if not show_slider:
		ax_slider = None

	use_widgets = ax is None and ax_bfield is None and ax_slider is None
	if use_widgets:
		if show_slider:
			fig = plt.figure(figsize=(12, 9), constrained_layout=True)
			gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[20, 1], figure=fig)
			ax_bfield = fig.add_subplot(gs[0, 0])
			ax = fig.add_subplot(gs[0, 1], sharey=ax_bfield)
			ax_slider = fig.add_subplot(gs[1, :])
		else:
			fig = plt.figure(figsize=(12, 8), constrained_layout=True)
			gs = GridSpec(1, 2, width_ratios=[1, 3], figure=fig)
			ax_bfield = fig.add_subplot(gs[0, 0])
			ax = fig.add_subplot(gs[0, 1], sharey=ax_bfield)
	else:
		if ax is None:
			raise ValueError("ax is required when providing custom axes.")
		fig = ax.get_figure()

	assert ax is not None
	ax.set_xlim(float(-simulator.field.size[0] / 2), float(simulator.field.size[0] / 2))
	ax.set_ylim(float(-simulator.field.size[1] / 2), float(simulator.field.size[1] / 2))
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel(r"$x$")
	ax.set_ylabel("")
	ax.tick_params(axis="y", which="both", left=False, labelleft=False)

	if show_field:
		plot_field(simulator.field, ax_x=ax, plot_dim=2, show_cells=False, bw=True)
	setattr(ax, "_diffsim_field", simulator.field)

	poly_collection = display_cell_scalar_field(
		ax,
		values=np.zeros((n_cells,), dtype=float),
		show_cell_ids=show_cell_ids,
		cmap="hsv",
	)
	base_cell_ids = np.asarray(getattr(poly_collection, "_diffsim_base_cell_ids"), dtype=int)
	hsv_cmap = plt.get_cmap("hsv")
	real_cmap = plt.get_cmap("coolwarm")

	if ax_bfield is not None:
		if bfield_y is None:
			ys_plot = np.asarray(simulator.field.tesselation.centers[:, 1], dtype=float)
			y_order = np.argsort(ys_plot)
			b_init = b_arr[t_index_init, y_order]
			ys_draw = ys_plot[y_order]
		else:
			ys_plot = np.asarray(bfield_y, dtype=float)
			y_order = None
			b_init = b_arr[t_index_init]
			ys_draw = ys_plot

		line, = ax_bfield.plot(b_init, ys_draw, lw=1.0)
		b_min = float(np.min(b_arr))
		b_max = float(np.max(b_arr))
		if b_min == b_max:
			delta = max(1e-8, abs(b_min) * 0.1 + 1e-8)
			b_min, b_max = b_min - delta, b_max + delta
		pad = 0.05 * (b_max - b_min)
		ax_bfield.axvline(0.0, color="0.7", lw=0.8, zorder=0)
		ax_bfield.set_xlim(b_min - pad, b_max + pad)
		ax_bfield.set_xlabel(r"$B$")
		ax_bfield.set_ylabel("")
		ax_bfield.set_box_aspect(3)
	else:
		line = None
		ys_plot = None
		y_order = None

	if show_complex:
		norm = mcolors.Normalize(vmin=0.0, vmax=2 * np.pi)
		sm = plt.cm.ScalarMappable(norm=norm, cmap=hsv_cmap)
		sm.set_array([])
		fig_any = cast(Any, fig)
		cbar = fig_any.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label(r"$\phi$")
		cbar.set_ticks([0.0, np.pi, 2 * np.pi])
		cbar.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
	else:
		real_vals_all = np.real(m_arr)
		vmax = float(np.max(np.abs(real_vals_all))) if real_vals_all.size > 0 else 1.0
		if vmax <= 0:
			vmax = 1.0
		norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
		sm = plt.cm.ScalarMappable(norm=norm, cmap=real_cmap)
		sm.set_array([])
		fig_any = cast(Any, fig)
		cbar = fig_any.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label(r"$m=\cos(\phi)$")

	def _set_cell_colors(idx: int) -> None:
		"""Internal helper to set cell colors.
		
		Args:
			idx: Input parameter.
		"""
		vals = m_arr[idx]
		if show_complex:
			phase = np.mod(np.angle(vals), 2.0 * np.pi)
			magnitude = np.clip(np.abs(vals), 0.0, 1.0)
			colors = np.asarray(hsv_cmap(phase[base_cell_ids] / (2.0 * np.pi)))
			colors[:, 3] = magnitude[base_cell_ids]
		else:
			real_vals = np.real(vals)
			colors = np.asarray(real_cmap(norm(real_vals[base_cell_ids])))
			colors[:, 3] = 1.0
		poly_collection.set_facecolors(colors)  # type: ignore[attr-defined]

		if line is not None and ys_plot is not None:
			if y_order is None:
				line.set_data(b_arr[idx], ys_plot)
			else:
				line.set_data(b_arr[idx, y_order], ys_plot[y_order])

		if ax_slider is not None:
			ax_slider.set_title(rf"$t={times_arr[idx]:.3g}$", pad=-10)

		canvas = ax.figure.canvas
		if canvas is not None:
			canvas.draw_idle()

	t_slider = None
	if ax_slider is not None:
		t_slider = Slider(
			ax_slider,
			r"$t$",
			valmin=float(times_arr[0]),
			valmax=float(times_arr[-1]),
			valinit=float(times_arr[t_index_init]),
			valstep=times_arr,
			valfmt="%.3g",
		)

		def _on_slider(val: float) -> None:
			"""Handle the slider event.
			
			Args:
				val: Input parameter.
			"""
			jj = int(np.argmin(np.abs(times_arr - float(val))))
			_set_cell_colors(jj)

		t_slider.on_changed(_on_slider)
		ax_slider.set_title(rf"$t={times_arr[t_index_init]:.3g}$", pad=-10)
		fig_any = cast(Any, fig)
		widget_refs = getattr(fig_any, "_diffsim_widget_refs", [])
		widget_refs.extend([t_slider, _on_slider])
		setattr(fig_any, "_diffsim_widget_refs", widget_refs)

	_set_cell_colors(t_index_init)

	video_path: str | None = None
	if video_dir is not None:
		os.makedirs(video_dir, exist_ok=True)
		file_stem = f"{simulator.name}_model_mri_response"
		video_mp4 = os.path.join(video_dir, f"{file_stem}.mp4")

		dynamic_artists: list[Any] = [poly_collection]
		if line is not None:
			dynamic_artists.append(line)

		def _animate(frame_idx: int):
			"""Internal helper to animate.
			
			Args:
				frame_idx: Input parameter.
			
			Returns:
			Output value computed by this function.
			"""
			_set_cell_colors(int(frame_idx))
			return tuple(dynamic_artists)

		fig_anim = ax.figure
		if not isinstance(fig_anim, Figure):
			raise ValueError("Video export requires a matplotlib Figure.")
		animation = FuncAnimation(fig_anim, _animate, frames=n_times, interval=80, blit=False, repeat=False)
		ffmpeg_exe = shutil.which("ffmpeg")
		if ffmpeg_exe is None:
			try:
				import imageio_ffmpeg
				ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
			except Exception:
				ffmpeg_exe = None
		if ffmpeg_exe is None:
			raise RuntimeError("Failed to save MP4 video. Install system ffmpeg or run `pip install imageio-ffmpeg` in the active environment.")
		prev_ffmpeg_path = mpl.rcParams.get("animation.ffmpeg_path", "ffmpeg")
		mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
		try:
			animation.save(
				video_mp4,
				writer=FFMpegWriter(fps=8, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]),
				dpi=110,
			)
		except Exception as exc:
			raise RuntimeError("Failed to save MP4 video even after locating ffmpeg executable.") from exc
		finally:
			mpl.rcParams["animation.ffmpeg_path"] = prev_ffmpeg_path
		video_path = video_mp4

	fig_any = cast(Any, fig)
	setattr(fig_any, "_diffsim_video_path", video_path)
	return fig, ax, ax_bfield, ax_slider, t_slider

def plot_cell_residence_times(
	simulator: "Simulator",
	cell_id: Int = 0,
	padding: float = 0.3,
	cmap: str = "tab10",
	transition_rates: bool = True,
	t_max_multiplier: Float | None = None,
	verbose: bool = True,
	yscale: str = "log",
	ax_cell: Axes | None = None,
	ax_all: Axes | None = None,
	ax_slider: Axes | None = None,
	ax_radio_in: Axes | None = None,
	ax_radio_metric: Axes | None = None,
	ax_check: Axes | None = None,
	ax_check_toggle: Axes | None = None,
	in_selection_init: str = "all",
	out_selection_init: Collection[int] | None = None,
	metric_init: str | None = None,
	t_max_init: float | None = None,
	toggle_data: bool = True,
	toggle_model: bool = True,
	toggle_graph: bool = True
) -> tuple[object, Axes, Axes, object | None, object | None, object | None, object | None]:
	"""Plot residence-time diagnostics for one cell.

	Returns the figure, main axes, and widget handles (None when widgets are disabled).
	"""
	assert simulator.field.dim == 2, "Only implemented for 2D fields."
	assert "transitions" in simulator.zarr_root.array_keys(), "No transition data found in the zarr store."
	assert simulator.field.tesselation is not None, "Tesselation must be defined in the field to plot cell residence times."

	cell_id = int(cell_id)
	padding = float(padding)

	transition_array = simulator.zarr_root["transitions"]
	assert isinstance(transition_array, zarr.Array)

	neighbors = np.asarray(simulator.field.tesselation.neighbors[cell_id], dtype=int)
	num_cells = simulator.field.num_vert
	assert 0 <= cell_id < int(num_cells), f"cell_id must be in [0, {int(num_cells)})"
	valid_mask = neighbors < num_cells
	k_labels = np.where(valid_mask, neighbors, -1)
	k_labels = np.concatenate([k_labels, np.array([-1], dtype=k_labels.dtype)])

	counts = np.asarray(transition_array[cell_id])
	valid_idx = np.where(k_labels >= 0)[0]
	if valid_idx.size == 0:
		if verbose:
			print(f"No neighbor transitions found for cell {cell_id}.")
		return None

	counts_valid = counts[np.ix_(valid_idx, valid_idx)]
	counts_by_out_t = counts_valid.sum(axis=0)
	total_counts = counts_by_out_t.sum()
	if total_counts == 0:
		if verbose:
			print(f"No transitions found for cell {cell_id}.")
		return None

	counts_in_t = counts_valid.sum(axis=1)
	total_in = counts_in_t.sum(axis=1)
	lost_in = np.cumsum(counts_in_t, axis=1)
	remaining_in = total_in[:, None] - np.concatenate([np.zeros((len(valid_idx), 1)), lost_in[:, :-1]], axis=1)
	remaining_in = np.maximum(remaining_in, 1.0)
	remaining_next_in = np.maximum(np.concatenate([remaining_in[:, 1:], np.ones((len(valid_idx), 1))], axis=1), 1.0)
	decay_rate_in = np.log(remaining_in / remaining_next_in) / simulator.dt
	decay_safe_in = np.where(decay_rate_in == 0, np.nan if yscale == "log" else 1e-12, decay_rate_in)
	remaining_avg_in = remaining_in * (1 - np.exp(-decay_safe_in * simulator.dt)) / (decay_safe_in * simulator.dt)

	total_marginal = counts_in_t.sum(axis=0)
	lost_marginal = np.cumsum(total_marginal)
	remaining_marginal = total_marginal.sum() - np.concatenate(([0.0], lost_marginal[:-1]))
	remaining_marginal = np.maximum(remaining_marginal, 1.0)
	remaining_next_marginal = np.maximum(np.concatenate((remaining_marginal[1:], [1.0])), 1.0)
	decay_rate_marginal = np.log(remaining_marginal / remaining_next_marginal) / simulator.dt
	decay_safe_marginal = np.where(decay_rate_marginal == 0, np.nan if yscale == "log" else 1e-12, decay_rate_marginal)
	remaining_avg_marginal = remaining_marginal * (1 - np.exp(-decay_safe_marginal * simulator.dt)) / (decay_safe_marginal * simulator.dt)

	t_int_max_cap = max(1, min(counts_by_out_t.shape[1], transition_array.shape[3] // 3))
	t_idx = np.arange(1, counts_by_out_t.shape[1] + 1)
	mean_lifetime = float((counts_by_out_t * t_idx[None, :]).sum() / total_counts) * simulator.dt
	if t_max_multiplier is None:
		t_int_max = t_int_max_cap
	else:
		t_int_max = max(1, int(min(t_int_max_cap, t_max_multiplier * mean_lifetime / simulator.dt)))

	counts_valid_full = counts_valid
	counts_by_out_t_full = counts_by_out_t
	remaining_avg_in_full = remaining_avg_in
	remaining_avg_marginal_full = remaining_avg_marginal

	counts_valid = counts_valid_full
	counts_by_out_t = counts_by_out_t_full
	remaining_avg_in = remaining_avg_in_full
	remaining_avg_marginal = remaining_avg_marginal_full

	k_sums = counts_by_out_t.sum(axis=1)
	valid_neighbor_ids = np.asarray(k_labels[valid_idx], dtype=int)
	close_neighbors = valid_neighbor_ids[k_sums > 0]

	center = simulator.field.tesselation.centers[cell_id]
	plot_ids = np.array([cell_id] + [n for n in close_neighbors if n != cell_id], dtype=int)
	shift_vectors = jnp.array([-jnp.round((simulator.field.tesselation.centers[i] - center) / simulator.field.size) * simulator.field.size for i in plot_ids])
	centers = simulator.field.tesselation.centers[plot_ids] + shift_vectors
	bounds = jnp.array([jnp.min(centers, axis=0) - padding * simulator.field.lambda0, jnp.max(centers, axis=0) + padding * simulator.field.lambda0]).T

	use_widgets = all(ax is None for ax in [ax_cell, ax_all, ax_slider, ax_radio_in, ax_radio_metric, ax_check, ax_check_toggle])
	if use_widgets:
		fig = plt.figure(figsize=(10, 10), constrained_layout=True)
		gs = GridSpec(3, 3, height_ratios=[6, 4, 0.35], width_ratios=[6.4, 0.6, 1.0], figure=fig)
		ax_cell = fig.add_subplot(gs[0, 0])
		radio_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[4, 1])
		ax_radio_in = fig.add_subplot(radio_gs[0, 0])
		ax_radio_metric = fig.add_subplot(radio_gs[1, 0])
		check_gs = gs[0, 2].subgridspec(2, 1, height_ratios=[4, 1])
		ax_check = fig.add_subplot(check_gs[0, 0])
		ax_check_toggle = fig.add_subplot(check_gs[1, 0])
		ax_all = fig.add_subplot(gs[1, :])
		ax_slider = fig.add_subplot(gs[2, :])
	else:
		assert ax_cell is not None or ax_all is not None, "At least ax_cell or ax_all must be provided."
		fig = ax_cell.get_figure() if ax_cell is not None else ax_all.get_figure()

	if ax_cell is not None:
		plot_field(simulator.field, ax_x=ax_cell, plot_dim=2, show_cells=False, bounds=bounds, pad=0.0)
		ax_cell.set_xlim(*ax_cell.get_xlim())
		ax_cell.set_ylim(*ax_cell.get_ylim())
		ax_cell.set_aspect("equal", adjustable="box")

	from helpers.cell_subgraph_inference import (
		fit_model_one_center,
		p_model_one_center,
		transition_rate_model_one_center,
		guesstimate_rates_model_one_center
	)

	neighbor_colors = plt.get_cmap(cmap)(jnp.linspace(0, 1, len(valid_neighbor_ids)))
	color_map = {cid: neighbor_colors[i] for i, cid in enumerate(valid_neighbor_ids)}
	colors = np.array([color_map.get(cid, neighbor_colors[0]) for cid in plot_ids])
	face_alphas = jnp.where(plot_ids == cell_id, 0.0, 0.45)
	edge_alphas = jnp.where(plot_ids == cell_id, 0.0, 0.9)
	assert isinstance(colors, np.ndarray)
	poly_collection = None
	if ax_cell is not None:
		plot_artists = plot_tesselation(
			simulator.field.tesselation,
			cell_ids=jnp.array(plot_ids),
			ax=ax_cell,
			shift_vectors=shift_vectors,
			colors=list(colors),
			edge_alpha=edge_alphas,
			face_alpha=face_alphas,
			annotation_size=12
		)
		poly_collection = plot_artists[-1] if len(plot_artists) > 0 else None

	frac_in = np.clip(counts_valid / remaining_avg_in[:, None, :], 0.0, 1.0 - 1e-12)
	rate_in = -np.log(1 - frac_in) / simulator.dt
	rate_in = np.maximum(rate_in, 1e-12)

	frac_marginal = np.clip(counts_by_out_t / remaining_avg_marginal[None, :], 0.0, 1.0 - 1e-12)
	rate_marginal = -np.log(1 - frac_marginal) / simulator.dt
	rate_marginal = np.maximum(rate_marginal, 1e-12)

	total_counts_per_in_vert = counts_valid_full.sum(axis=(1, 2), keepdims=True)
	normalized_counts_valid_full = counts_valid_full / (total_counts_per_in_vert * simulator.dt + 1e-12)
	normalized_counts_by_out_t_full = counts_by_out_t_full / (float(total_counts) * simulator.dt + 1e-12)

	counts_cube_full = np.concatenate([normalized_counts_valid_full, normalized_counts_by_out_t_full[None, :, :]], axis=0)
	rate_cube_full = np.concatenate([rate_in, rate_marginal[None, :, :]], axis=0)
	times_full = np.asarray(jnp.arange(0.5, counts_by_out_t_full.shape[1] + 0.5) * simulator.dt)
	current_t_int_max = int(t_int_max)
	times = times_full

	if metric_init is None:
		metric_init = "rate" if transition_rates else "count"

	selected_out = set(valid_neighbor_ids) if out_selection_init is None else set(int(v) for v in out_selection_init)
	in_selection_init = "all" if in_selection_init not in ["all"] + [str(v) for v in valid_neighbor_ids] else in_selection_init

	scatter_map = {}
	model_line_map = {}
	if ax_all is not None:
		for out_idx, out_cell in enumerate(valid_neighbor_ids):
			c = color_map.get(out_cell, "k")
			y = normalized_counts_by_out_t_full[out_idx]
			scatter = ax_all.scatter(times, y, color=c, label=f"{out_cell}", s=18, alpha=0.8, zorder=3)
			scatter_map[out_cell] = scatter

	default_in_index = len(valid_neighbor_ids)
	if ax_all is not None:
		for out_idx, out_cell in enumerate(valid_neighbor_ids):
			c = color_map.get(out_cell, "k")
			c_dark = _darken_color(c, 0.25)
			y = normalized_counts_by_out_t_full[out_idx]
			line, = ax_all.plot(times_full, y, color=c_dark, linestyle="--", alpha=0.95, linewidth=2.0, zorder=5)
			model_line_map[out_cell] = line

	if ax_all is not None:
		ax_all.set_xlabel(r"$t$", labelpad=-2)
		ax_all.set_xlim(0.0, current_t_int_max * simulator.dt)
		ax_all.set_yscale(yscale)

	legend_data = None
	legend_model = None

	def update_legends() -> None:
		"""Update legends.
		"""
		nonlocal legend_data, legend_model
		if ax_all is None:
			return
		if legend_data is not None:
			legend_data.remove()
		if legend_model is not None:
			legend_model.remove()
		visible_scatter = [s for s in scatter_map.values() if s.get_visible()]
		if visible_scatter:
			labels = [s.get_label() for s in visible_scatter]
			legend_data = ax_all.legend(visible_scatter, labels, title="exiting id", loc="upper right", frameon=True, ncol=2)
			ax_all.add_artist(legend_data)
		visible_model = any(line.get_visible() for line in model_line_map.values())
		if visible_model:
			legend_model = ax_all.legend(
				handles=[Line2D([0], [0], color="gray", linestyle="--")],
				labels=["model"],
				loc="lower right",
				frameon=True
			)

	t_max_allowed = min(float(t_int_max_cap * simulator.dt), 2.0)
	t_max_min = max(20 * simulator.dt, simulator.dt)
	if t_max_allowed < t_max_min:
		t_max_allowed = t_max_min
	t_max_init = min(max(0.2, t_max_min), t_max_allowed) if t_max_init is None else float(t_max_init)
	current_t_int_max = max(1, int(t_max_init // simulator.dt))
	if ax_all is not None:
		ax_all.set_xlim(0.0, current_t_int_max * simulator.dt)

	if use_widgets:
		check_labels = [str(n) for n in valid_neighbor_ids]
		check_states = [int(n) in selected_out for n in valid_neighbor_ids]
		check = CheckButtons(ax_check, check_labels, check_states)
		toggle_labels = ["data", "model", "graph"]
		toggle_check = CheckButtons(ax_check_toggle, toggle_labels, [toggle_data, toggle_model, toggle_graph])

		in_labels = ["all"] + [str(n) for n in valid_neighbor_ids]
		in_active = in_labels.index(in_selection_init) if in_selection_init in in_labels else 0
		radio_in = RadioButtons(ax_radio_in, in_labels, active=in_active)

		metric_labels = ["rate", "count"]
		radio_metric = RadioButtons(ax_radio_metric, metric_labels, active=metric_labels.index(metric_init))

		ax_radio_in.set_title("in-cell", pad=-10)
		ax_radio_metric.set_title("metric", pad=-10)
		ax_check.set_title("out-cell", pad=-10)
		ax_check_toggle.set_title("toggle visible", pad=-10)

		t_max_slider = Slider(ax_slider, r"$t_{\max}$", valmin=t_max_min, valmax=t_max_allowed, valinit=t_max_init)
	else:
		check = toggle_check = radio_in = radio_metric = t_max_slider = None

	def darken(color: object, factor: float = 0.7) -> tuple[float, float, float]:
		"""Darken.
		
		Args:
			color: Input parameter.
			factor: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		rgb = np.array(mcolors.to_rgb(color))
		return tuple((rgb * factor).clip(0.0, 1.0))

	if use_widgets:
		for text, out_cell in zip(check.labels, valid_neighbor_ids):
			text.set_color(darken(color_map.get(out_cell, "k")))
		for text, label in zip(radio_in.labels, ["all"] + [str(n) for n in valid_neighbor_ids]):
			if label == "all":
				text.set_color("k")
			else:
				text.set_color(darken(color_map.get(int(label), "k")))

	edge_lines = {}
	edge_shift = np.asarray(shift_vectors[0])
	if ax_cell is not None:
		for idx, out_cell in zip(valid_idx, valid_neighbor_ids):
			edge_vertices = np.asarray(simulator.field.tesselation.edge_polygon_vertices[cell_id, idx, :2, :]) + edge_shift
			c = color_map.get(out_cell, "k")
			line, = ax_cell.plot(edge_vertices[:, 0], edge_vertices[:, 1], color=c, lw=4, alpha=0.9, zorder=5)
			edge_lines[out_cell] = line

	data_array = counts_valid_full
	r_fitted, fit_success = fit_model_one_center(data_array, dt=simulator.dt, verbose=verbose)
	if not fit_success and verbose:
		print(f"Warning: Model fit did not converge for cell {cell_id}")
	r_init = guesstimate_rates_model_one_center(data_array, dt=simulator.dt, verbose=False)
	if np.allclose(r_fitted, r_init, rtol=1e-5, atol=1e-10) and verbose:
		print(f"Warning: Model fit did not change from initial guess for cell {cell_id}")

	num_neighbors = len(valid_neighbor_ids)
	neighbor_list = [int(n) for n in valid_neighbor_ids]

	in_edge_labels = [f"e({n},{cell_id})" for n in neighbor_list]
	center_label = f"c({cell_id})"
	out_edge_labels = [f"e({cell_id},{n})" for n in neighbor_list]
	vertex_labels_graph = in_edge_labels + [center_label] + out_edge_labels

	n_vertices = 2 * num_neighbors + 1
	adjacency_mat = np.zeros((n_vertices, n_vertices))
	adjacency_mat[:num_neighbors, num_neighbors] = r_fitted[1:, 0]
	adjacency_mat[:num_neighbors, num_neighbors + 1:] = r_fitted[1:, 1:]
	adjacency_mat[num_neighbors, num_neighbors + 1:] = r_fitted[0, 1:]

	neighbor_shifts = {}
	for n_id in neighbor_list:
		neighbor_center = simulator.field.tesselation.centers[n_id]
		shift = -jnp.round((neighbor_center - center) / simulator.field.size) * simulator.field.size
		neighbor_shifts[n_id] = np.asarray(shift)

	pos_map_graph = {}
	for n_id in neighbor_list:
		label = f"e({n_id},{cell_id})"
		pos = _get_vertex_position(label, simulator.field.tesselation, edge_offset_scale=0.3, tangent_offset_scale=0.1)
		pos_map_graph[label] = tuple(np.asarray(pos + neighbor_shifts[n_id], dtype=float))

	pos = _get_vertex_position(center_label, simulator.field.tesselation, edge_offset_scale=0.3, tangent_offset_scale=0.1)
	pos_map_graph[center_label] = tuple(np.asarray(pos + edge_shift, dtype=float))

	for n_id in neighbor_list:
		label = f"e({cell_id},{n_id})"
		pos = _get_vertex_position(label, simulator.field.tesselation, edge_offset_scale=0.3, tangent_offset_scale=0.1)
		pos_map_graph[label] = tuple(np.asarray(pos + edge_shift, dtype=float))

	base_colors_graph = {}
	for i, label in enumerate(in_edge_labels):
		base_colors_graph[label] = color_map.get(neighbor_list[i], "gray")
	base_colors_graph[center_label] = color_map.get(cell_id, "gray")
	for i, label in enumerate(out_edge_labels):
		base_colors_graph[label] = color_map.get(neighbor_list[i], "gray")

	G_sub = _build_labeled_digraph(adjacency_mat, vertex_labels_graph)

	graph_node_artists = {}
	graph_edge_artists = {}
	graph_label_artists = {}

	if ax_cell is not None:
		for label in in_edge_labels:
			pos = pos_map_graph[label]
			color = _lighten_color(base_colors_graph[label], 0.3)
			rect = mpatches.Rectangle((pos[0] - 0.015, pos[1] - 0.015), 0.03, 0.03, facecolor=color, edgecolor='black', linewidth=0.6, zorder=100)
			ax_cell.add_patch(rect)
			graph_node_artists[label] = rect
			text = ax_cell.text(pos[0], pos[1], label.split('(')[1].split(',')[0], fontsize=6, ha='center', va='center', zorder=101)
			graph_label_artists[label] = text

		pos = pos_map_graph[center_label]
		color = _lighten_color(base_colors_graph[center_label], 0.3)
		circle = mpatches.Circle(pos, 0.02, facecolor=color, edgecolor='black', linewidth=0.6, zorder=100)
		ax_cell.add_patch(circle)
		graph_node_artists[center_label] = circle
		text = ax_cell.text(pos[0], pos[1], str(cell_id), fontsize=6, ha='center', va='center', zorder=101)
		graph_label_artists[center_label] = text

		for label in out_edge_labels:
			pos = pos_map_graph[label]
			color = _lighten_color(base_colors_graph[label], 0.3)
			rect = mpatches.Rectangle((pos[0] - 0.015, pos[1] - 0.015), 0.03, 0.03, facecolor=color, edgecolor='black', linewidth=0.6, zorder=100)
			ax_cell.add_patch(rect)
			graph_node_artists[label] = rect
			text = ax_cell.text(pos[0], pos[1], label.split(',')[1].split(')')[0], fontsize=6, ha='center', va='center', zorder=101)
			graph_label_artists[label] = text

		edge_list = list(G_sub.edges(data=True))
		max_weight = max((float(d.get("weight", 0.0)) for _, _, d in edge_list), default=1.0)
		max_sqrt_w = float(np.sqrt(max_weight)) if max_weight > 0 else 1.0
		max_edge_width_val = 0.015
		for u, v, d in edge_list:
			weight = float(d.get("weight", 0.0))
			width = max_edge_width_val * np.sqrt(weight) / max_sqrt_w if max_sqrt_w > 0 else 0.001
			pos_u = pos_map_graph[u]
			pos_v = pos_map_graph[v]
			color = _darken_color(base_colors_graph[v], 0.5)
			arrow = mpatches.FancyArrowPatch(
				pos_u, pos_v,
				arrowstyle='Simple,head_length=0.4,head_width=0.3,tail_width=0.1',
				color=color, linewidth=width * 50, zorder=99, mutation_scale=20
			)
			ax_cell.add_patch(arrow)
			graph_edge_artists[(u, v)] = arrow

	ts_model = jnp.asarray(times_full)
	n_times = int(ts_model.shape[0])
	model_count_full = np.empty((num_neighbors, num_neighbors, n_times), dtype=float)
	model_rate_full = np.empty((num_neighbors, num_neighbors, n_times), dtype=float)
	for i in range(num_neighbors):
		for j in range(num_neighbors):
			model_count_full[i, j] = np.asarray(p_model_one_center(ts_model, i, j, r_fitted))
			model_rate_full[i, j] = np.asarray(transition_rate_model_one_center(ts_model, i, j, r_fitted))
	model_count_all = model_count_full.mean(axis=0, keepdims=True)
	model_rate_all = model_rate_full.mean(axis=0, keepdims=True)
	model_count_cube_full = np.concatenate([model_count_full, model_count_all], axis=0)
	model_rate_cube_full = np.concatenate([model_rate_full, model_rate_all], axis=0)

	if ax_all is not None:
		for out_idx, out_cell in enumerate(valid_neighbor_ids):
			c = color_map.get(out_cell, "k")
			c_dark = _darken_color(c, 0.25)
			y = model_count_cube_full[default_in_index, out_idx]
			model_line_map[out_cell].set_ydata(y)
			model_line_map[out_cell].set_color(c_dark)

	def get_in_index(selection: str) -> int:
		"""Return in index.
		
		Args:
			selection: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		if selection == "all":
			return len(valid_neighbor_ids)
		return int(np.where(valid_neighbor_ids == int(selection))[0][0])

	def update_face_alphas(selected_out: set[int]) -> None:
		"""Update face alphas.
		
		Args:
			selected_out: Input parameter.
		"""
		if poly_collection is None:
			return
		facecolors = poly_collection.get_facecolors()
		if len(facecolors) != len(plot_ids):
			return
		for i, pid in enumerate(plot_ids):
			if int(pid) == cell_id:
				facecolors[i, 3] = 0.0
			elif int(pid) in selected_out:
				facecolors[i, 3] = 0.5
			else:
				facecolors[i, 3] = 0.3
		poly_collection.set_facecolors(facecolors)

	def update_graph_colors(in_selection: str, selected_out: set[int]) -> None:
		"""Update graph colors.
		
		Args:
			in_selection: Input parameter.
			selected_out: Input parameter.
		"""
		if not graph_node_artists:
			return
		in_idx = None if in_selection == "all" else neighbor_list.index(int(in_selection))
		in_edge_active = {}
		out_edge_active = {}
		for i, label in enumerate(in_edge_labels):
			artist = graph_node_artists[label]
			is_active = (in_idx is None or i == in_idx)
			in_edge_active[label] = is_active
			color = _lighten_color(base_colors_graph[label], 0.3) if is_active else _lighten_color("gray", 0.3)
			artist.set_facecolor(color)
		for i, label in enumerate(out_edge_labels):
			artist = graph_node_artists[label]
			neighbor_id = neighbor_list[i]
			is_active = neighbor_id in selected_out
			out_edge_active[label] = is_active
			color = _lighten_color(base_colors_graph[label], 0.3) if is_active else _lighten_color("gray", 0.3)
			artist.set_facecolor(color)
		for (u, v), arrow in graph_edge_artists.items():
			u_active = in_edge_active.get(u, True if u == center_label else False)
			v_active = out_edge_active.get(v, True if v == center_label else False)
			is_relevant = u_active and v_active
			if is_relevant:
				if v == center_label:
					color = _darken_color(base_colors_graph[center_label], 0.2)
				elif v in out_edge_active:
					color = _darken_color(base_colors_graph[v], 0.2)
				else:
					color = _darken_color(base_colors_graph.get(u, "gray"), 0.2)
				arrow.set_color(color)
				arrow.set_alpha(0.95)
			else:
				arrow.set_color("lightgray")
				arrow.set_alpha(0.3)

	def set_graph_visible(visible: bool) -> None:
		"""Set graph visible.
		
		Args:
			visible: Input parameter.
		"""
		for artist in graph_node_artists.values():
			artist.set_visible(visible)
		for artist in graph_edge_artists.values():
			artist.set_visible(visible)
		for artist in graph_label_artists.values():
			artist.set_visible(visible)

	def compute_log_ylim(values: np.ndarray) -> tuple[float, float] | None:
		"""Compute log ylim.
		
		Args:
			values: Input parameter.
		
		Returns:
		Output value computed by this function.
		"""
		finite_pos = values[np.isfinite(values) & (values > 0)]
		filtered = finite_pos[finite_pos > 1.1e-12]
		if filtered.size == 0:
			filtered = finite_pos
		if filtered.size == 0:
			return None
		y_min = float(np.percentile(filtered, 1.0))
		y_max = float(filtered.max())
		if y_max <= y_min:
			y_max = y_min * 1.1
		log_min = np.log10(y_min)
		log_max = np.log10(y_max)
		pad = 0.03 * max(log_max - log_min, 0.2)
		return max(10 ** (log_min - pad), 1e-12), 10 ** (log_max + pad)

	def update_visibility(_=None) -> None:
		"""Update visibility.
		
		Args:
			_: Input parameter.
		"""
		selected_out_local = selected_out if not use_widgets else {valid_neighbor_ids[i] for i, state in enumerate(check.get_status()) if state}
		data_visible = toggle_data if not use_widgets else toggle_check.get_status()[0]
		model_visible = toggle_model if not use_widgets else toggle_check.get_status()[1]
		graph_visible = toggle_graph if not use_widgets else toggle_check.get_status()[2]
		in_selection = in_selection_init if not use_widgets else radio_in.value_selected
		model_visible_actual = model_visible and (in_selection != "all")
		for out_cell, scatter in scatter_map.items():
			scatter.set_visible(data_visible and (out_cell in selected_out_local))
		for out_cell, line in model_line_map.items():
			line.set_visible(model_visible_actual and (out_cell in selected_out_local))
		set_graph_visible(graph_visible)
		update_face_alphas(selected_out_local)
		update_graph_colors(in_selection, selected_out_local)
		update_legends()
		if ax_all is not None:
			ax_all.figure.canvas.draw_idle()

	def update_data(_=None) -> None:
		"""Update data.
		
		Args:
			_: Input parameter.
		"""
		in_selection = in_selection_init if not use_widgets else radio_in.value_selected
		metric = metric_init if not use_widgets else radio_metric.value_selected
		in_index = get_in_index(in_selection)
		data_cube_full = rate_cube_full if metric == "rate" else counts_cube_full
		for out_idx, out_cell in enumerate(valid_neighbor_ids):
			scatter = scatter_map[out_cell]
			y = data_cube_full[in_index, out_idx]
			scatter.set_offsets(np.column_stack((times_full, y)))
		if in_selection == "all":
			for line in edge_lines.values():
				line.set_visible(True)
			for line in model_line_map.values():
				line.set_visible(False)
		else:
			selected_in = int(in_selection)
			for out_cell, line in edge_lines.items():
				line.set_visible(out_cell == selected_in)
			model_cube_full = model_rate_cube_full if metric == "rate" else model_count_cube_full
			for out_idx, out_cell in enumerate(valid_neighbor_ids):
				line = model_line_map[out_cell]
				y = model_cube_full[in_index, out_idx]
				line.set_ydata(y)
		if ax_all is not None:
			ax_all.set_ylabel(r"$\lambda$" if metric == "rate" else r"$p$")
			y_vals = data_cube_full[in_index].ravel()
			y_lim = compute_log_ylim(y_vals)
			if y_lim is not None:
				ax_all.set_ylim(*y_lim)
		update_visibility()

	def update_tmax(val: float) -> None:
		"""Update tmax.
		
		Args:
			val: Input parameter.
		"""
		nonlocal current_t_int_max
		t_int_max_cap_local = max(1, int(t_max_allowed // simulator.dt))
		t_int_max_new = max(1, min(t_int_max_cap_local, int(float(val) // simulator.dt)))
		if t_int_max_new == current_t_int_max:
			return
		current_t_int_max = t_int_max_new
		if ax_all is not None:
			ax_all.set_xlim(0.0, current_t_int_max * simulator.dt)
		in_selection = in_selection_init if not use_widgets else radio_in.value_selected
		metric = metric_init if not use_widgets else radio_metric.value_selected
		in_index = get_in_index(in_selection)
		data_cube_full = rate_cube_full if metric == "rate" else counts_cube_full
		y_vals = data_cube_full[in_index, :, :current_t_int_max].ravel()
		y_lim = compute_log_ylim(y_vals)
		if y_lim is not None and ax_all is not None:
			ax_all.set_ylim(*y_lim)
		if ax_all is not None:
			ax_all.figure.canvas.draw_idle()

	if use_widgets:
		check.on_clicked(update_visibility)
		toggle_check.on_clicked(update_visibility)
		radio_in.on_clicked(update_data)
		radio_metric.on_clicked(update_data)
		t_max_slider.on_changed(update_tmax)

	update_data()
	update_visibility()

	return fig, ax_cell, ax_all, check if use_widgets else None, radio_in if use_widgets else None, radio_metric if use_widgets else None, t_max_slider if use_widgets else None


def plot_effective_diffusion_matrix(
	simulator: "Simulator",
	Exsq: Array,
	D: Array,
	D_err: Array | None = None,
	colors: list[str] = COLORS,
	ax: Axes | None = None,
	ax_ell: Axes | None = None
) -> tuple[object, Axes, Axes | None]:
	"""Plot effective diffusion matrix diagnostics."""
	if ax is None:
		if simulator.field.dim == 2:
			fig, (ax, ax_ell) = plt.subplots(ncols=2, figsize=(12, 6))
		else:
			fig, ax = plt.subplots()
			ax_ell = None
	else:
		fig = ax.get_figure()

	ci = 0
	lines = []
	for i in range(simulator.field.dim):
		for j in range(i, simulator.field.dim):
			spacing = len(simulator.ts) // 10
			ax.plot(simulator.ts[::spacing], jnp.transpose(np.asarray(Exsq[:, ::spacing, i, j])), ".", c=colors[ci], markersize=3)
			line, = ax.plot(simulator.ts, 2 * D[i, j] * simulator.ts, "--", c=colors[ci], label=fr"$D_{{{i},{j}}} t$")
			lines.append(line)
			ci += 1

	ax.set_xlabel(r"$t$")
	ax.set_ylabel(r"$\langle (x(t) - x(0))^2 \rangle$")
	legend2 = ax.legend(handles=lines, loc="center left", title="linear fits")
	dummy_line = ax.plot([], [], ".-", c="k", linewidth=0.2, alpha=0.1, label=fr"data $\langle x_ix_j \rangle_{{\text{{subgroup}}}}$")
	ax.legend(handles=dummy_line, loc="upper left")
	ax.add_artist(legend2)

	if simulator.field.dim == 2 and ax_ell is not None:
		plot_field(simulator.field, ax_x=ax_ell, plot_dim=2, show_cells=False)
		eigvals, eigvecs = jnp.linalg.eigh(D)
		scale = 4 * jnp.sqrt(jnp.max(eigvals)) / jnp.min(simulator.field.size)
		major_axis = eigvecs[:, 1] * jnp.sqrt(eigvals[1]) / scale
		minor_axis = eigvecs[:, 0] * jnp.sqrt(eigvals[0]) / scale
		signs = jnp.array([-1, 1])
		ax_ell.plot(signs * major_axis[0], signs * major_axis[1], "C3", lw=2, label=f"major axis: {eigvals[1]:.3f}")
		ax_ell.plot(signs * minor_axis[0], signs * minor_axis[1], "C4", lw=2, label=f"minor axis: {eigvals[0]:.3f}")
		ellipse = mpatches.Ellipse(
			(0, 0),
			width=2 * jnp.linalg.norm(major_axis),
			height=2 * jnp.linalg.norm(minor_axis),
			angle=float(jnp.degrees(jnp.arctan2(major_axis[1], major_axis[0]))),
			edgecolor='C5',
			facecolor="none",
			lw=2,
			label='diffusion ellipse'
		)
		ax_ell.add_patch(ellipse)
		ax_ell.legend(frameon=True, framealpha=0.9, loc='upper left')

	_ = D_err
	return fig, ax, ax_ell


def plot_simulator_comparison(
	sims: Collection["Simulator"],
	roots: Collection[zarr.Group],
	mean_errors: Collection[Array],
	errors: Array,
	times: Array,
	colors: list[str] = COLORS,
	line_styles: list[str] = ["-", "--", "-.", ":"],
	ids: Int[Array, "n_sample"] | None = None,
	ax_x: Axes | None = None,
	ax_error: Axes | None = None,
	ax_time: Axes | None = None
) -> tuple[object, tuple[Axes, Axes, Axes]]:
	"""Plot comparison of simulator trajectories, errors, and timings."""
	sims_list = list(sims)
	roots_list = list(roots)
	mean_errors_list = list(mean_errors)
	ids_arr = None if ids is None else np.asarray(ids)
	ids_len = 0 if ids_arr is None else int(ids_arr.size)
	show_first_plot = ids is None or ids_len > 0
	if ax_x is None or ax_error is None or ax_time is None:
		if show_first_plot:
			fig, (ax_x, ax_error, ax_time) = plt.subplots(ncols=3, figsize=(7, 3), constrained_layout=True)
		else:
			fig, (ax_error, ax_time) = plt.subplots(ncols=2, figsize=(7, 3), constrained_layout=True)
			# Keep return signature stable without occupying layout space.
			ax_x = fig.add_axes((0.0, 0.0, 0.0, 0.0))
			ax_x.set_axis_off()
	else:
		fig = ax_x.get_figure()

	assert ax_x is not None and ax_error is not None and ax_time is not None

	# Use a consistent style mapping: each dt gets a color, each solver gets a linestyle.
	unique_dts = sorted({float(sim.dt) for sim in sims_list})
	dt_to_color = {dt: colors[i % len(colors)] for i, dt in enumerate(unique_dts)}

	style_by_solver = {"ShARK": "--", "EulerHeun": ":"}
	for i, solver in enumerate(dict.fromkeys(sim.solver for sim in sims_list)):
		if solver not in style_by_solver:
			style_by_solver[solver] = line_styles[i % len(line_styles)]

	marker_by_solver = {"ShARK": "o", "EulerHeun": "D"}
	for solver in dict.fromkeys(sim.solver for sim in sims_list):
		if solver not in marker_by_solver:
			marker_by_solver[solver] = "o"

	for i, (sim, root) in enumerate(zip(sims_list, roots_list)):
		_ = root["traj_cont"]
		c = dt_to_color[float(sim.dt)]
		ls = style_by_solver.get(sim.solver, "-")
		if show_first_plot:
			plot_simulation_sample(sim, ax=ax_x, n_sample=1 if ids is None else ids_len, ids=ids, seed=sim.seed, plot_field_flag=(i == 0), field_args={"res": 30}, colors=[c], line_styles=[ls])
		if i == len(sims_list) - 1:
			break
		mean_error = mean_errors_list[i]
		ax_error.plot(sim.ts, mean_error, color=c, linestyle=ls, linewidth=0.8)

	ax_error.set_xlabel(r"$t$")
	ax_error.set_ylabel("mean absolute error")
	if not show_first_plot:
		ax_x.set_visible(False)
	leg_x = ax_x.get_legend()
	if leg_x is not None:
		leg_x.remove()
	leg_error = ax_error.get_legend()
	if leg_error is not None:
		leg_error.remove()

	errors_mean = jnp.array(errors)
	times_arr = jnp.asarray(times)
	ordered_solvers = list(dict.fromkeys(s.solver for s in sims_list))
	for sol in ordered_solvers:
		mask = jnp.array([s.solver == sol for s in sims_list])
		x_sol = jnp.asarray(errors_mean)[mask]
		y_sol = times_arr[mask]
		valid = jnp.isfinite(x_sol) & jnp.isfinite(y_sol)
		if bool(jnp.any(valid)):
			ax_time.plot(x_sol[valid], y_sol[valid], color="k", linestyle=style_by_solver.get(sol, "-"), linewidth=0.9)

	for e, t, sim in zip(errors_mean, times_arr, sims_list):
		if not (jnp.isfinite(e) and jnp.isfinite(t)):
			continue
		c = dt_to_color[float(sim.dt)]
		m = marker_by_solver.get(sim.solver, "o")
		ax_time.scatter(e, t, color=c, marker=m, edgecolors="k", linewidths=0.5, zorder=3)

	ax_time.set_xlabel("mean absolute error")
	ax_time.set_ylabel("compute time (s)")
	ax_time.set_xscale("log")
	ax_time.set_yscale("log")

	solver_handles = []
	for sol in ordered_solvers:
		solver_handles.append(
			Line2D(
				[0], [0],
				color="k",
				linestyle=style_by_solver.get(sol, "-"),
				marker=marker_by_solver.get(sol, "o"),
				markerfacecolor="white",
				markeredgecolor="k",
				markersize=6,
				label=sol,
			)
		)
	legend_solver = ax_time.legend(handles=solver_handles, title="solver", loc="upper right", handlelength=3.5)

	legend_dts = unique_dts[1:] if len(unique_dts) > 1 else []

	dt_handles = [
		Line2D(
			[0], [0],
			linestyle="None",
			marker="s",
			markerfacecolor=dt_to_color[dt],
			markeredgecolor="k",
			markersize=6,
			label=f"{dt:.1g}",
		)
		for dt in legend_dts
	]
	legend_dt = ax_error.legend(handles=dt_handles, title="dt =", loc="upper left")
	ax_time.add_artist(legend_solver)
	ax_error.add_artist(legend_dt)

	return fig, (ax_x, ax_error, ax_time)


__all__ = [
    "display_cell_scalar_field",
    "update_cell_scalar_field",
    "plot_simulation_sample",
    "plot_simulation_propagator",
    "plot_simulation_mfpt",
    "plot_simulation_disc_stationary_distribution",
	"plot_simulation_final_magnetization",
	"plot_simulation_model_mri_response",
	"plot_cell_residence_times",
    "plot_effective_diffusion_matrix",
    "plot_simulator_comparison",
]
