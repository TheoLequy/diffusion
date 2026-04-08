import numpy as np
import pandas as pd
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerPatch
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from typing import Any, Callable, TYPE_CHECKING, cast
from IPython.display import display
from jaxtyping import Float, Array, Bool, Int
from helpers.plotting_fields import plot_field

if TYPE_CHECKING:
    from coarse_graining.fields import Field
    from coarse_graining.simulations import Simulator
    from helpers.geometry_helpers import Tesselation


def _parse_vertex_label(label: str) -> tuple[str, tuple[int, ...]]:
    """Parse vertex label into type and ids."""
    label = label.strip()
    if label.startswith("c(") and label.endswith(")"):
        return ("c", (int(label[2:-1]),))
    if label.startswith("e(") and label.endswith(")"):
        parts = label[2:-1].split(",")
        if len(parts) != 2:
            raise ValueError(f"Edge label must have format 'e(i,j)', got '{label}'")
        return ("e", (int(parts[0].strip()), int(parts[1].strip())))
    raise ValueError(f"Vertex label must start with 'c(' or 'e(', got '{label}'")


def _get_vertex_position(
    label: str,
    tesselation: "Tesselation",
    edge_offset_scale: Float = 0.3,
    tangent_offset_scale: Float = 0.1,
) -> Float[Array, "2"]:
    """Compute 2D vertex position from tesselation geometry."""
    assert tesselation.dim == 2, f"Only 2D tesselation supported, got dim={tesselation.dim}"
    vtype, ids = _parse_vertex_label(label)

    if vtype == "c":
        cell_id = ids[0]
        assert 0 <= cell_id < tesselation.n_cells, f"Cell ID {cell_id} out of bounds"
        return tesselation.centers[cell_id]

    from_cell, to_cell = ids
    assert 0 <= from_cell < tesselation.n_cells, f"Cell ID {from_cell} out of bounds"
    degree = int(tesselation.degrees[from_cell])
    neighbors = tesselation.neighbors[from_cell, :degree]
    edge_idx = jnp.where(neighbors == to_cell)[0]
    if len(edge_idx) == 0:
        raise ValueError(f"No edge from cell {from_cell} to cell {to_cell} in tesselation")
    edge_id = int(edge_idx[0])

    edge_midpoint = tesselation.edge_midpoints[from_cell, edge_id]
    edge_normal = tesselation.normals[from_cell, edge_id]
    edge_distance = tesselation.edge_center_distances[from_cell, edge_id]
    rot90 = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    offset_vec = edge_distance * (edge_offset_scale * edge_normal + tangent_offset_scale * rot90 @ edge_normal)
    return edge_midpoint + offset_vec


def _darken_color(color: Any, mix_black: float = 0.3) -> tuple[float, float, float, float]:
    """Darken a color by mixing with black."""
    r, g, b, a = mcolors.to_rgba(color)
    s = 1.0 - mix_black
    return (r * s, g * s, b * s, a)


def _lighten_color(color: Any, mix_white: float = 0.3) -> tuple[float, float, float, float]:
    """Lighten a color by mixing with white."""
    r, g, b, a = mcolors.to_rgba(color)
    return (
        r + (1.0 - r) * mix_white,
        g + (1.0 - g) * mix_white,
        b + (1.0 - b) * mix_white,
        a,
    )


def _round_sig(x: float, sig: int = 1) -> float:
    """Round to significant figures."""
    if x == 0:
        return 0.0
    return float(np.round(x, sig - int(np.floor(np.log10(abs(x)))) - 1))


def _build_labeled_digraph(transition_matrix: np.ndarray, vertex_labels: list[str]) -> nx.DiGraph:
    """Build a directed graph from a weighted adjacency matrix and explicit node labels."""
    assert transition_matrix.shape[0] == transition_matrix.shape[1], "Transition matrix must be square"
    assert transition_matrix.shape[0] == len(vertex_labels), "vertex_labels length must match matrix size"
    graph = nx.DiGraph()
    graph.add_nodes_from(vertex_labels)
    for i, u in enumerate(vertex_labels):
        for j, v in enumerate(vertex_labels):
            weight = float(transition_matrix[i, j])
            if weight > 0:
                graph.add_edge(u, v, weight=weight)
    return graph


def plot_subgraph(
    transition_matrix: Float[Array, "n n"],
    vertex_labels: list[str],
    tesselation: "Tesselation",
    ax: Axes | None = None,
    edge_offset_scale: Float = 0.3,
    tangent_offset_scale: Float = 0.1,
    vertex_cmap: Callable = plt.get_cmap("hsv"),
    node_size: Float = 500,
    max_edge_width: Float = 0.3,
    arrow_head_width_mult: Float = 2.0,
    arrow_head_length_mult: Float = 3.0,
    show_labels: Bool = True,
    label_fontsize: Float = 6,
    show_legend: Bool = True,
) -> dict[str, list]:
    """Plot a directed subgraph with geometry-aware node placement."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 10))

    assert tesselation.dim == 2, f"Only 2D tesselation supported, got dim={tesselation.dim}"
    assert transition_matrix.shape[0] == transition_matrix.shape[1], "Transition matrix must be square"
    assert len(vertex_labels) == transition_matrix.shape[0], "vertex_labels length must match matrix size"

    pos_map = {
        label: tuple(np.asarray(_get_vertex_position(label, tesselation, edge_offset_scale, tangent_offset_scale), dtype=float))
        for label in vertex_labels
    }
    all_cell_ids: list[int] = []
    for label in vertex_labels:
        _, ids = _parse_vertex_label(label)
        all_cell_ids.extend(ids)
    max_id = int(max(all_cell_ids)) if all_cell_ids else 0

    def vertex_color(cell_id: int) -> Any:
        """Vertex color.
        
        Args:
            cell_id: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        if isinstance(vertex_cmap, mcolors.Colormap):
            norm_val = 0.0 if max_id <= 0 else float(cell_id) / max_id
            return vertex_cmap(norm_val)
        if callable(vertex_cmap):
            try:
                return vertex_cmap(int(cell_id))
            except TypeError:
                norm_val = 0.0 if max_id <= 0 else float(cell_id) / max_id
                return vertex_cmap(norm_val)
        return "gray"

    node_color_map = {label: vertex_color(_parse_vertex_label(label)[1][0]) for label in vertex_labels}
    node_color_map_light = {label: _lighten_color(color, 0.3) for label, color in node_color_map.items()}

    G = _build_labeled_digraph(np.asarray(transition_matrix, dtype=float), vertex_labels)
    edge_items = list(G.edges(data=True))
    edge_weights = [float(data.get("weight", 0.0)) for _, _, data in edge_items]
    max_sqrt_w = float(np.sqrt(max(edge_weights))) if edge_weights and max(edge_weights) > 0 else 1.0
    edge_colors = [_darken_color(node_color_map[v], 0.3) for _, v, _ in edge_items]
    edge_widths = [float(max_edge_width * np.sqrt(w) / max_sqrt_w) for w in edge_weights]

    center_labels = [label for label in vertex_labels if label.startswith("c(")]
    other_labels = [label for label in vertex_labels if not label.startswith("c(")]

    node_artists: list[Any] = []
    if other_labels:
        nodes_other = nx.draw_networkx_nodes(
            G,
            pos_map,
            nodelist=other_labels,
            node_color=[node_color_map_light[n] for n in other_labels],
            node_shape="s",
            node_size=node_size,
            edgecolors="black",
            linewidths=0.6,
            ax=ax,
        )
        if nodes_other is not None:
            node_artists.append(nodes_other)

    if center_labels:
        nodes_center = nx.draw_networkx_nodes(
            G,
            pos_map,
            nodelist=center_labels,
            node_color=[node_color_map_light[n] for n in center_labels],
            node_shape="o",
            node_size=node_size,
            edgecolors="black",
            linewidths=0.6,
            ax=ax,
        )
        if nodes_center is not None:
            node_artists.append(nodes_center)

    edge_artists = nx.draw_networkx_edges(
        G,
        pos_map,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowstyle="Simple",
        arrowsize=12,
        min_source_margin=2,
        min_target_margin=2,
        ax=ax,
    )

    if isinstance(edge_artists, list):
        for idx, patch in enumerate(edge_artists):
            width = edge_widths[idx] if idx < len(edge_widths) else 1.0
            patch.set_path_effects([
                mpatheffects.Stroke(linewidth=width + 0.6, foreground="black"),
                mpatheffects.Normal(),
            ])
            patch.set_arrowstyle(
                "Simple",
                head_length=arrow_head_length_mult * width,
                head_width=arrow_head_width_mult * width,
                tail_width=width,
            )

    label_artists: list[Any] = []
    if show_labels:
        label_dict = nx.draw_networkx_labels(G, pos_map, font_size=label_fontsize, ax=ax)
        label_artists = list(label_dict.values()) if label_dict else []

    legend_artist = None
    if show_legend:
        nonzero_weights = [w for w in edge_weights if w > 0]
        legend_rates = np.quantile(nonzero_weights, [0.25, 0.5, 0.75]) if nonzero_weights else np.array([0.1, 0.5, 1.0], dtype=float)
        legend_rates = np.array([_round_sig(float(rate), sig=1) for rate in legend_rates], dtype=float)
        legend_widths = [float(max_edge_width * np.sqrt(w) / max_sqrt_w) for w in legend_rates]

        class _ArrowHandler(HandlerPatch):
            """_ArrowHandler data structure and utilities.
            """
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                """Create artists.
                
                Args:
                    legend: Input parameter.
                    orig_handle: Input parameter.
                    xdescent: Input parameter.
                    ydescent: Input parameter.
                    width: Input parameter.
                    height: Input parameter.
                    fontsize: Input parameter.
                    trans: Input parameter.
                
                Returns:
                    Output value computed by this function.
                """
                arrow = mpatches.FancyArrowPatch(
                    (xdescent, ydescent + height / 2.0),
                    (xdescent + 2.0 * width, ydescent + height / 2.0),
                    arrowstyle="Simple",
                    mutation_scale=fontsize * 1.2,
                    lw=orig_handle.get_linewidth(),  # type: ignore
                    edgecolor=orig_handle.get_edgecolor(),  # type: ignore
                    facecolor=orig_handle.get_facecolor(),  # type: ignore
                )
                lw = orig_handle.get_linewidth()  # type: ignore
                arrow.set_arrowstyle(
                    "Simple",
                    head_length=arrow_head_length_mult * lw,
                    head_width=arrow_head_width_mult * lw,
                    tail_width=lw,
                )
                arrow.set_transform(trans)
                return [arrow]

        legend_handles = [
            mpatches.FancyArrowPatch((0, 0), (2, 0), arrowstyle="Simple", lw=w, edgecolor="gray", facecolor="gray")
            for w in legend_widths
        ]
        legend_labels = [f"r={rate:.2g}" for rate in legend_rates]
        legend_artist = ax.legend(
            legend_handles,
            legend_labels,
            title="transition rates",
            loc="upper right",
            frameon=True,
            framealpha=0.8,
            handletextpad=2.9,
            labelspacing=0.6,
            borderpad=0.8,
            handler_map={mpatches.FancyArrowPatch: _ArrowHandler()},
        )

    return {
        "nodes": node_artists,
        "edges": edge_artists if isinstance(edge_artists, list) else [edge_artists] if edge_artists else [],
        "labels": label_artists,
        "legend": [legend_artist] if legend_artist else [],
    }


def plot_cell_subgraph(
    r: Float[Array, "n+1 n+1"],
    tesselation: "Tesselation",
    cell_id: int,
    ax: Axes | None = None,
    edge_offset_scale: Float = 0.3,
    tangent_offset_scale: Float = 0.1,
    vertex_cmap: Callable = plt.get_cmap("hsv"),
    node_size: Float = 500,
    max_edge_width: Float = 0.3,
    arrow_head_width_mult: Float = 2.0,
    arrow_head_length_mult: Float = 3.0,
    show_labels: Bool = True,
    label_fontsize: Float = 6,
    show_legend: Bool = True,
    show_matrix: Bool = True,
) -> dict[str, list]:
    """Plot transition subgraph for one cell."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 10))

    assert tesselation.dim == 2, f"plot_cell_subgraph expects 2D tesselation, got dim={tesselation.dim}"
    assert 0 <= cell_id < tesselation.n_cells, f"cell_id={cell_id} out of bounds for {tesselation.n_cells} cells"

    num_neighbors = r.shape[0] - 1
    degree = int(tesselation.degrees[cell_id])
    assert num_neighbors == degree, f"Expected {num_neighbors} neighbors, got {degree}"

    neighbors = tesselation.neighbors[cell_id, :degree]
    neighbor_ids = [int(n) for n in neighbors]
    vertex_labels = [f"e({n},{cell_id})" for n in neighbor_ids] + [f"c({cell_id})"] + [f"e({cell_id},{n})" for n in neighbor_ids]

    perm = jnp.arange(1, num_neighbors + 1)
    perm = jnp.append(perm, 0)
    r_reordered = r[perm][:, perm]

    adjacency_mat = np.zeros((2 * num_neighbors + 1, 2 * num_neighbors + 1))
    adjacency_mat[:num_neighbors, num_neighbors] = r_reordered[:-1, num_neighbors]
    adjacency_mat[:num_neighbors, num_neighbors + 1 :] = r_reordered[:-1, :num_neighbors]
    adjacency_mat[num_neighbors, num_neighbors + 1 :] = r_reordered[num_neighbors, :num_neighbors]

    if show_matrix:
        display(
            pd.DataFrame(
                adjacency_mat,
                columns=pd.Series(vertex_labels, name="out_vert"),
                index=pd.Series(vertex_labels, name="in_vert"),
            )
            .style.format("{:.2f}")
            .background_gradient(cmap="Reds", vmin=0, axis=None)
            .set_caption("Transition Rates Subgraph")
        )

    return plot_subgraph(
        transition_matrix=jnp.asarray(adjacency_mat),
        vertex_labels=vertex_labels,
        tesselation=tesselation,
        ax=ax,
        edge_offset_scale=edge_offset_scale,
        tangent_offset_scale=tangent_offset_scale,
        vertex_cmap=vertex_cmap,
        node_size=node_size,
        max_edge_width=max_edge_width,
        arrow_head_width_mult=arrow_head_width_mult,
        arrow_head_length_mult=arrow_head_length_mult,
        show_labels=show_labels,
        label_fontsize=label_fontsize,
        show_legend=show_legend,
    )


def plot_cell_subgraph_interactive(
    tesselation: "Tesselation",
    cell_id: int,
    data: Any = None,
    r: Float[Array, "n+1 n+1"] | None = None,
    dt: Float = 0.002,
    edge_offset_scale: Float = 0.3,
    tangent_offset_scale: Float = 0.1,
    vertex_cmap: Callable = plt.get_cmap("hsv"),
    node_size: Float = 500,
    max_edge_width: Float = 0.3,
    arrow_head_width_mult: Float = 2.0,
    arrow_head_length_mult: Float = 3.0,
    show_labels: Bool = True,
    label_fontsize: Float = 6,
    show_legend: Bool = True,
    show_matrix: Bool = True,
    fit_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Interactive plot of one-cell transition graph with in-edge and out-edge controls."""
    from helpers.cell_subgraph_inference import fit_model_one_center

    assert tesselation.dim == 2, f"plot_cell_subgraph expects 2D tesselation, got dim={tesselation.dim}"
    assert 0 <= cell_id < tesselation.n_cells, f"cell_id={cell_id} out of bounds for {tesselation.n_cells} cells"

    if r is None:
        assert data is not None, "Must provide either r or data"
        fit_kwargs = {} if fit_kwargs is None else fit_kwargs
        r, success = fit_model_one_center(data, dt=dt, **fit_kwargs)
        if not success:
            print("Warning: Model fit did not converge successfully")

    assert r is not None
    num_neighbors = r.shape[0] - 1
    degree = int(tesselation.degrees[cell_id])
    assert num_neighbors == degree, f"Expected {num_neighbors} neighbors, got {degree}"

    neighbors = tesselation.neighbors[cell_id, :degree]
    neighbor_ids = [int(n) for n in neighbors]
    sorted_indices = np.argsort(neighbor_ids)
    sorted_neighbors = [neighbor_ids[i] for i in sorted_indices]

    permutation = jnp.append(jnp.array(sorted_indices), num_neighbors)
    p_inv = jnp.argsort(permutation)
    r_sorted = r[p_inv][:, p_inv]

    in_edge_labels = [f"e({n},{cell_id})" for n in sorted_neighbors]
    center_label = f"c({cell_id})"
    out_edge_labels = [f"e({cell_id},{n})" for n in sorted_neighbors]
    vertex_labels = in_edge_labels + [center_label] + out_edge_labels

    n_vertices = 2 * num_neighbors + 1
    adjacency_mat = np.zeros((n_vertices, n_vertices))
    adjacency_mat[:num_neighbors, num_neighbors] = r_sorted[:-1, -1]
    adjacency_mat[:num_neighbors, num_neighbors + 1 :] = r_sorted[:-1, :-1]
    adjacency_mat[num_neighbors, num_neighbors + 1 :] = r_sorted[-1, :-1]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(1, 3, width_ratios=[10, 1.5, 1.5], figure=fig, wspace=0.15)
    ax_main = fig.add_subplot(gs[0])
    ax_radio = fig.add_subplot(gs[1])
    ax_check = fig.add_subplot(gs[2])
    ax_radio.axis("off")
    ax_check.axis("off")

    if show_matrix:
        display(
            pd.DataFrame(
                adjacency_mat,
                columns=pd.Series(vertex_labels, name="out_vert"),
                index=pd.Series(vertex_labels, name="in_vert"),
            )
            .style.format("{:.2f}")
            .background_gradient(cmap="Reds", vmin=0, axis=None)
            .set_caption("Transition Rates Subgraph")
        )

    pos_map = {
        label: tuple(np.asarray(_get_vertex_position(label, tesselation, edge_offset_scale, tangent_offset_scale), dtype=float))
        for label in vertex_labels
    }
    max_id = max(sorted_neighbors + [cell_id])

    def vertex_color(cid: int) -> Any:
        """Vertex color.
        
        Args:
            cid: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        if isinstance(vertex_cmap, mcolors.Colormap):
            norm_val = float(cid) / max_id if max_id > 0 else 0.0
            return vertex_cmap(norm_val)
        return vertex_cmap(cid) if callable(vertex_cmap) else "gray"

    base_colors = {
        **{in_edge_labels[i]: vertex_color(sorted_neighbors[i]) for i in range(num_neighbors)},
        center_label: vertex_color(cell_id),
        **{out_edge_labels[i]: vertex_color(sorted_neighbors[i]) for i in range(num_neighbors)},
    }

    G = _build_labeled_digraph(adjacency_mat, vertex_labels)
    state = {"in_index": 0, "out_visible": [True] * num_neighbors}

    def get_active_colors() -> dict[str, Any]:
        """Return active colors.
        
        Returns:
            Output value computed by this function.
        """
        in_idx = state["in_index"]
        colors: dict[str, Any] = {}
        for i, label in enumerate(in_edge_labels):
            colors[label] = _lighten_color(base_colors[label], 0.3) if i == in_idx else _lighten_color("gray", 0.3)
        colors[center_label] = _lighten_color(base_colors[center_label], 0.3)
        for i, label in enumerate(out_edge_labels):
            colors[label] = _lighten_color(base_colors[label], 0.3) if state["out_visible"][i] else _lighten_color("gray", 0.3)
        return colors

    def get_edge_colors_widths() -> tuple[list[Any], list[float]]:
        """Return edge colors widths.
        
        Returns:
            Output value computed by this function.
        """
        in_idx = state["in_index"]
        edge_list = list(G.edges(data=True))
        colors: list[Any] = []
        widths: list[float] = []
        max_weight = max((float(d.get("weight", 0.0)) for _, _, d in edge_list), default=1.0)
        max_sqrt_w = float(np.sqrt(max_weight)) if max_weight > 0 else 1.0
        for u, v, d in edge_list:
            weight = float(d.get("weight", 0.0))
            width = max_edge_width * np.sqrt(weight) / max_sqrt_w if max_sqrt_w > 0 else 0.01
            u_in_idx = in_edge_labels.index(u) if u in in_edge_labels else None
            v_out_idx = out_edge_labels.index(v) if v in out_edge_labels else None
            is_relevant = False
            if u_in_idx == in_idx and v == center_label:
                is_relevant = True
            elif u_in_idx == in_idx and v_out_idx is not None and state["out_visible"][v_out_idx]:
                is_relevant = True
            elif u == center_label and v_out_idx is not None and state["out_visible"][v_out_idx]:
                is_relevant = True
            colors.append(_darken_color(base_colors[v], 0.3) if is_relevant else "lightgray")
            widths.append(width)
        return colors, widths

    def redraw() -> None:
        """Redraw.
        """
        ax_main.clear()
        node_colors = get_active_colors()
        edge_colors, edge_widths = get_edge_colors_widths()

        if in_edge_labels:
            nx.draw_networkx_nodes(
                G,
                pos_map,
                nodelist=in_edge_labels,
                node_color=[node_colors[n] for n in in_edge_labels],
                node_shape="s",
                node_size=node_size,
                edgecolors="black",
                linewidths=0.6,
                ax=ax_main,
            )

        nx.draw_networkx_nodes(
            G,
            pos_map,
            nodelist=[center_label],
            node_color=[node_colors[center_label]],
            node_shape="o",
            node_size=node_size,
            edgecolors="black",
            linewidths=0.6,
            ax=ax_main,
        )

        if out_edge_labels:
            nx.draw_networkx_nodes(
                G,
                pos_map,
                nodelist=out_edge_labels,
                node_color=[node_colors[n] for n in out_edge_labels],
                node_shape="s",
                node_size=node_size,
                edgecolors="black",
                linewidths=0.6,
                ax=ax_main,
            )

        edge_collection = nx.draw_networkx_edges(
            G,
            pos_map,
            width=edge_widths,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle="Simple",
            arrowsize=12,
            min_source_margin=2,
            min_target_margin=2,
            ax=ax_main,
        )

        if isinstance(edge_collection, list):
            for idx, patch in enumerate(edge_collection):
                width = edge_widths[idx] if idx < len(edge_widths) else 1.0
                patch.set_path_effects([
                    mpatheffects.Stroke(linewidth=width + 0.6, foreground="black"),
                    mpatheffects.Normal(),
                ])
                patch.set_arrowstyle(
                    "Simple",
                    head_length=arrow_head_length_mult * width,
                    head_width=arrow_head_width_mult * width,
                    tail_width=width,
                )

        if show_labels:
            nx.draw_networkx_labels(G, pos_map, font_size=label_fontsize, ax=ax_main)

        if show_legend:
            edge_list = list(G.edges(data=True))
            edge_weights = [float(d.get("weight", 0.0)) for _, _, d in edge_list]
            nonzero_weights = [w for w in edge_weights if w > 0]
            legend_rates = np.quantile(nonzero_weights, [0.25, 0.5, 0.75]) if nonzero_weights else np.array([0.1, 0.5, 1.0])
            legend_rates = np.array([_round_sig(float(rate), sig=1) for rate in legend_rates])
            max_weight = max(edge_weights) if edge_weights else 1.0
            max_sqrt_w = float(np.sqrt(max_weight)) if max_weight > 0 else 1.0
            legend_widths = [max_edge_width * np.sqrt(w) / max_sqrt_w for w in legend_rates]
            legend_handles = [
                mpatches.FancyArrowPatch((0, 0), (2, 0), arrowstyle="Simple", lw=w, edgecolor="gray", facecolor="gray")
                for w in legend_widths
            ]
            legend_labels = [f"r={rate:.2g}" for rate in legend_rates]
            ax_main.legend(
                legend_handles,
                legend_labels,
                title="transition rates",
                loc="upper right",
                frameon=True,
                framealpha=0.8,
            )

        ax_main.set_aspect("equal")
        ax_main.axis("off")
        fig.canvas.draw_idle()

    radio_labels = [f"{sorted_neighbors[i]}" for i in range(num_neighbors)]
    radio = RadioButtons(ax_radio, radio_labels, active=0)

    def on_radio_change(label: str | None) -> None:
        """Handle the radio change event.
        
        Args:
            label: Input parameter.
        """
        if label is None:
            return
        state["in_index"] = radio_labels.index(label)
        redraw()

    radio.on_clicked(on_radio_change)

    check_labels = [f"{sorted_neighbors[i]}" for i in range(num_neighbors)]
    check = CheckButtons(ax_check, check_labels, [True] * num_neighbors)

    def on_check_change(label: str | None) -> None:
        """Handle the check change event.
        
        Args:
            label: Input parameter.
        """
        if label is None:
            return
        idx = check_labels.index(label)
        state["out_visible"][idx] = not state["out_visible"][idx]
        redraw()

    check.on_clicked(on_check_change)
    redraw()

    return {
        "fig": fig,
        "axes": {"main": ax_main, "radio": ax_radio, "check": ax_check},
        "widgets": {"radio": radio, "check": check},
        "state": state,
    }


def plot_graph_model_eigenmodes(
    simulator: "Simulator",
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    vertex_labels: list[str],
    ax_main: Axes | None = None,
    ax_eig: Axes | None = None,
    ax_slider: Axes | None = None,
    ax_radio: Axes | None = None,
    eigval_init: float | None = None,
    part_init: str = "real"
) -> tuple[object, Axes, Axes, Axes | None, Axes | None]:
    """Plot graph eigenmodes with optional interactive selection."""
    if simulator.field.dim != 2:
        raise NotImplementedError("fit_graph_model plotting is only implemented for 2D fields.")
    if simulator.field.tesselation is None:
        raise ValueError("Tesselation must be defined in the field to plot graph eigenmodes.")
    if len(vertex_labels) == 0:
        raise ValueError("No vertex labels found for plotting.")

    positions = np.array([np.asarray(_get_vertex_position(label, simulator.field.tesselation), dtype=float) for label in vertex_labels])
    x = positions[:, 0]
    y = positions[:, 1]

    size_ratio = float(simulator.field.size[0] / simulator.field.size[1])
    main_width = 12.0
    main_height = max(6.0, main_width / max(size_ratio, 1e-6))
    fig_width = 15.0
    fig_height = max(6.0, main_height)

    use_widgets = ax_main is None and ax_eig is None and ax_slider is None and ax_radio is None
    if use_widgets:
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        gs = GridSpec(2, 3, height_ratios=[12, 1], width_ratios=[12, 2.2, 0.6], figure=fig)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_eig = fig.add_subplot(gs[0, 1])
        ax_slider = fig.add_subplot(gs[0, 2])
        ax_radio = fig.add_subplot(gs[1, 1])
        ax_radio.axis("off")
    else:
        assert ax_main is not None and ax_eig is not None
        fig = ax_main.get_figure()
    fig_any = cast(Any, fig)

    ax_main.set_aspect(size_ratio, adjustable="box")
    ax_main.set_xlim(float(-simulator.field.size[0] / 2), float(simulator.field.size[0] / 2))
    ax_main.set_ylim(float(-simulator.field.size[1] / 2), float(simulator.field.size[1] / 2))
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")

    eig_real = np.real(eigvals)
    eig_imag = np.imag(eigvals)
    eig_x = -eig_imag
    eig_y = eig_real

    ax_eig.scatter(eig_x, eig_y, s=18, c="gray", alpha=0.7)
    selected_point = ax_eig.scatter([eig_x[0]], [eig_y[0]], s=60, c="C3", zorder=3)
    ax_eig.set_xlabel(r"$-\mathrm{Im}(\lambda)$")
    ax_eig.set_ylabel(r"$\mathrm{Re}(\lambda)$")

    v0 = eigvecs[:, 0]
    v0_real = np.real(v0)
    vmax = float(np.max(np.abs(v0_real))) if v0_real.size > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    scatter = ax_main.scatter(x, y, c=v0_real, cmap=cmap, norm=norm, s=18)
    cbar = fig_any.colorbar(scatter, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label(r"eigenvector")

    def update_selected(idx: int, part: str) -> None:
        """Update selected.
        
        Args:
            idx: Input parameter.
            part: Input parameter.
        """
        vec = eigvecs[:, idx]
        values = np.real(vec) if part == "real" else np.imag(vec)
        vabs = float(np.max(np.abs(values))) if values.size > 0 else 1.0
        if vabs == 0:
            vabs = 1.0
        scatter.set_array(values)
        scatter.set_norm(mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs))
        cbar.update_normal(scatter)
        selected_point.set_offsets(np.array([[eig_x[idx], eig_y[idx]]]))
        fig_any.canvas.draw_idle()

    if use_widgets:
        assert ax_slider is not None and ax_radio is not None
        real_min = float(np.min(eig_real))
        real_max = float(np.max(eig_real))
        if real_min == real_max:
            real_min -= 1.0
            real_max += 1.0
        if eigval_init is None:
            eigval_init = float(eig_real[0])
        slider = Slider(ax_slider, r"$\mathrm{Re}(\lambda)$", valmin=real_min, valmax=real_max, valinit=eigval_init, orientation="vertical")
        radio = RadioButtons(ax_radio, ["real", "imag"], active=0 if part_init == "real" else 1)

        state = {"part": part_init}

        def on_slider(val: float) -> None:
            """Handle the slider event.
            
            Args:
                val: Input parameter.
            """
            idx = int(np.argmin(np.abs(eig_real - val)))
            update_selected(idx, state["part"])

        def on_radio(label: str | None) -> None:
            """Handle the radio event.
            
            Args:
                label: Input parameter.
            """
            if label is None:
                return
            state["part"] = label
            idx = int(np.argmin(np.abs(eig_real - slider.val)))
            update_selected(idx, state["part"])

        slider.on_changed(on_slider)
        radio.on_clicked(on_radio)
    else:
        if eigval_init is None:
            eigval_init = float(eig_real[0])
        idx = int(np.argmin(np.abs(eig_real - eigval_init)))
        update_selected(idx, part_init)

    return fig, ax_main, ax_eig, ax_slider if use_widgets else None, ax_radio if use_widgets else None


def plot_graph(
    locations,
    laplacian,
    size,
    stationary=None,
    ax: Axes | None = None,
    edge_color: str = "k",
    node_color: str = "C0",
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    min_edge_weight: float = 0.0,
):
    """Plot an undirected graph from a Hermitianized Laplacian on a periodic 2D box.

    If `stationary` is provided, node sizes are proportional to stationary values.
    """
    locations_np = np.asarray(locations, dtype=float)
    laplacian_np = np.asarray(laplacian)
    size_np = np.asarray(size, dtype=float)
    stationary_np = None if stationary is None else np.asarray(stationary, dtype=float)

    assert locations_np.ndim == 2, f"locations must have shape (n_nodes, dim), got {locations_np.shape}"
    assert laplacian_np.ndim == 2 and laplacian_np.shape[0] == laplacian_np.shape[1], "laplacian must be square"
    assert laplacian_np.shape[0] == locations_np.shape[0], "laplacian size must match number of locations"
    assert size_np.ndim == 1 and size_np.shape[0] == locations_np.shape[1], "size must have one entry per dimension"
    assert locations_np.shape[1] == 2, "plot_graph currently supports 2D locations"
    if stationary_np is not None:
        assert stationary_np.ndim == 1 and stationary_np.shape[0] == locations_np.shape[0], "stationary must have one value per node"

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    lap_real = np.real_if_close(laplacian_np)
    if np.iscomplexobj(lap_real):
        lap_real = lap_real.real

    if stationary_np is not None:
        stationary_pos = np.clip(stationary_np, a_min=0.0, a_max=None)
        mean_stationary = float(np.mean(stationary_pos))
        if mean_stationary > 0:
            node_sizes = node_size_scale * (stationary_pos / mean_stationary)
        else:
            node_sizes = np.full_like(stationary_pos, fill_value=node_size_scale, dtype=float)
    else:
        node_sizes = node_size_scale * np.clip(np.diag(lap_real), a_min=0.0, a_max=None)

    line_segments = []
    line_widths = []

    n_nodes = locations_np.shape[0]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_weight = float(np.abs(lap_real[i, j]))
            if edge_weight <= min_edge_weight:
                continue

            delta = locations_np[j] - locations_np[i]
            wrap_shift = np.round(delta / size_np) * size_np
            delta_min = delta - wrap_shift
            loc_j_image = locations_np[i] + delta_min

            line_segments.append([locations_np[i], loc_j_image])
            line_widths.append(edge_width_scale * edge_weight)

            if np.any(wrap_shift != 0):
                loc_i_image = locations_np[j] - delta_min
                line_segments.append([locations_np[j], loc_i_image])
                line_widths.append(edge_width_scale * edge_weight)

    if line_segments:
        edge_collection = LineCollection(line_segments, colors=edge_color, linewidths=line_widths, zorder=1)
        ax.add_collection(edge_collection)

    edge_weights_np = np.asarray(line_widths, dtype=float) / float(edge_width_scale) if line_widths else np.asarray([], dtype=float)
    if edge_weights_np.size > 0:
        w_min = float(np.min(edge_weights_np))
        w_max = float(np.max(edge_weights_np))
        rounded_max = w_max
        if rounded_max > 0:
            p10 = 10.0 ** np.floor(np.log10(rounded_max))
            rounded_max = np.floor(rounded_max / p10) * p10

        rate_candidates = np.asarray([rounded_max, 0.5 * rounded_max, 0.25 * rounded_max], dtype=float)

        rate_values = np.asarray([float(f"{v:.1g}") for v in rate_candidates], dtype=float)
        rate_values = np.clip(rate_values, w_min, w_max)

        edge_handles = [
            Line2D([0], [0], color=edge_color, linewidth=max(edge_width_scale * float(v), 0.6), label=f"{v:.1g}")
            for v in rate_values
        ]
        ax.legend(handles=edge_handles, title="rate", loc="upper right", frameon=True, framealpha=0.8)

    node_artist = ax.scatter(locations_np[:, 0], locations_np[:, 1], s=node_sizes, c=node_color, zorder=2)

    half = size_np / 2
    ax.set_xlim(-half[0], half[0])
    ax.set_ylim(-half[1], half[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    return ax, node_artist


def plot_inferred_graph_on_field(
    field: "Field",
    transition_matrix: np.ndarray,
    vertex_labels: list[str],
    ax: Axes | None = None,
    show_field: bool = True,
    field_kwargs: dict | None = None,
    min_edge_rate: float = 0.0,
    max_edge_width: float = 2.0,
    node_size: float = 45.0,
    show_labels: bool = False,
) -> tuple[object, Axes]:
    """Overlay the inferred graph on a 2D field/tessellation view."""
    assert field.tesselation is not None, "Field tesselation is required to plot inferred graph."
    assert field.dim == 2, f"Only 2D fields are supported, got dim={field.dim}."

    matrix = np.asarray(transition_matrix, dtype=float)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "transition_matrix must be square"
    assert matrix.shape[0] == len(vertex_labels), "vertex_labels length must match transition_matrix size"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    else:
        fig = ax.get_figure()

    if show_field:
        kwargs = {} if field_kwargs is None else dict(field_kwargs)
        kwargs.setdefault("plot_dim", 2)
        kwargs.setdefault("show_cells", False)
        kwargs.setdefault("verbose", False)
        plot_field(field, ax_x=ax, ax_k=None, **kwargs)
    else:
        half = np.asarray(field.size, dtype=float) / 2
        ax.set_xlim(-half[0], half[0])
        ax.set_ylim(-half[1], half[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    graph = _build_labeled_digraph(matrix, vertex_labels)
    pos_map = {
        label: tuple(np.asarray(_get_vertex_position(label, field.tesselation), dtype=float))
        for label in vertex_labels
    }

    n_cells = int(field.num_vert)
    cmap = plt.get_cmap("hsv")

    def _node_color(label: str):
        """Internal helper to node color.
        
        Args:
            label: Input parameter.
        
        Returns:
            Output value computed by this function.
        """
        vtype, ids = _parse_vertex_label(label)
        if vtype == "c":
            cid = ids[0]
        else:
            cid = ids[0]
        frac = 0.0 if n_cells <= 1 else float(cid) / float(max(n_cells - 1, 1))
        return cmap(frac)

    center_nodes = [label for label in vertex_labels if label.startswith("c(")]
    edge_nodes = [label for label in vertex_labels if label.startswith("e(")]

    if edge_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos=pos_map,
            nodelist=edge_nodes,
            node_shape="s",
            node_size=node_size,
            node_color=[_lighten_color(_node_color(label), 0.25) for label in edge_nodes],
            edgecolors="black",
            linewidths=0.4,
            ax=ax,
        )

    if center_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos=pos_map,
            nodelist=center_nodes,
            node_shape="o",
            node_size=1.35 * node_size,
            node_color=[_lighten_color(_node_color(label), 0.15) for label in center_nodes],
            edgecolors="black",
            linewidths=0.6,
            ax=ax,
        )

    edge_items = list(graph.edges(data=True))
    rates = [float(d.get("weight", 0.0)) for _, _, d in edge_items]
    max_rate = max(rates) if rates else 1.0
    widths = [
        0.0 if rate <= min_edge_rate else max_edge_width * np.sqrt(rate / max_rate)
        for rate in rates
    ]
    visible_edges = [item for item, width in zip(edge_items, widths) if width > 0]
    visible_widths = [width for width in widths if width > 0]
    visible_colors = [_darken_color(_node_color(v), 0.25) for _, v, _ in visible_edges]

    if visible_edges:
        edge_collection = nx.draw_networkx_edges(
            graph,
            pos=pos_map,
            edgelist=[(u, v) for u, v, _ in visible_edges],
            width=visible_widths,
            edge_color=visible_colors,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            alpha=0.8,
            min_source_margin=2,
            min_target_margin=2,
            ax=ax,
        )
        if isinstance(edge_collection, list):
            for patch in edge_collection:
                patch.set_path_effects([
                    mpatheffects.Stroke(linewidth=patch.get_linewidth() + 0.4, foreground="black"),
                    mpatheffects.Normal(),
                ])

    if show_labels:
        labels = {label: label for label in vertex_labels}
        nx.draw_networkx_labels(graph, pos=pos_map, labels=labels, font_size=6, ax=ax)

    return fig, ax

__all__ = [
    "_parse_vertex_label",
    "_get_vertex_position",
    "_darken_color",
    "_lighten_color",
    "_round_sig",
    "_build_labeled_digraph",
    "plot_subgraph",
    "plot_cell_subgraph",
    "plot_cell_subgraph_interactive",
    "plot_graph_model_eigenmodes",
    "plot_graph",
    "plot_inferred_graph_on_field",
]
