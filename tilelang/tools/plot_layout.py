from __future__ import annotations
import tilelang.language as T
import itertools


def plot_layout(
    layout,
    save_directory="./tmp",
    name: str = "layout",
    colormap: str = None,
    verbose: bool = False,
    formats: str | list[str] = "pdf",
    view: str = "input",
    grid_shape: tuple[int, int] | None = None,
) -> None:
    """
    Plot the layout mapping as a 2D grid visualization.

    Dispatches to Fragment-specific or Layout-specific plotting based on the
    type of the layout object.

    Parameters
    ----------
    layout : T.Layout or T.Fragment
        The layout object to visualize.
    save_directory : str, optional
        Output directory (default "./tmp").
    name : str, optional
        Base filename for saved images (default "layout").
    colormap : str, optional
        Matplotlib colormap name. Defaults to "RdPu" for Fragment, "Spectral" for Layout.
    verbose : bool, optional
        If True, print mapping details.
    formats : str | list[str], optional
        Output format(s): "pdf", "png", "svg", "all", or comma-separated (default "pdf").
    view : str, optional
        For `T.Layout` only: choose which space is shown as the 2D grid.

        - "input" (default): grid is input space, labels show output (flattened) coordinates.
        - "output": grid is output space, labels show input coordinates.
    grid_shape : tuple[int, int] | None, optional
        For `view="input"` only: override the 2D grid shape (rows, cols). The
        product must match the total number of input elements.
    """
    from tilelang.layout.fragment import Fragment

    if isinstance(layout, Fragment):
        _plot_fragment_layout(layout, save_directory, name, colormap or "RdPu", verbose, formats)
    elif isinstance(layout, T.Layout):
        _plot_layout_map(
            layout,
            save_directory,
            name,
            colormap or "Spectral",
            verbose,
            formats,
            view=view,
            grid_shape=grid_shape,
        )
    else:
        raise TypeError(f"Expected T.Layout or T.Fragment, but got {type(layout).__name__}.")


def _parse_formats(formats):
    """Parse the formats parameter into a list of format strings."""
    if isinstance(formats, str):
        formats_str = formats.strip().lower()
        if formats_str == "all":
            return ["pdf", "png", "svg"]
        elif "," in formats_str:
            return [f.strip() for f in formats_str.split(",")]
        else:
            return [formats_str]
    else:
        raise TypeError(
            f"Expected str, but got {type(formats).__name__}. Please pass a string like 'png', 'pdf', 'svg', 'all', or 'png,pdf'."
        )


def _save_plot(plt, save_directory, name, formats):
    """Save the current matplotlib figure in the specified format(s)."""
    import os
    import pathlib

    formats_list = _parse_formats(formats)

    tmp_directory = pathlib.Path(save_directory)
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    if "pdf" in formats_list:
        pdf_path = tmp_directory / f"{name}.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved pdf format into {pdf_path}")

    if "png" in formats_list:
        png_path = tmp_directory / f"{name}.png"
        plt.savefig(png_path, bbox_inches="tight", transparent=False, dpi=255)
        print(f"Saved png format into {png_path}")

    if "svg" in formats_list:
        svg_path = tmp_directory / f"{name}.svg"
        plt.savefig(svg_path, bbox_inches="tight", format="svg")
        print(f"Saved svg format into {svg_path}")


# ---------------------------------------------------------------------------
# Fragment-specific layout visualization (thread ID + local ID per cell)
# ---------------------------------------------------------------------------


def _plot_fragment_layout(
    layout: T.Fragment,
    save_directory="./tmp",
    name: str = "layout",
    colormap: str = "RdPu",
    verbose: bool = False,
    formats: str | list[str] = "pdf",
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Get the input shape of the layout and convert it to a list of integers
    input_shape = layout.get_input_shape()
    input_shape = [int(var) for var in input_shape]
    replicate_size = int(layout.replicate_size)

    # Get the total number of threads
    num_threads = int(layout.get_thread_size())

    # Initialize a 2D array to store thread mappings
    thread_map = np.empty(input_shape, dtype=object)
    for idx in np.ndindex(thread_map.shape):
        thread_map[idx] = []

    # Initialize a 2D array to store value mappings
    value_map = np.zeros(input_shape, dtype=object)
    for idx in np.ndindex(value_map.shape):
        value_map[idx] = []

    # Iterate over all possible indices in the input shape
    for i in range(replicate_size):
        for idx in itertools.product(*[range(dim) for dim in input_shape]):
            index = list(idx)
            # If replication is enabled, adjust the index
            if replicate_size > 1:
                index.insert(0, i)
            # Map the index to a thread ID
            thread_id = layout.map_forward_thread(index)
            assert len(thread_id) == 1  # Ensure a single-thread mapping
            thread_map[idx].append(int(thread_id[0]))  # Store the thread ID

    # Iterate again to map values
    for i in range(replicate_size):
        for idx in itertools.product(*[range(dim) for dim in input_shape]):
            index = list(idx)
            if replicate_size > 1:
                index.insert(0, i)
            thread_id = layout.map_forward_thread(index)
            value_id = layout.map_forward_index(index)
            assert len(value_id) == 1  # Ensure a single-value mapping
            value_map[idx].append(int(value_id[0]))  # Store the value ID

    # Load the colormap with twice as many colors as the number of threads
    cmap = plt.get_cmap(colormap, num_threads * 2 // replicate_size)

    # Generate a list of colors based on the colormap
    raw_colors = [cmap(i) for i in range(num_threads)]
    colors = raw_colors.copy()

    # Show the distribution of registers in each thread of a warp.
    warp_size = 32
    # Warn if the number of threads is less than the warp size
    if num_threads < warp_size:
        import warnings

        warnings.warn(
            f"Layout visualization has {num_threads} threads, which is less than the warp size ({warp_size}). "
            f"For the best viewing experience, it is recommended to have at least {warp_size} threads.",
            UserWarning,
            stacklevel=2,
        )
    spectral_camp = plt.get_cmap("hsv", warp_size * 6)

    for i in range(min(warp_size, num_threads)):
        colors[i] = spectral_camp(i * 6)

    # Determine the number of rows and columns in the input shape
    nrows, ncols = input_shape
    # Adjust figure size to maintain square cells
    cell_size = 1  # Base size for each cell
    plt.figure(figsize=(cell_size * ncols, cell_size * nrows))  # Set the figure size proportionally
    ax = plt.gca()  # Get the current axis
    font_size = 24  # Set font size for text annotation

    # Iterate through each row and column
    for i in range(nrows):
        for j in range(ncols):
            thread_ids = thread_map[i, j]  # Get the thread ID
            local_ids = value_map[i, j]  # Get the value ID
            if verbose:
                print(f"thread_map[{i}, {j}] = {thread_ids} value_map[{i}, {j}] = {local_ids}")

            color = colors[thread_ids[0]]  # Select color based on thread ID
            # Create a rectangle patch for visualization
            rect = patches.Rectangle((j, i), 1, 1, linewidth=0.5, edgecolor="black", facecolor=color)
            ax.add_patch(rect)  # Add the rectangle to the plot

            # Add text annotations inside the rectangles
            thread_str = []
            for thread_id in thread_ids:
                thread_str.append(f"{thread_id}")
            thread_str = "T" + "/".join(thread_str)
            local_id = local_ids[0]
            # assert local id in local_ids is equal
            assert all(local_id == local_id for local_id in local_ids)

            # Calculate thread font size based on string length
            thread_fontsize = min(font_size, font_size * (4 / len(thread_str)))

            # Add thread ID text with adjusted font size
            ax.text(j + 0.5, i + 0.3, thread_str, ha="center", va="center", color="black", fontsize=thread_fontsize)
            # Add local ID text with original font size
            ax.text(j + 0.5, i + 0.7, f"L{local_id}", ha="center", va="center", color="black", fontsize=font_size)

    # Add row labels to the left side of the plot
    for i in range(nrows):
        text = f"row {i}"
        ax.text(-0.75, i + 0.5, text, ha="center", va="center", color="black", fontsize=font_size)

    # Add column labels at the top of the plot
    for j in range(ncols):
        text = f"col {j}"
        ax.text(j + 0.5, -0.5, text, ha="center", va="center", color="black", fontsize=font_size, rotation=45)

    # Set the plot limits
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()  # Invert the y-axis for proper visualization
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks

    # Calculate legend position based on figure size
    fig = plt.gcf()
    fig_width = fig.get_size_inches()[0]
    fig_height = fig.get_size_inches()[1]
    legend_x = 1.0 + (0.5 / fig_width)  # Adjust x position based on figure width
    legend_y = 1.0 + (1.7 / fig_height)  # Adjust y position based on figure height

    legend_patches = [patches.Patch(color="black", label="T: Thread ID"), patches.Patch(color="black", label="L: Local ID")]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=font_size - 4,
        frameon=False,
        bbox_to_anchor=(legend_x, legend_y),  # Dynamic position
        ncols=2,
    )

    plt.tight_layout()
    _save_plot(plt, save_directory, name, formats)
    plt.close()


# ---------------------------------------------------------------------------
# Layout-specific visualization (position mapping, no thread/local ID)
# ---------------------------------------------------------------------------


def _plot_layout_map(
    layout: T.Layout,
    save_directory="./tmp",
    name: str = "layout",
    colormap: str = "Spectral",
    verbose: bool = False,
    formats: str | list[str] = "pdf",
    view: str = "input",
    grid_shape: tuple[int, int] | None = None,
) -> None:
    """
    Visualize a Layout object as a 2D grid showing position mappings.

    By default (`view="input"`), the grid represents the input space (viewed as
    2D by keeping the last dimension and flattening all preceding dimensions),
    and each cell displays the output (flattened) coordinate.

    With `view="output"`, the grid represents the output space (similarly viewed
    as 2D), and each cell displays the original input coordinate that maps to
    that output position.

    Parameters
    ----------
    layout : T.Layout
        The layout object to visualize.
    save_directory : str
        Output directory.
    name : str
        Base filename.
    colormap : str
        Matplotlib colormap for coloring cells by source position.
    verbose : bool
        Print mapping details.
    formats : str | list[str]
        Output format(s).
    view : str
        Choose which space is shown as the 2D grid: "input" (default) or
        "output".
    grid_shape : tuple[int, int] | None
        For `view="input"` only: override the 2D grid shape (rows, cols). The
        product must match the total number of input elements.
    """
    import functools
    import operator
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    input_shape = [int(v) for v in layout.get_input_shape()]
    total_in = functools.reduce(operator.mul, input_shape, 1)

    # -- helpers for N-D → 2-D conversion --------------------------------

    def _flatten_to_2d(shape):
        """Keep last dim, merge all preceding dims into one row dim."""
        if len(shape) <= 1:
            return (1, shape[0]) if shape else (1, 1)
        return (functools.reduce(operator.mul, shape[:-1], 1), shape[-1])

    def _nd_to_2d(idx, shape):
        """Convert an N-D index to (row, col) in the flattened 2-D view."""
        if len(shape) <= 1:
            return (0, idx[0]) if shape else (0, 0)
        row = 0
        for k in range(len(shape) - 1):
            row = row * shape[k] + idx[k]
        return (row, idx[-1])

    def _nd_to_flat(idx, shape):
        """Convert an N-D index to a row-major flat index."""
        flat = 0
        for k in range(len(shape)):
            flat = flat * shape[k] + idx[k]
        return flat

    # -- collect all input→output mappings ---------------------------------

    mappings = []
    num_out_dims = None
    for in_idx in itertools.product(*[range(d) for d in input_shape]):
        out_vals = layout.map_forward_index(list(in_idx))
        out_idx = tuple(int(v) for v in out_vals)
        if num_out_dims is None:
            num_out_dims = len(out_idx)
        mappings.append((tuple(in_idx), out_idx))

    # determine output shape from actual output indices
    output_shape = [0] * num_out_dims
    for _, out_idx in mappings:
        for k in range(num_out_dims):
            output_shape[k] = max(output_shape[k], out_idx[k] + 1)

    view_norm = (view or "output").strip().lower()

    if view_norm in ("output", "out", "dst"):
        out_rows, out_cols = _flatten_to_2d(output_shape)

        if verbose:
            print(f"Input shape : {input_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Grid size   : {out_rows} x {out_cols} (output view)")

        # -- build the output grid ---------------------------------------------

        grid_labels = [[None] * out_cols for _ in range(out_rows)]
        grid_src_flat = np.full((out_rows, out_cols), -1, dtype=int)

        for in_idx, out_idx in mappings:
            out_r, out_c = _nd_to_2d(out_idx, output_shape)
            src_flat = _nd_to_flat(in_idx, input_shape)

            grid_labels[out_r][out_c] = list(in_idx)
            grid_src_flat[out_r, out_c] = src_flat

            if verbose:
                print(f"  {list(in_idx)} -> {list(out_idx)} -> grid[{out_r}, {out_c}]")

        # -- plotting ----------------------------------------------------------

        cmap = plt.get_cmap(colormap, max(total_in, 2))

        # dynamic sizing
        max_dim = max(out_rows, out_cols, 1)
        cell_size = max(0.5, min(1.2, 16.0 / max_dim))

        fig_w = cell_size * out_cols + 1.5
        fig_h = cell_size * out_rows + 1.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # font size: adapt to cell size and longest label
        sample_label = "[" + ",".join(str(d - 1) for d in input_shape) + "]"
        max_label_len = len(sample_label)
        cell_pts = cell_size * 72  # approximate cell width in points
        base_font = max(5, min(16, cell_pts * 0.9 / max(max_label_len * 0.55, 1)))

        for i in range(out_rows):
            for j in range(out_cols):
                sf = grid_src_flat[i, j]
                if sf >= 0:
                    color = cmap(sf / max(total_in - 1, 1))
                else:
                    color = (0.95, 0.95, 0.95, 1.0)

                rect = patches.Rectangle(
                    (j, i),
                    1,
                    1,
                    linewidth=0.8,
                    edgecolor="#aaaaaa",
                    facecolor=color,
                )
                ax.add_patch(rect)

                coords = grid_labels[i][j]
                if coords is not None:
                    label = "[" + ",".join(str(x) for x in coords) + "]"
                    r, g, b = color[0], color[1], color[2]
                    brightness = r * 0.299 + g * 0.587 + b * 0.114
                    text_color = "white" if brightness < 0.5 else "black"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        label,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=base_font,
                        fontfamily="monospace",
                        fontweight="bold",
                    )

        # axis labels
        label_font = max(5, min(10, base_font * 0.85))
        # row labels on the left
        for i in range(out_rows):
            ax.text(-0.15, i + 0.5, str(i), ha="right", va="center", fontsize=label_font, color="#666666")
        # column labels at the bottom
        for j in range(out_cols):
            ax.text(j + 0.5, out_rows + 0.15, str(j), ha="center", va="top", fontsize=label_font, color="#666666")

        ax.set_xlim(-0.3, out_cols)
        ax.set_ylim(-0.1, out_rows + 0.5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # outer border
        border = patches.Rectangle(
            (0, 0),
            out_cols,
            out_rows,
            linewidth=1.5,
            edgecolor="#333333",
            facecolor="none",
        )
        ax.add_patch(border)

        # title: show shape transformation
        in_str = "x".join(str(d) for d in input_shape)
        out_str = "x".join(str(d) for d in output_shape)
        title_font = max(8, min(14, base_font * 1.1))
        ax.set_title(f"[{in_str}] -> [{out_str}]", fontsize=title_font, color="#333333", pad=8)

        plt.tight_layout()
        _save_plot(plt, save_directory, name, formats)
        plt.close()
        return

    if view_norm in ("input", "in", "src"):
        # Plot in input-space: keep the input arrangement, label by output.

        if grid_shape is not None:
            if len(grid_shape) != 2:
                raise ValueError(f"grid_shape must be a (rows, cols) tuple, got: {grid_shape}")
            in_rows, in_cols = int(grid_shape[0]), int(grid_shape[1])
            if in_rows <= 0 or in_cols <= 0:
                raise ValueError(f"grid_shape values must be positive, got: {grid_shape}")
            if in_rows * in_cols != total_in:
                raise ValueError(
                    f"grid_shape product must match total input elements ({total_in}), got {in_rows}x{in_cols}={in_rows * in_cols}."
                )
        else:
            in_rows, in_cols = _flatten_to_2d(input_shape)

        if verbose:
            print(f"Input shape : {input_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Grid size   : {in_rows} x {in_cols} (input view)")

        grid_out_flat = np.full((in_rows, in_cols), -1, dtype=int)
        grid_labels = [[None] * in_cols for _ in range(in_rows)]

        max_out_flat = -1
        for in_idx, out_idx in mappings:
            in_flat = _nd_to_flat(in_idx, input_shape)
            if grid_shape is None:
                in_r, in_c = _nd_to_2d(in_idx, input_shape)
            else:
                in_r, in_c = divmod(in_flat, in_cols)

            out_flat = _nd_to_flat(out_idx, output_shape)
            max_out_flat = max(max_out_flat, out_flat)

            grid_out_flat[in_r, in_c] = out_flat
            grid_labels[in_r][in_c] = str(out_flat)

            if verbose:
                print(f"  {list(in_idx)} -> {list(out_idx)} -> grid[{in_r}, {in_c}] = {out_flat}")

        cmap_n = max(max_out_flat + 1, 2)
        cmap = plt.get_cmap(colormap, cmap_n)

        max_dim = max(in_rows, in_cols, 1)
        cell_size = max(0.5, min(1.2, 16.0 / max_dim))

        fig_w = cell_size * in_cols + 1.5
        fig_h = cell_size * in_rows + 1.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sample_label = str(max_out_flat)
        max_label_len = len(sample_label)
        cell_pts = cell_size * 72  # approximate cell width in points
        base_font = max(5, min(16, cell_pts * 0.9 / max(max_label_len * 0.55, 1)))

        for i in range(in_rows):
            for j in range(in_cols):
                of = grid_out_flat[i, j]
                if of >= 0:
                    color = cmap(of / max(max_out_flat, 1))
                else:
                    color = (0.95, 0.95, 0.95, 1.0)

                rect = patches.Rectangle(
                    (j, i),
                    1,
                    1,
                    linewidth=0.8,
                    edgecolor="#aaaaaa",
                    facecolor=color,
                )
                ax.add_patch(rect)

                label = grid_labels[i][j]
                if label is not None:
                    r, g, b = color[0], color[1], color[2]
                    brightness = r * 0.299 + g * 0.587 + b * 0.114
                    text_color = "white" if brightness < 0.5 else "black"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        label,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=base_font,
                        fontfamily="monospace",
                        fontweight="bold",
                    )

        label_font = max(5, min(10, base_font * 0.85))
        for i in range(in_rows):
            ax.text(-0.15, i + 0.5, str(i), ha="right", va="center", fontsize=label_font, color="#666666")
        for j in range(in_cols):
            ax.text(j + 0.5, in_rows + 0.15, str(j), ha="center", va="top", fontsize=label_font, color="#666666")

        ax.set_xlim(-0.3, in_cols)
        ax.set_ylim(-0.1, in_rows + 0.5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        border = patches.Rectangle(
            (0, 0),
            in_cols,
            in_rows,
            linewidth=1.5,
            edgecolor="#333333",
            facecolor="none",
        )
        ax.add_patch(border)

        in_str = "x".join(str(d) for d in input_shape)
        out_str = "x".join(str(d) for d in output_shape)
        title_font = max(8, min(14, base_font * 1.1))
        ax.set_title(f"[{in_str}] -> [{out_str}] (input view)", fontsize=title_font, color="#333333", pad=8)

        plt.tight_layout()
        _save_plot(plt, save_directory, name, formats)
        plt.close()
        return

    raise ValueError(f"Unknown view '{view}'. Expected 'output' or 'input'.")


# ---------------------------------------------------------------------------
# Fragment thread-value (TV) view
# ---------------------------------------------------------------------------


def plot_fragment_tv(
    frag: T.Fragment,
    save_directory: str | None = None,
    name: str = "layout",
    apply_idx_fn=lambda *args: args,
    colormap: str = "RdPu",
    item_scale: float = 0.75,
    formats: str | list[str] = "pdf",
    dpi=80,
):
    """
    Plot fragment in terms of thread and local index mapping.
    Parameters
    ----------
    frag : T.Fragment
        The fragment object that describes how indices are mapped.
    save_directory : str | None, optional
        The directory where the output images will be saved.
    name : str, optional
        The base name of the output files (default is "layout").
    apply_idx_fn : function, optional
        A function to apply to the source indices for labeling (default is identity).
    colormap : str, optional
        The colormap to use for visualization (default is "RdPu").
    item_scale : float, optional
        The scale factor for each item in the plot (default is 0.75).
    formats : str | list[str], optional
        The formats to save the image in (default is "pdf").
    dpi : int, optional
        The resolution in dots per inch for the saved image (default is 80).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    src_shape = [i.value for i in frag.get_input_shape()]
    num_local_dim = frag.get_output_shape()[0].value
    num_thread_dim = frag.get_thread_size().value
    dst_shape = [num_local_dim, num_thread_dim]
    num_rep = frag.replicate_size.value
    src_flat_idx = np.zeros(dst_shape, dtype=np.int64)
    src_idx_str = np.full(dst_shape, "", dtype="<U32")
    if num_rep > 1:
        for rep in range(num_rep):
            for src_idx, item in enumerate(itertools.product(*([range(i) for i in src_shape]))):
                th = frag.map_forward_thread([rep] + list(item))[0].value
                dst_idx = frag.map_forward_index([rep] + list(item))[0].value
                src_flat_idx[dst_idx, th] = src_idx
                src_idx_str[dst_idx, th] = "(" + ",".join([str(i) for i in apply_idx_fn(*item)]) + ")"
    else:
        for src_idx, item in enumerate(itertools.product(*([range(i) for i in src_shape]))):
            th = frag.map_forward_thread(item)[0].value
            dst_idx = frag.map_forward_index(item)[0].value
            src_flat_idx[dst_idx, th] = src_idx
            src_idx_str[dst_idx, th] = "(" + ",".join([str(i) for i in apply_idx_fn(*item)]) + ")"

    plt.figure(figsize=(item_scale * num_thread_dim, item_scale * num_local_dim))
    cmap = plt.get_cmap(colormap)
    plt.pcolormesh(src_flat_idx, cmap=colormap, edgecolors="k", linewidth=0.5)
    mx = np.max(src_flat_idx) + 1
    for i in range(num_local_dim):
        for j in range(num_thread_dim):
            r, g, b, a = cmap(src_flat_idx[i, j] / mx)
            light_color = r + g + b < 1.5
            plt.text(j + 0.5, i + 0.5, src_idx_str[i, j], ha="center", va="center", color="white" if light_color else "black")
    plt.xticks(np.arange(num_thread_dim) + 0.5, [f"T{i}" for i in range(num_thread_dim)])
    plt.yticks(np.arange(num_local_dim) + 0.5, [f"I{i}" for i in range(num_local_dim)])
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.tight_layout()

    if save_directory is not None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(formats, str):
            formats = [formats]
        for fmt in formats:
            plt.savefig(save_dir / f"{name}.{fmt}", bbox_inches="tight", dpi=dpi)
        plt.close()
