from __future__ import annotations
import tilelang.language as T


def plot_layout(
    layout: T.Fragment,
    save_directory="./tmp",
    name: str = "layout",
    colormap: str = "RdPu",
    verbose: bool = False,
    formats: str | list[str] = "png",
) -> None:
    """
    Plot the layout of a buffer.

    Parameters
    ----------
    layout : T.Layout
        The layout object that describes how indices are mapped.
    save_directory : str, optional
        The directory where the output images will be saved (default is "./tmp").
    name : str, optional
        The base name of the output files (default is "layout").
    colormap : str, optional
        The colormap to use for visualization (default is "RdPu").
    verbose : bool, optional
        If True, prints additional information about the mapping (default is False).
    formats : str | list[str], optional
        The formats to save the image in (default is "png").
    Returns
    -------
    None
    """
    import os
    import pathlib
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Get the input shape of the layout and convert it to a list of integers
    input_shape = layout.get_input_shape()
    input_shape = [int(var) for var in input_shape]
    replicate_size = int(layout.replicate_size)

    # Get the total number of threads
    num_threads = int(layout.get_thread_size())

    import itertools

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

    # Create the output directory if it does not exist
    tmp_directory = pathlib.Path(save_directory)
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    # Save the figure in multiple formats
    plt.tight_layout()

    if isinstance(formats, str):
        formats_str = formats.strip().lower()
        if formats_str == "all":
            formats_list = ["pdf", "png", "svg"]
        elif "," in formats_str:
            formats_list = [f.strip() for f in formats_str.split(",")]
        else:
            formats_list = [formats_str]
    else:
        raise TypeError(
            f"Expected str, but got {type(formats).__name__}. Please pass a string like 'png', 'pdf', 'svg', 'all', or 'png,pdf'."
        )

    # Save the figure
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
