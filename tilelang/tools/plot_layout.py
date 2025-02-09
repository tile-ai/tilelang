# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang.language as T


def plot_layout(layout: T.Layout,
                save_directory="./tmp",
                name: str = "layout",
                colormap: str = "RdPu",
                verbose: bool = False) -> None:
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

    # Get the total number of threads
    num_threads = int(layout.get_thread_size())

    import itertools

    # Initialize a 2D array to store thread mappings
    thread_map = np.zeros(input_shape, dtype=int)

    # Iterate over all possible indices in the input shape
    for idx in itertools.product(*[range(dim) for dim in input_shape]):
        index = list(idx)
        # If replication is enabled, adjust the index
        if layout.replicate_size > 1:
            index.insert(0, 0)
        # Map the index to a thread ID
        thread_id = layout.map_forward_thread(index)
        assert len(thread_id) == 1  # Ensure a single-thread mapping
        thread_map[idx] = int(thread_id[0])  # Store the thread ID

    # Initialize a 2D array to store value mappings
    value_map = np.zeros(input_shape, dtype=int)

    # Iterate again to map values
    for idx in itertools.product(*[range(dim) for dim in input_shape]):
        index = list(idx)
        if layout.replicate_size > 1:
            index.insert(0, 0)
        thread_id = layout.map_forward_thread(index)
        value_id = layout.map_forward_index(index)
        assert len(value_id) == 1  # Ensure a single-value mapping
        value_map[idx] = int(value_id[0])  # Store the value ID

    # Load the colormap with twice as many colors as the number of threads
    cmap = plt.get_cmap(colormap, num_threads * 2)

    # Generate a list of colors based on the colormap
    raw_colors = [cmap(i) for i in range(num_threads)]
    colors = raw_colors.copy()

    # Determine the number of rows and columns in the input shape
    nrows, ncols = input_shape
    plt.figure(figsize=(nrows, ncols))  # Set the figure size
    ax = plt.gca()  # Get the current axis
    font_size = 24  # Set font size for text annotatio

    # Iterate through each row and column
    for i in range(nrows):
        for j in range(ncols):
            thread_id = thread_map[i, j]  # Get the thread ID
            local_id = value_map[i, j]  # Get the value ID
            if verbose:
                print(f"thread_map[{i}, {j}] = {thread_id} value_map[{i}, {j}] = {local_id}")

            color = colors[thread_id]  # Select color based on thread ID
            # Create a rectangle patch for visualization
            rect = patches.Rectangle((j, i),
                                     1,
                                     1,
                                     linewidth=0.5,
                                     edgecolor='black',
                                     facecolor=color)
            ax.add_patch(rect)  # Add the rectangle to the plot

            # Add text annotations inside the rectangles
            text = f"T{thread_id}\nL{local_id}"
            ax.text(
                j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=font_size)

    # Add row labels to the left side of the plot
    for i in range(nrows):
        text = f"row {i}"
        ax.text(-0.75, i + 0.5, text, ha='center', va='center', color='black', fontsize=font_size)

    # Add column labels at the top of the plot
    for j in range(ncols):
        text = f"col {j}"
        ax.text(
            j + 0.5,
            -0.5,
            text,
            ha='center',
            va='center',
            color='black',
            fontsize=font_size,
            rotation=45)

    # Set the plot limits
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()  # Invert the y-axis for proper visualization
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks

    legend_patches = [
        patches.Patch(color='black', label="T: Thread ID"),
        patches.Patch(color='black', label="L: Local ID")
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=font_size - 4,
        frameon=False,
        bbox_to_anchor=(1.0, 1.12),
        ncols=2)

    # Create the output directory if it does not exist
    tmp_directory = pathlib.Path(save_directory)
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    # Save the figure in multiple formats
    plt.tight_layout()

    # Save as PDF
    pdf_path = tmp_directory / f"{name}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")

    # Save as PNG
    png_path = tmp_directory / f"{name}.png"
    plt.savefig(png_path, bbox_inches="tight", transparent=False, dpi=255)

    # Save as SVG
    svg_path = tmp_directory / f"{name}.svg"
    plt.savefig(svg_path, bbox_inches="tight", format="svg")
