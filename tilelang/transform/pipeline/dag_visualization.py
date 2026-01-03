"""
DAG Visualization utilities for Pipeline Planning.

This module provides functions to visualize pipeline dependency DAGs
in various formats: DOT (Graphviz), ASCII art, and Mermaid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_planing import PipelineStageInfo


def dag_to_dot(stage_infos: list[PipelineStageInfo], title: str = "Pipeline DAG") -> str:
    """
    Generate DOT format string for the dependency DAG.

    Parameters
    ----------
    stage_infos : List[PipelineStageInfo]
        List of pipeline stage info objects with DAG fields populated.
    title : str
        Title for the graph.

    Returns
    -------
    str
        DOT format string that can be rendered by graphviz.

    Example
    -------
    >>> print(dag_to_dot(stage_infos))
    digraph PipelineDAG {
      rankdir=TB;
      label="Pipeline DAG";
      0 [label="S0 [copy]" style=filled fillcolor=lightblue];
      1 [label="S1 [copy]" style=filled fillcolor=lightblue];
      2 [label="S2"];
      0 -> 2 [label="RAW"];
      1 -> 2 [label="RAW"];
    }
    """
    lines = [
        "digraph PipelineDAG {",
        "  rankdir=TB;",
        f'  label="{title}";',
        "  labelloc=t;",
        "  node [shape=box];",
    ]

    # Add nodes
    for info in stage_infos:
        idx = info.original_stmt_index
        label = info.get_label()

        # Style based on stage type
        if info.copy_stage:
            style = "style=filled fillcolor=lightblue"
        elif info.producer_for_copy:
            style = "style=filled fillcolor=lightyellow"
        else:
            style = "style=filled fillcolor=white"

        lines.append(f'  {idx} [label="{label}" {style}];')

    # Add edges with dependency type labels
    added_edges: set[tuple[int, int]] = set()
    for info in stage_infos:
        for dep in info.dependencies:
            edge_key = (dep.from_stage, dep.to_stage)
            if edge_key not in added_edges:
                # Collect all dependency types for this edge
                dep_types = [d.dep_type.name for d in info.dependencies if d.from_stage == dep.from_stage and d.to_stage == dep.to_stage]
                label = ",".join(sorted(set(dep_types)))
                lines.append(f'  {dep.from_stage} -> {dep.to_stage} [label="{label}"];')
                added_edges.add(edge_key)

    lines.append("}")
    return "\n".join(lines)


def dag_to_ascii(stage_infos: list[PipelineStageInfo], style: str = "vertical") -> str:
    """
    Generate ASCII representation of the dependency DAG.

    Parameters
    ----------
    stage_infos : List[PipelineStageInfo]
        List of pipeline stage info objects with DAG fields populated.
    style : str
        "vertical" for top-down graph, "list" for simple list format.

    Returns
    -------
    str
        ASCII art representation of the DAG.
    """
    if style == "list":
        return _dag_to_ascii_list(stage_infos)
    return _dag_to_ascii_vertical(stage_infos)


def _dag_to_ascii_list(stage_infos: list[PipelineStageInfo]) -> str:
    """Simple list-style ASCII output."""
    lines = ["Pipeline Dependency DAG", "=" * 40]

    for info in stage_infos:
        label = info.get_label()
        pred_str = ""
        if info.predecessors:
            pred_list = sorted(info.predecessors)
            pred_str = f"  <- [S{', S'.join(map(str, pred_list))}]"
        lines.append(f"  {label}{pred_str}")

        for succ_idx in sorted(info.successors):
            succ_info = stage_infos[succ_idx]
            deps_to_succ = [d for d in succ_info.dependencies if d.from_stage == info.original_stmt_index]
            if deps_to_succ:
                # Deduplicate dependencies
                seen = set()
                unique_deps = []
                for d in deps_to_succ:
                    key = (d.dep_type.name, d.buffer_name)
                    if key not in seen:
                        seen.add(key)
                        unique_deps.append(f"{d.dep_type.name}:{d.buffer_name}")
                dep_details = ", ".join(unique_deps)
                lines.append(f"    └─> S{succ_idx} ({dep_details})")
            else:
                lines.append(f"    └─> S{succ_idx}")

    return "\n".join(lines)


def _dag_to_ascii_vertical(stage_infos: list[PipelineStageInfo]) -> str:
    """
    Generate a vertical top-down ASCII DAG visualization.

    Example output:
    ┌──────────────────────────────────────┐
    │       Pipeline Dependency DAG        │
    └──────────────────────────────────────┘

        ┌──────────┐          ┌──────────┐
        │ S0 copy  │          │ S1 copy  │
        └────┬─────┘          └────┬─────┘
             │   A_shared          │   B_shared
             └─────────┬───────────┘
                       ▼
                 ┌──────────┐
                 │    S2    │
                 └──────────┘
    """
    if not stage_infos:
        return "Empty DAG"

    # Group stages by their "level" (topological order based on dependencies)
    levels: dict[int, list[int]] = {}
    stage_level: dict[int, int] = {}

    # Compute levels using BFS from sources
    for info in stage_infos:
        idx = info.original_stmt_index
        if not info.predecessors:
            stage_level[idx] = 0
        else:
            max_pred_level = max(stage_level.get(p, 0) for p in info.predecessors)
            stage_level[idx] = max_pred_level + 1

    for idx, level in stage_level.items():
        if level not in levels:
            levels[level] = []
        levels[level].append(idx)

    # Sort stages within each level
    for level in levels:
        levels[level].sort()

    # Build the visualization
    lines = []
    box_width = 12

    # Title
    title = "Pipeline Dependency DAG"
    title_width = max(50, len(stage_infos) * (box_width + 6))
    lines.append("┌" + "─" * title_width + "┐")
    lines.append("│" + title.center(title_width) + "│")
    lines.append("└" + "─" * title_width + "┘")
    lines.append("")

    max_level = max(levels.keys()) if levels else 0
    total_width = title_width + 2

    # Store positions for each stage for drawing connections
    stage_positions: dict[int, int] = {}

    for level in range(max_level + 1):
        stage_indices = levels.get(level, [])
        num_stages = len(stage_indices)

        if num_stages == 0:
            continue

        # Calculate spacing
        stage_spacing = total_width // (num_stages + 1)

        # Calculate positions for this level
        positions = []
        for i in range(num_stages):
            pos = stage_spacing * (i + 1)
            positions.append(pos)
            stage_positions[stage_indices[i]] = pos

        # Build box lines
        box_top = [" "] * total_width
        box_mid = [" "] * total_width
        box_bot = [" "] * total_width

        for i, stage_idx in enumerate(stage_indices):
            info = stage_infos[stage_idx]
            label = f"S{stage_idx}"
            if info.copy_stage:
                label += " copy"
            elif info.producer_for_copy:
                label += " prod"

            label = label[: box_width - 2].center(box_width - 2)
            pos = positions[i]
            start = pos - box_width // 2

            # Draw box
            box_top[start] = "┌"
            for j in range(1, box_width - 1):
                box_top[start + j] = "─"
            box_top[start + box_width - 1] = "┐"

            box_mid[start] = "│"
            for j, c in enumerate(label):
                box_mid[start + 1 + j] = c
            box_mid[start + box_width - 1] = "│"

            box_bot[start] = "└"
            for j in range(1, box_width - 1):
                box_bot[start + j] = "─"
            box_bot[start + box_width - 1] = "┘"

        lines.append("".join(box_top))
        lines.append("".join(box_mid))
        lines.append("".join(box_bot))

        # Draw arrows to next level if not last level
        if level < max_level:
            next_stages = levels.get(level + 1, [])
            if next_stages:
                # Collect connections: from current level to next level
                connections = []
                for stage_idx in stage_indices:
                    info = stage_infos[stage_idx]
                    for succ_idx in info.successors:
                        if succ_idx in next_stages:
                            # Get dependency info
                            succ_info = stage_infos[succ_idx]
                            deps = [d for d in succ_info.dependencies if d.from_stage == stage_idx]
                            # Deduplicate
                            seen = set()
                            dep_names = []
                            for d in deps:
                                if d.buffer_name not in seen:
                                    seen.add(d.buffer_name)
                                    dep_names.append(d.buffer_name)
                            connections.append((stage_idx, succ_idx, dep_names))

                # Draw vertical lines with labels
                line1 = [" "] * total_width
                for stage_idx in stage_indices:
                    info = stage_infos[stage_idx]
                    if any(s in next_stages for s in info.successors):
                        pos = stage_positions[stage_idx]
                        line1[pos] = "│"

                lines.append("".join(line1))

                # Draw dependency labels next to lines
                label_line = [" "] * total_width
                for stage_idx, _succ_idx, dep_names in connections:
                    if dep_names:
                        pos = stage_positions[stage_idx]
                        label = dep_names[0][:10]  # Truncate long names
                        start = pos + 2
                        for j, c in enumerate(label):
                            if start + j < total_width:
                                label_line[start + j] = c

                lines.append("".join(label_line))

                # Draw horizontal merge line
                merge_line = [" "] * total_width
                for succ_idx in next_stages:
                    # Find all sources for this target
                    source_positions = []
                    for stage_idx in stage_indices:
                        info = stage_infos[stage_idx]
                        if succ_idx in info.successors:
                            source_positions.append(stage_positions[stage_idx])

                    if source_positions:
                        min_pos = min(source_positions)
                        max_pos = max(source_positions)

                        # Draw horizontal line from sources to target
                        for p in range(min_pos, max_pos + 1):
                            if merge_line[p] == " ":
                                merge_line[p] = "─"

                        # Draw corners
                        for sp in source_positions:
                            if sp == min_pos:
                                merge_line[sp] = "└"
                            elif sp == max_pos:
                                merge_line[sp] = "┘"
                            else:
                                merge_line[sp] = "┴"

                        # Draw down arrow point
                        center = (min_pos + max_pos) // 2
                        merge_line[center] = "┬"

                lines.append("".join(merge_line))

                # Draw arrow to target
                arrow_line = [" "] * total_width
                for succ_idx in next_stages:
                    # Find center of sources
                    source_positions = []
                    for stage_idx in stage_indices:
                        info = stage_infos[stage_idx]
                        if succ_idx in info.successors:
                            source_positions.append(stage_positions[stage_idx])
                    if source_positions:
                        center = (min(source_positions) + max(source_positions)) // 2
                        arrow_line[center] = "▼"

                lines.append("".join(arrow_line))
                lines.append("")

    return "\n".join(lines)


def dag_to_mermaid(stage_infos: list[PipelineStageInfo]) -> str:
    """
    Generate Mermaid format string for the dependency DAG.

    Mermaid is supported by GitHub markdown and many documentation tools.

    Parameters
    ----------
    stage_infos : List[PipelineStageInfo]
        List of pipeline stage info objects with DAG fields populated.

    Returns
    -------
    str
        Mermaid format string.

    Example
    -------
    >>> print(dag_to_mermaid(stage_infos))
    ```mermaid
    graph TD
        S0[S0 copy]:::copy --> S2
        S1[S1 copy]:::copy --> S2
        classDef copy fill:#add8e6
    ```
    """
    lines = ["```mermaid", "graph TD"]

    # Define nodes
    for info in stage_infos:
        idx = info.original_stmt_index
        label = f"S{idx}"
        if info.copy_stage:
            label += " copy"
            lines.append(f"    S{idx}[{label}]:::copy")
        elif info.producer_for_copy:
            label += " prod"
            lines.append(f"    S{idx}[{label}]:::prod")
        else:
            lines.append(f"    S{idx}[{label}]")

    # Define edges
    added_edges: set[tuple[int, int]] = set()
    for info in stage_infos:
        for succ_idx in info.successors:
            edge_key = (info.original_stmt_index, succ_idx)
            if edge_key not in added_edges:
                lines.append(f"    S{info.original_stmt_index} --> S{succ_idx}")
                added_edges.add(edge_key)

    # Add styles
    lines.append("    classDef copy fill:#add8e6")
    lines.append("    classDef prod fill:#fffacd")
    lines.append("```")

    return "\n".join(lines)
