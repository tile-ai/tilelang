"""Z3-based scheduler for auto-scheduling.

This module provides a Python implementation of the Z3 scheduler that can be
called from C++ via TVM FFI.
"""

import tvm_ffi
import os
import json
import time
from pathlib import Path

# Try to import z3, but handle missing installation gracefully
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("[Python Z3] WARNING: z3-solver package not installed. Z3 scheduling will not work.")


def _find_next_schedule_number(base_dir="test"):
    """Find the smallest available number for schedule_xxx directory."""
    if not os.path.exists(base_dir):
        return 0

    existing_numbers = []
    for item in os.listdir(base_dir):
        if item.startswith("schedule_") and os.path.isdir(os.path.join(base_dir, item)):
            try:
                num = int(item[9:])  # Extract number from "schedule_xxx"
                existing_numbers.append(num)
            except ValueError:
                continue

    if not existing_numbers:
        return 0

    existing_numbers.sort()
    # Find the first gap in the sequence
    for i, num in enumerate(existing_numbers):
        if i != num:
            return i

    return len(existing_numbers)


def z3_schedule_python(
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int]],
    resource_deps: list[tuple[int, int]],
    verbose: bool = False,
) -> tuple[list[int], list[int]]:
    """Z3-based scheduler implemented in Python.

    Parameters
    ----------
    latencies : list[int]
        Latency for each task in cycles
    iis : list[int]
        Initiation interval for each task in cycles
    resource_flags : list[int]
        Resource usage flags for each task (bitmask):
        1 = uses CUDA core, 2 = uses TMA core, 4 = uses Tensor core
    data_deps : list[tuple[int, int]]
        Data dependency pairs (i, j) where task j depends on task i
    resource_deps : list[tuple[int, int]]
        Resource dependency pairs (i, j) where tasks i and j use same resource

    Returns
    -------
    tuple[list[int], list[int]]
        start_times: Start time for each task
        sorted_indices: Task indices sorted by start time
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n <= 1:
        if n == 1:
            return [0], [0]
        return [], []

    if verbose:
        print(f"[Python Z3] Starting scheduling for {n} tasks")
        print(f"[Python Z3] Latencies: {latencies}")
        print(f"[Python Z3] IIs: {iis}")
        print(f"[Python Z3] Resource flags: {resource_flags}")
        print(f"[Python Z3] Data dependencies: {data_deps}")
        print(f"[Python Z3] Resource dependencies: {resource_deps}")

    # Check if z3 is available
    if not Z3_AVAILABLE:
        print("[Python Z3] WARNING: z3-solver not available, using fallback scheduler")
        return fallback_schedule(latencies, data_deps, resource_deps)

    try:
        # Create Z3 solver
        solver = z3.Optimize()
        solver.set("timeout", 10000)

        # Create start time variables
        start_vars = [z3.Int(f"start_{i}") for i in range(n)]

        # Add constraints: start times must be non-negative
        for var in start_vars:
            solver.add(var >= 0)

        # Add data dependency constraints
        # For each data dependency (i, j), task j must start after task i completes
        for i, j in data_deps:
            if i < j:  # Ensure i < j as in original order
                latency_i = latencies[i]
                solver.add(start_vars[j] >= start_vars[i] + latency_i)
                if verbose:
                    print(f"[Python Z3] Data dependency: task {j} >= task {i} + {latency_i}")

        # Add resource dependency constraints
        # For tasks i and j that use same resource, they cannot execute simultaneously
        # We create ordering variable O_i,j (True means i before j, False means j before i)
        for i, j in resource_deps:
            if i < j:  # Only consider each pair once
                ii_i = iis[i]
                ii_j = iis[j]

                # Create ordering variable
                o_ij = z3.Bool(f"O_{i}_{j}")

                # If o_ij is True (i before j), then start_j >= start_i + ii_i
                solver.add(z3.Implies(o_ij, start_vars[j] >= start_vars[i] + ii_i))

                # If o_ij is False (j before i), then start_i >= start_j + ii_j
                solver.add(z3.Implies(z3.Not(o_ij), start_vars[i] >= start_vars[j] + ii_j))

                if verbose:
                    print(f"[Python Z3] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

        # Objective: minimize maximum completion time (makespan)
        makespan = z3.Int("makespan")
        for i in range(n):
            latency_i = latencies[i]
            solver.add(makespan >= start_vars[i] + latency_i)
        solver.add(makespan >= 0)

        # Minimize makespan
        solver.minimize(makespan)

        # Check satisfiability
        if verbose:
            print("[Python Z3] Checking satisfiability...")
        if solver.check() == z3.sat:
            model = solver.model()

            # Extract start times
            start_times = []
            for i in range(n):
                start_time = model.eval(start_vars[i]).as_long()
                start_times.append(start_time)

            # Get makespan
            makespan_val = model.eval(makespan).as_long()

            # Sort tasks by start time (and by index as tie-breaker)
            task_indices = list(range(n))
            task_indices.sort(key=lambda idx: (start_times[idx], idx))

            if verbose:
                print(f"[Python Z3] Scheduling completed. Makespan = {makespan_val}")
            for i in range(n):
                idx = task_indices[i]
                if verbose:
                    print(
                        f"[Python Z3]   Task {idx}: start_time={start_times[idx]}, "
                        f"latency={latencies[idx]}, II={iis[idx]}, "
                        f"resource_flags={resource_flags[idx]:03b}"
                    )

            return start_times, task_indices
        else:
            if verbose:
                print("[Python Z3] Z3 scheduling failed, falling back to topological sort")
            # Fallback: return tasks in original order with start times based on dependencies
            return fallback_schedule(latencies, data_deps, resource_deps)

    except Exception as e:
        print(f"[Python Z3] Error in Z3 scheduling: {e}")
        # Fallback to simple schedule
        return fallback_schedule(latencies, data_deps, resource_deps)


def fallback_schedule(
    latencies: list[int], data_deps: list[tuple[int, int]], resource_deps: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    """Fallback schedule when Z3 solver fails.

    Simple topological sort based on data dependencies.
    """
    n = len(latencies)
    if n == 0:
        return [], []

    # Build adjacency list for data dependencies
    adj = [[] for _ in range(n)]
    indegree = [0] * n

    for i, j in data_deps:
        if i < j:  # Ensure i < j as in original order
            adj[i].append(j)
            indegree[j] += 1

    # Kahn's algorithm for topological sort
    result = []
    start_times = [0] * n

    # Initialize queue with tasks having indegree 0
    queue = [i for i in range(n) if indegree[i] == 0]

    while queue:
        # Always take the smallest index to maintain some determinism
        i = min(queue)
        queue.remove(i)
        result.append(i)

        # Update start times for successors
        for j in adj[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)
            # Update start time: max(current, completion time of predecessor)
            start_times[j] = max(start_times[j], start_times[i] + latencies[i])

    # If there's a cycle (not all tasks processed), return original order
    if len(result) != n:
        result = list(range(n))
        # Simple linear schedule
        for i in range(1, n):
            start_times[i] = start_times[i - 1] + latencies[i - 1]

    return start_times, result


# FFI-exposed function that matches C++ interface
@tvm_ffi.register_global_func("tl.transform.z3_schedule_python")
def z3_schedule_ffi(latencies, iis, resource_flags, data_deps, resource_deps):
    """FFI wrapper for z3_schedule_python.

    This function accepts TVM containers and converts them to Python lists.
    """
    # Convert TVM containers to Python lists
    latencies_list = list(latencies)
    iis_list = list(iis)
    resource_flags_list = list(resource_flags)

    # Convert data dependencies
    data_deps_list = []
    if data_deps is not None:
        # Assuming data_deps is a list of pairs
        for i in range(len(data_deps)):
            if hasattr(data_deps[i], "__len__") and len(data_deps[i]) == 2:
                data_deps_list.append((int(data_deps[i][0]), int(data_deps[i][1])))

    # Convert resource dependencies
    resource_deps_list = []
    if resource_deps is not None:
        for i in range(len(resource_deps)):
            if hasattr(resource_deps[i], "__len__") and len(resource_deps[i]) == 2:
                resource_deps_list.append((int(resource_deps[i][0]), int(resource_deps[i][1])))

    # Call the actual scheduler
    start_times, _ = z3_schedule_python(latencies_list, iis_list, resource_flags_list, data_deps_list, resource_deps_list)

    # Return only start_times, C++ side will sort by start_time
    return start_times


def z3_schedule_loop_python(
    num_stages: int,
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int, int]],  # (i, j, distance)
    resource_deps: list[tuple[int, int]],
    verbose: bool = False,
) -> tuple[list[int], list[int], int]:
    """Z3-based scheduler for loops with distance-aware dependencies.

    New modeling:
    - Data dependency: start_v - start_u >= latency_u - II * distance
    - Resource dependency: start_i = k_i * II + r_i, where 0 <= r_i < II
      delta_i,j: boolean variable for modulo ordering
      Constraints: r_i - r_j + II * delta_i,j >= ii_i
                   r_i - r_j + II * (1 - delta_i,j) >= ii_j
    - Objective: minimize II using binary search

    Parameters
    ----------
    latencies : list[int]
        Latency for each task in cycles
    iis : list[int]
        Initiation interval for each task in cycles
    resource_flags : list[int]
        Resource usage flags for each task (bitmask):
        1 = uses CUDA core, 2 = uses TMA core, 4 = uses Tensor core
    data_deps : list[tuple[int, int, int]]
        Data dependency tuples (i, j, distance) where task j depends on task i
        with distance d (loop iterations distance)
    resource_deps : list[tuple[int, int]]
        Resource dependency pairs (i, j) where tasks i and j use same resource

    Returns
    -------
    tuple[list[int], list[int]]
        start_times: Start time for each task
        sorted_indices: Task indices sorted by start time
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n <= 1:
        if n == 1:
            return [0], [0]
        return [], []

    if verbose:
        print(f"[Python Z3 Loop] Starting scheduling for {n} tasks")
        print(f"[Python Z3 Loop] Latencies: {latencies}")
        print(f"[Python Z3 Loop] IIs: {iis}")
        print(f"[Python Z3 Loop] Resource flags: {resource_flags}")
        print(f"[Python Z3 Loop] Data dependencies with distances: {data_deps}")
        print(f"[Python Z3 Loop] Resource dependencies: {resource_deps}")

    # Check if z3 is available
    if not Z3_AVAILABLE:
        if verbose:
            print("[Python Z3 Loop] WARNING: z3-solver not available, using fallback scheduler")
        # Convert data_deps to pairs for fallback (ignore distance)
        data_deps_pairs = [(i, j) for i, j, d in data_deps]
        return fallback_schedule(latencies, data_deps_pairs, resource_deps)

    try:
        # Binary search for minimal II
        # Lower bound: 1 cycle
        # Upper bound: sum of all latencies (conservative)
        ii_lower = 1
        ii_upper = sum(latencies) + 1
        best_ii = ii_upper
        best_model = None
        best_start_vars = None

        if verbose:
            print(f"[Python Z3 Loop] Binary search range: [{ii_lower}, {ii_upper})")

        while ii_lower < ii_upper:
            ii_mid = (ii_lower + ii_upper) // 2
            if verbose:
                print(f"[Python Z3 Loop] Testing II = {ii_mid}")

            # Create solver for feasibility check
            solver = z3.Solver()
            solver.set("timeout", 10000)

            start_vars = [z3.Int(f"start_{i}") for i in range(n)]
            stage_vars = [z3.Int(f"pro_{i}") for i in range(n)]
            begin = z3.Int("begin")

            for i in range(n):
                solver.add(start_vars[i] >= 0)
                solver.add(stage_vars[i] >= 0)
                solver.add(stage_vars[i] < num_stages * 2)
                solver.add(begin <= start_vars[i] + stage_vars[i] * ii_mid)
                ii_i = iis[i]
                solver.add(start_vars[i] + ii_i + stage_vars[i] * ii_mid - begin <= ii_mid)

            # Add data dependency constraints with distance
            for u, v, distance in data_deps:
                latency_u = latencies[u]
                if verbose:
                    print(f"[Python Z3 Loop] Data dependency: task {v} - task {u} >= {latency_u} - {ii_mid}*{distance}")
                solver.add(start_vars[v] - start_vars[u] >= latency_u - ii_mid * distance)

            # Add resource dependency constraints
            # For tasks i and j that use same resource, they cannot execute simultaneously
            # We create ordering variable O_i,j (True means i before j, False means j before i)
            for i, j in resource_deps:
                if i < j:  # Only consider each pair once
                    ii_i = iis[i]
                    ii_j = iis[j]

                    # Create ordering variable
                    o_ij = z3.Bool(f"O_{i}_{j}")

                    # If o_ij is True (i before j), then start_j >= start_i + ii_i
                    solver.add(z3.Implies(o_ij, start_vars[j] >= start_vars[i] + ii_i))

                    # If o_ij is False (j before i), then start_i >= start_j + ii_j
                    solver.add(z3.Implies(z3.Not(o_ij), start_vars[i] >= start_vars[j] + ii_j))

                    if verbose:
                        print(f"[Python Z3 Loop] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

            # Check feasibility
            if verbose:
                print(f"[Python Z3 Loop] Checking feasibility for II = {ii_mid}...")
            if solver.check() == z3.sat:
                if verbose:
                    print(f"[Python Z3 Loop] II = {ii_mid} is feasible")
                best_ii = ii_mid
                best_model = solver.model()
                best_start_vars = start_vars
                best_stage_vars = stage_vars
                # Try smaller II
                ii_upper = ii_mid
            else:
                if verbose:
                    print(f"[Python Z3 Loop] II = {ii_mid} is infeasible")
                # Need larger II
                ii_lower = ii_mid + 1

        if best_model is None:
            if verbose:
                print("[Python Z3 Loop] No feasible II found in range, falling back to topological sort")
            data_deps_pairs = [(i, j) for i, j, d in data_deps]
            return fallback_schedule(latencies, data_deps_pairs, resource_deps), sum(latencies) + 1

        if verbose:
            print(f"[Python Z3 Loop] Minimal feasible II = {best_ii}")

        # Extract start times from best model
        start_times = []
        stages = []
        begin = best_model.eval(begin).as_long()
        for i in range(n):
            start_time = best_model.eval(best_start_vars[i]).as_long()
            stage = best_model.eval(best_stage_vars[i]).as_long()
            start_times.append(start_time)
            stages.append(stage)

        # Sort tasks by start time (and by index as tie-breaker)
        task_indices = list(range(n))
        task_indices.sort(key=lambda idx: (start_times[idx] + stages[idx] * best_ii, idx))

        if verbose:
            print(f"[Python Z3 Loop] Scheduling completed. Minimal II = {best_ii}. Begin = {begin}")
        for i in range(n):
            idx = task_indices[i]
            if verbose:
                print(
                    f"[Python Z3 Loop]   Task {idx}: start_time={start_times[idx]}, "
                    f"latency={latencies[idx]}, II={iis[idx]}, stage={stages[idx]}, "
                    f"resource_flags={resource_flags[idx]:03b}"
                )

        # Save schedule visualization when verbose is True
        if verbose:
            try:
                # Find the next available schedule number
                schedule_num = _find_next_schedule_number()
                schedule_dir = Path(f"test/schedule_{schedule_num:03d}")
                schedule_dir.mkdir(parents=True, exist_ok=True)

                # Save schedule information
                schedule_info = {
                    "num_stages": num_stages,
                    "num_tasks": n,
                    "minimal_II": best_ii,
                    "latencies": latencies,
                    "iis": iis,
                    "resource_flags": resource_flags,
                    "data_dependencies": data_deps,
                    "resource_dependencies": resource_deps,
                    "start_times": start_times,
                    "stages": stages,
                    "sorted_indices": task_indices,
                    "timestamp": time.time(),
                }

                # Save as JSON
                info_file = schedule_dir / "schedule_info.json"
                with open(info_file, "w") as f:
                    json.dump(schedule_info, f, indent=2)

                # Create a simple text visualization
                viz_file = schedule_dir / "schedule_visualization.txt"
                with open(viz_file, "w") as f:
                    f.write(f"Z3 Schedule Visualization (II={best_ii})\n")
                    f.write("=" * 50 + "\n\n")

                    f.write("Task Information:\n")
                    f.write("-" * 30 + "\n")
                    for i in range(n):
                        idx = task_indices[i]
                        f.write(
                            f"Task {idx:3d}: start={start_times[idx]:3d}, "
                            f"latency={latencies[idx]:3d}, II={iis[idx]:3d}, "
                            f"stage={stages[idx]:2d}, "
                            f"resources={resource_flags[idx]:03b}\n"
                        )

                    f.write("\nData Dependencies:\n")
                    f.write("-" * 30 + "\n")
                    for u, v, distance in data_deps:
                        f.write(f"  Task {v} depends on Task {u} (distance={distance})\n")

                    f.write("\nResource Dependencies:\n")
                    f.write("-" * 30 + "\n")
                    for i, j in resource_deps:
                        f.write(f"  Task {i} and Task {j} share resources\n")

                    f.write("\nDetailed Schedule Timeline:\n")
                    f.write("-" * 50 + "\n")
                    f.write("Phase Legend:\n")
                    f.write("  # = Initial Phase [start, start+II] (first II cycles)\n")
                    f.write("  = = Remaining Phase [start+II, end] (rest of execution)\n")
                    f.write("  | = II boundary between phases\n")
                    f.write("  [ = Task start\n")
                    f.write("  ] = Task end\n")
                    f.write("  B = Loop Begin (earliest task start)\n")
                    f.write("  E = Loop End (Begin + Minimal II)\n")
                    f.write("-" * 50 + "\n")

                    # Find max time for timeline using actual start times
                    actual_start_times = [start_times[i] + best_ii * stages[i] for i in range(n)]
                    max_time = max(actual_start_times[i] + latencies[i] for i in range(n))
                    timeline_end = max_time + 5

                    loop_end = begin + best_ii

                    # Create timeline header
                    f.write("Time: ")
                    for t in range(0, timeline_end, 10):
                        f.write(f"{t:10d}")
                    f.write("\n")
                    f.write("      ")
                    for _ in range(0, timeline_end, 10):
                        f.write("|         ")
                    f.write("\n")

                    # Create detailed timeline for each task
                    for idx in task_indices:
                        # Get task info
                        # Calculate actual start time: start_times[idx] + best_ii * stages[idx]
                        start = start_times[idx] + best_ii * stages[idx]
                        end = start + latencies[idx]
                        ii = iis[idx]
                        stage = stages[idx]
                        resource = resource_flags[idx]

                        # Calculate ii boundary
                        ii_boundary = start + ii if start + ii < end else end

                        # Determine resource type symbol
                        if resource & 4:  # Tensor core
                            symbol = "T"
                        elif resource & 2:  # TMA core
                            symbol = "M"
                        elif resource & 1:  # CUDA core
                            symbol = "C"
                        else:
                            symbol = "?"

                        # Create timeline row for this task
                        f.write(f"Task {idx:2d} ({symbol}): ")

                        for t in range(timeline_end):
                            if t == begin:
                                f.write("B")  # Loop Begin
                            elif t == loop_end:
                                f.write("E")  # Loop End (Begin + best_ii)
                            elif t == start:
                                f.write("[")  # Start of task
                            elif t == ii_boundary - 1 and ii_boundary < end:
                                f.write("|")  # II boundary (end of initial phase)
                            elif t == end - 1:
                                f.write("]")  # End of task
                            elif start < t < ii_boundary:
                                f.write("#")  # Initial phase [start, start+ii]
                            elif ii_boundary <= t < end - 1:
                                f.write("=")  # Remaining Phase [start+ii, end]
                            elif t == 0 or t % 10 == 0:
                                f.write("|")  # Time marker
                            else:
                                f.write(" ")  # Empty space

                        # Add phase information
                        phase1_len = ii_boundary - start
                        phase2_len = end - ii_boundary
                        f.write(
                            f"  (start={start}, II={ii}, phase1[{phase1_len}]=[start,start+II], phase2[{phase2_len}]=[start+II,end], stage={stage})\n"
                        )

                    f.write("\nASCII Gantt Chart:\n")
                    f.write("-" * 40 + "\n")

                    # Create a more compact ASCII Gantt chart
                    # Group tasks by resource type for better visualization
                    resource_groups = {
                        "CUDA": [i for i in range(n) if resource_flags[i] & 1],
                        "TMA": [i for i in range(n) if resource_flags[i] & 2],
                        "Tensor": [i for i in range(n) if resource_flags[i] & 4],
                    }

                    for resource_name, task_list in resource_groups.items():
                        if not task_list:
                            continue

                        f.write(f"\n{resource_name} Core Tasks:\n")

                        # Create timeline for this resource group
                        time_scale = 2  # Each character represents 2 time units
                        scaled_end = (timeline_end + time_scale - 1) // time_scale

                        # Time axis
                        f.write("Time: ")
                        for t_scaled in range(0, scaled_end, 5):
                            t_actual = t_scaled * time_scale
                            f.write(f"{t_actual:5d}")
                        f.write("\n")
                        f.write("      ")
                        for _ in range(0, scaled_end, 5):
                            f.write("|    ")
                        f.write("\n")

                        # Task rows
                        for idx in sorted(task_list, key=lambda i: start_times[i]):
                            start = start_times[idx]
                            end = start + latencies[idx]
                            ii = iis[idx]
                            start_scaled = start // time_scale
                            end_scaled = (end + time_scale - 1) // time_scale
                            ii_boundary_scaled = (start + ii) // time_scale if start + ii < end else end_scaled

                            f.write(f"Task {idx:2d}: ")

                            for t_scaled in range(scaled_end):
                                if start_scaled <= t_scaled < end_scaled:
                                    if t_scaled == start_scaled:
                                        f.write("[")
                                    elif t_scaled == ii_boundary_scaled - 1 and ii_boundary_scaled < end_scaled:
                                        f.write("|")  # II boundary
                                    elif t_scaled == end_scaled - 1:
                                        f.write("]")
                                    elif start_scaled <= t_scaled < ii_boundary_scaled:
                                        f.write("#")  # Initial phase
                                    else:
                                        f.write("=")  # Remaining phase
                                else:
                                    f.write(" ")
                            f.write(f"  ({start}-{end}, ii={ii})\n")

                # Directly generate matplotlib visualization (PNG and PDF)
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches
                    import colorsys
                    from matplotlib.patches import Patch

                    fig, ax = plt.subplots(figsize=(16, 10))
                    # Set background color to light gray
                    fig.patch.set_facecolor("#f0f0f0")
                    ax.set_facecolor("#888888")

                    # Create color map for resources
                    resource_colors = {
                        1: "green",  # CUDA core
                        2: "red",  # TMA core
                        4: "gold",  # Tensor core
                        3: "orange",  # CUDA + TMA
                        5: "lime",  # CUDA + Tensor
                        6: "pink",  # TMA + Tensor
                        7: "purple",  # All three
                    }

                    # Plot each task as a horizontal bar with two phases
                    for i in range(n):
                        color = resource_colors.get(resource_flags[i], "blue")
                        # Calculate actual start time: start_times[i] + best_ii * stages[i]
                        start = start_times[i] + best_ii * stages[i]
                        latency = latencies[i]
                        ii = iis[i]

                        # Calculate ii boundary
                        ii_boundary = start + ii if start + ii < start + latency else start + latency

                        # Plot [start, start+ii] phase with darker color
                        phase1_width = ii_boundary - start
                        if phase1_width > 0:
                            # Darken the color for initial phase
                            rgb = plt.cm.colors.to_rgb(color)
                            hls = colorsys.rgb_to_hls(*rgb)
                            darker_color = colorsys.hls_to_rgb(hls[0], max(0, hls[1] * 0.7), hls[2])

                            rect1 = patches.Rectangle(
                                (start, i - 0.4),  # (x, y)
                                phase1_width,
                                0.8,  # width, height
                                linewidth=1,
                                edgecolor="black",
                                facecolor=darker_color,
                                alpha=0.9,
                                hatch="//",
                                label="Initial Phase [start, start+II]" if i == 0 else "",
                            )
                            ax.add_patch(rect1)

                        # Plot [start+ii, end] phase with original color
                        phase2_width = (start + latency) - ii_boundary
                        if phase2_width > 0:
                            rect2 = patches.Rectangle(
                                (ii_boundary, i - 0.4),  # (x, y)
                                phase2_width,
                                0.8,  # width, height
                                linewidth=1,
                                edgecolor="black",
                                facecolor=color,
                                alpha=0.7,
                                label="Remaining Phase [start+II, end]" if i == 0 else "",
                            )
                            ax.add_patch(rect2)

                        # Add task label (black text for better visibility)
                        ax.text(
                            start + latency / 2,
                            i,
                            f"T{i}\nS{stages[i]}\nII={ii}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white",
                            fontweight="bold",
                        )

                        # Add vertical line at ii boundary
                        if phase1_width > 0 and phase2_width > 0:
                            ax.axvline(x=ii_boundary, color="white", linestyle="-", alpha=0.8, linewidth=1)

                    # Set limits and labels
                    max_time = max(start_times[i] + best_ii * stages[i] + latencies[i] for i in range(n))
                    ax.set_xlim(begin - 5, max_time + 5)
                    ax.set_ylim(-1, n)
                    ax.set_xlabel("Time (cycles)")
                    ax.set_ylabel("Task Index")
                    ax.set_title(f"Z3 Schedule Timeline (Minimal II={best_ii})")
                    ax.grid(True, alpha=0.3)

                    # Add legend for resource types and phases (outside the plot)
                    legend_elements = [
                        Patch(facecolor="green", edgecolor="black", label="CUDA Core"),
                        Patch(facecolor="red", edgecolor="black", label="TMA Core"),
                        Patch(facecolor="gold", edgecolor="black", label="Tensor Core"),
                        Patch(facecolor="blue", edgecolor="black", label="Other"),
                        Patch(facecolor="darkgreen", edgecolor="black", hatch="//", alpha=0.9, label="Initial Phase [start, start+II]"),
                        Patch(facecolor="green", edgecolor="black", alpha=0.7, label="Remaining Phase [start+II, end]"),
                    ]
                    # Place legend outside the plot on the right side
                    ax.legend(
                        handles=legend_elements,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=9,
                        frameon=True,
                        framealpha=0.9,
                        facecolor="white",
                    )

                    # Add vertical lines for each 10 time units
                    for t in range(0, max_time + 10, 10):
                        ax.axvline(x=t, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

                    # Add two vertical lines for loop boundaries (begin, begin+best_ii)
                    # Find the earliest start time as begin
                    loop_end = begin + best_ii

                    # Add thick red line for loop begin
                    ax.axvline(x=begin, color="red", linestyle="-", alpha=0.8, linewidth=2, label="Loop Begin")
                    # Add thick blue line for loop end (begin + best_ii)
                    ax.axvline(x=loop_end, color="blue", linestyle="-", alpha=0.8, linewidth=2, label=f"Loop End (Begin+II={loop_end})")

                    # Shade the loop region
                    ax.axvspan(begin, loop_end, alpha=0.1, color="yellow", label="Loop Region")

                    # Add text annotation for loop boundaries
                    ax.text(
                        begin,
                        n - 0.5,
                        f"Begin\n{begin}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="red", alpha=0.3, edgecolor="red"),
                    )
                    ax.text(
                        loop_end,
                        n - 0.5,
                        f"Begin+II\n{loop_end}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="blue", alpha=0.3, edgecolor="blue"),
                    )

                    # # Adjust layout to make room for the legend and phase explanation outside the plot
                    # plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave 25% space on the right for legend and phase explanation

                    # # Add phase explanation text outside the plot (to the right of legend)
                    # fig.text(
                    #     0.88,  # x position (88% of figure width)
                    #     0.5,  # y position (center)
                    #     "Phase Explanation:\n"
                    #     "• Initial Phase [start, start+II]:\n  Hatched pattern, darker color\n"
                    #     "• Remaining Phase [start+II, end]:\n  Solid fill, lighter color\n"
                    #     "• Red vertical line: Loop Begin\n"
                    #     "• Blue vertical line: Loop End\n  (Begin + Minimal II)\n"
                    #     "• Yellow shaded area: Loop Region",
                    #     transform=fig.transFigure,
                    #     fontsize=9,
                    #     verticalalignment="center",
                    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black", pad=10),
                    # )

                    # Save PNG and PDF files
                    png_file = schedule_dir / "schedule_timeline.png"
                    pdf_file = schedule_dir / "schedule_timeline.pdf"
                    plt.savefig(png_file, dpi=150, bbox_inches="tight")
                    plt.savefig(pdf_file, bbox_inches="tight")
                    plt.close(fig)  # Close the figure to free memory

                    print(f"[Python Z3 Loop] Schedule visualization saved as PNG: {png_file}")
                    print(f"[Python Z3 Loop] Schedule visualization saved as PDF: {pdf_file}")

                except ImportError as e:
                    print(f"[Python Z3 Loop] Warning: matplotlib not available, skipping visualization: {e}")
                except Exception as e:
                    print(f"[Python Z3 Loop] Warning: Failed to generate visualization: {e}")

                print(f"[Python Z3 Loop] Schedule visualization saved to {schedule_dir}/ (text, JSON, PNG, PDF)")

            except Exception as e:
                print(f"[Python Z3 Loop] Warning: Failed to save schedule visualization: {e}")

        return start_times, stages, best_ii

    except Exception as e:
        if verbose:
            print(f"[Python Z3 Loop] Error in Z3 scheduling: {e}")
        # Fallback to simple schedule
        data_deps_pairs = [(i, j) for i, j, d in data_deps]
        return fallback_schedule(latencies, data_deps_pairs, resource_deps), sum(latencies) + 1


# FFI-exposed function for loop scheduling
@tvm_ffi.register_global_func("tl.transform.z3_schedule_loop_python")
def z3_schedule_loop_ffi(num_stages, latencies, iis, resource_flags, data_deps, resource_deps):
    """FFI wrapper for z3_schedule_loop_python.

    This function accepts TVM containers and converts them to Python lists.
    Data dependencies are expected as triples (i, j, distance).
    """
    # Convert TVM containers to Python lists
    latencies_list = list(latencies)
    iis_list = list(iis)
    resource_flags_list = list(resource_flags)

    # Convert data dependencies (triples)
    data_deps_list = []
    if data_deps is not None:
        # Assuming data_deps is a list of triples
        for i in range(len(data_deps)):
            if hasattr(data_deps[i], "__len__") and len(data_deps[i]) == 3:
                data_deps_list.append((int(data_deps[i][0]), int(data_deps[i][1]), int(data_deps[i][2])))

    # Convert resource dependencies (pairs)
    resource_deps_list = []
    if resource_deps is not None:
        for i in range(len(resource_deps)):
            if hasattr(resource_deps[i], "__len__") and len(resource_deps[i]) == 2:
                resource_deps_list.append((int(resource_deps[i][0]), int(resource_deps[i][1])))

    # Call the actual scheduler
    start_times, stages, best_ii = z3_schedule_loop_python(
        num_stages, latencies_list, iis_list, resource_flags_list, data_deps_list, resource_deps_list
    )

    # Return start_times and promotes as separate arrays for easier FFI handling
    # C++ side expects a tuple of (start_times_array, promotes_array)
    return (start_times, stages, best_ii)
