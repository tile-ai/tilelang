"""Z3-based scheduler for auto-scheduling.

This module provides a Python implementation of the Z3 scheduler that can be
called from C++ via TVM FFI.
"""

from __future__ import annotations

import tvm_ffi
import os
import json
import time
import threading
from pathlib import Path

# Global lock to serialize Z3 calls — Z3 is not thread-safe
_z3_lock = threading.Lock()

# Try to import z3, but handle missing installation gracefully
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("[Python Z3] WARNING: z3-solver package not installed. Z3 scheduling will not work.")


def _find_next_schedule_number(base_dir="debug"):
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


def _convert_int_tuples(values, arity: int) -> list[tuple[int, ...]]:
    if values is None:
        return []
    result = []
    for value in values:
        if not hasattr(value, "__len__") or len(value) != arity:
            actual_arity = len(value) if hasattr(value, "__len__") else "non-sequence"
            raise ValueError(f"Expected {arity} integers per tuple, got {actual_arity}")
        result.append(tuple(int(value[i]) for i in range(arity)))
    return result


def z3_schedule_python(
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int, int]],
    resource_deps: list[tuple[int, int]],
    owner_exclusion_deps: list[tuple[int, int, int, int]] | None = None,
    pipe_order_deps: list[tuple[int, int]] | None = None,
    verbose: bool = False,
) -> tuple[list[int], list[int]]:
    """Schedule a straight-line task list.

    Parameters
    ----------
    latencies : list[int]
        Execution latency of each task in cycles.
    iis : list[int]
        Initiation interval of each task in cycles.
    resource_flags : list[int]
        Resource pipe mask for each task (bitmask of ResourcePipe values):
        MTE1=1, MTE2=2, MTE3=4, Cube=8, Vector=16, Fixpipe=32, Scalar=64.
    data_deps : list[tuple[int, int, int]]
        Directed ``(i, j, latency)`` constraints requiring task ``j`` to start
        at least ``latency`` cycles after task ``i``.
    resource_deps : list[tuple[int, int]]
        Unordered task pairs that cannot issue concurrently on one resource.
    owner_exclusion_deps : list[tuple[int, int, int, int]] | None
        Undirected shared-storage owner conflicts ``(i, j, latency_i_j,
        latency_j_i)``. Either owner may run first, but their conflicting
        storage lifetimes cannot overlap.
    pipe_order_deps : list[tuple[int, int]] | None
        Source-ordered task pairs that share a hardware pipe.
    verbose : bool
        Print solver inputs and the selected schedule.

    Returns
    -------
    tuple[list[int], list[int]]
        start_times: Start time for each task
        sorted_indices: Task indices sorted by start time
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n < 1:
        raise RuntimeError("Z3 scheduling failed: n too small")
    if pipe_order_deps is None:
        pipe_order_deps = []
    if owner_exclusion_deps is None:
        owner_exclusion_deps = []

    if verbose:
        print(f"[Python Z3] Starting scheduling for {n} tasks")
        print(f"[Python Z3] Latencies: {latencies}")
        print(f"[Python Z3] IIs: {iis}")
        print(f"[Python Z3] Resource flags: {resource_flags}")
        print(f"[Python Z3] Data dependencies: {data_deps}")
        print(f"[Python Z3] Resource dependencies: {resource_deps}")
        print(f"[Python Z3] Owner exclusions: {owner_exclusion_deps}")

    assert Z3_AVAILABLE, "z3-solver package is required but not installed"

    # Use a private Z3 context so this solve is isolated from global-context
    # state accumulated by other solves in the same process (see the loop
    # scheduler for why this is required for deterministic results).
    ctx = z3.Context()

    # Create Z3 solver
    solver = z3.Optimize(ctx=ctx)
    solver.set("rlimit", 1000000)

    # Create start time variables
    start_vars = [z3.Int(f"start_{i}", ctx) for i in range(n)]

    # Add constraints: start times must be non-negative
    for var in start_vars:
        solver.add(var >= 0)

    # Add data dependency constraints
    # For each data dependency (i, j, latency), task j must start after task i starts + latency
    for i, j, latency in data_deps:
        solver.add(start_vars[j] >= start_vars[i] + latency)
        if verbose:
            print(f"[Python Z3] Data dependency: task {j} >= task {i} + {latency}")

    # Add resource dependency constraints
    # For tasks i and j that use same resource, they cannot execute simultaneously
    # We create ordering variable O_i,j (True means i before j, False means j before i)
    for i, j in resource_deps:
        if i < j:  # Only consider each pair once
            ii_i = iis[i]
            ii_j = iis[j]

            # Create ordering variable
            o_ij = z3.Bool(f"O_{i}_{j}", ctx)

            # If o_ij is True (i before j), then start_j >= start_i + ii_i
            solver.add(z3.Implies(o_ij, start_vars[j] >= start_vars[i] + ii_i))

            # If o_ij is False (j before i), then start_i >= start_j + ii_j
            solver.add(z3.Implies(z3.Not(o_ij), start_vars[i] >= start_vars[j] + ii_j))

            if verbose:
                print(f"[Python Z3] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

    for i, j, latency_i_j, latency_j_i in owner_exclusion_deps:
        i_before_j = z3.Bool(f"Owner_{i}_{j}", ctx)
        solver.add(z3.Implies(i_before_j, start_vars[j] - start_vars[i] >= latency_i_j))
        solver.add(z3.Implies(z3.Not(i_before_j), start_vars[i] - start_vars[j] >= latency_j_i))
        if verbose:
            print(f"[Python Z3] Owner exclusion between {i} and {j}: latency_i_j={latency_i_j}, latency_j_i={latency_j_i}")

    for previous, current in pipe_order_deps:
        solver.add(start_vars[previous] <= start_vars[current])

    # Objective: minimize maximum completion time (makespan)
    makespan = z3.Int("makespan", ctx)
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
                    f"resource_flags={resource_flags[idx]:07b}"
                )

        return start_times, task_indices
    else:
        raise RuntimeError("Z3 scheduling failed: solver returned unsat")


# FFI-exposed function that matches C++ interface
@tvm_ffi.register_global_func("tl.transform.z3_schedule_python")
def z3_schedule_ffi(
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps,
    owner_exclusion_deps=None,
    pipe_order_deps=None,
):
    """FFI wrapper for z3_schedule_python.

    This function accepts TVM containers and converts them to Python lists.
    """
    # Convert TVM containers to Python lists
    latencies_list = list(latencies)
    iis_list = list(iis)
    resource_flags_list = list(resource_flags)

    data_deps_list = _convert_int_tuples(data_deps, 3)
    resource_deps_list = _convert_int_tuples(resource_deps, 2)
    pipe_order_deps_list = _convert_int_tuples(pipe_order_deps, 2)
    owner_exclusion_deps_list = _convert_int_tuples(owner_exclusion_deps, 4)

    # Call the actual scheduler (Z3 is not thread-safe, serialize access)
    with _z3_lock:
        start_times, _ = z3_schedule_python(
            latencies_list,
            iis_list,
            resource_flags_list,
            data_deps_list,
            resource_deps_list,
            owner_exclusion_deps=owner_exclusion_deps_list,
            pipe_order_deps=pipe_order_deps_list,
        )

    # Return only start_times, C++ side will sort by start_time
    return start_times


def z3_schedule_loop_python(
    num_stages: int,
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int, int, int]],  # (i, j, distance, latency)
    resource_deps: list[tuple[int, int]],
    owner_exclusion_deps: list[tuple[int, int, int, int]] | None,
    pipe_order_deps: list[tuple[int, int]] | None,
    buffer_sizes: list[int],
    memory_groups: list[list[int]],  # [[capacity, idx0, idx1, ...], ...]
    stage_order_deps: list[tuple[int, int]] | None = None,  # (u, w): k_u <= k_w
    recalculate_buffer_versions: bool = True,
    enable_offset: bool = False,
    manual_stages: list[int] | None = None,
    manual_schedule: bool = False,
    seed: int | None = 42,
    verbose: bool = False,
) -> tuple[list[int], list[int], int]:
    """Schedule a loop body with modulo, stage, and storage constraints.

    Each start time is represented as ``k * II + r``. Directed dependencies
    constrain absolute start times, while resource and owner exclusions choose
    a non-overlapping modulo order. The solver finds the minimum feasible II by
    binary search.

    Parameters
    ----------
    num_stages : int
        Maximum number of physical versions available to automatic buffers.
    latencies : list[int]
        Execution latency of each task in cycles.
    iis : list[int]
        Initiation interval of each task in cycles.
    resource_flags : list[int]
        Resource pipe mask for each task (bitmask of ResourcePipe values):
        MTE1=1, MTE2=2, MTE3=4, Cube=8, Vector=16, Fixpipe=32, Scalar=64.
    data_deps : list[tuple[int, int, int, int]]
        Directed ``(i, j, distance, latency)`` constraints. Negative distances
        encode automatic buffer-version variables.
    resource_deps : list[tuple[int, int]]
        Unordered task pairs that cannot issue concurrently on one resource.
    owner_exclusion_deps : list[tuple[int, int, int, int]] | None
        Undirected shared-storage owner conflicts ``(i, j, latency_i_j,
        latency_j_i)``. The chosen modulo order also constrains the reverse
        wraparound hand-off.
    pipe_order_deps : list[tuple[int, int]] | None
        Source-ordered task pairs sharing a hardware pipe.
    buffer_sizes : list[int]
        Bytes occupied by one version of each automatic buffer.
    memory_groups : list[list[int]]
        Capacity followed by member buffer indices for each memory scope.
    stage_order_deps : list[tuple[int, int]] | None
        Stage-order constraints (u, w) requiring k_u <= k_w (same or earlier
        iteration-stage).
    recalculate_buffer_versions : bool
        Recompute the minimum version counts from the selected schedule.
    enable_offset : bool
        When False, constrain each resource so that, among the tasks using that
        resource (pipe bit in resource_flags), max(start_time) - min(start_time) < II.
        This keeps all tasks on a resource within a single II window (no offsetting
        a resource's tasks across pipeline stages). When True, no such
        constraint is added. Defaults to False (constraint applied).
    manual_stages : list[int] | None
        Frontend stages for tasks in source order. Used only when
        ``manual_schedule`` is True; omitted entries are not allowed.
    manual_schedule : bool
        Preserve source issue order independently on every hardware pipe and
        constrain every task's Z3 stage to the corresponding
        ``manual_stages`` entry.
    seed : int | None
        Z3 random seed; ``None`` leaves the solver default unchanged.
    verbose : bool
        Print solver inputs, search progress, and the selected schedule.

    Returns
    -------
    tuple[list[int], list[int], int]
        start_times: Start time for each task
        buffer_versions: Number of versions for each buffer
        minimal_II: The minimal initiation interval found
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n < 1:
        raise RuntimeError("Z3 loop scheduling failed: n too small")

    if stage_order_deps is None:
        stage_order_deps = []
    if pipe_order_deps is None:
        pipe_order_deps = []
    if owner_exclusion_deps is None:
        owner_exclusion_deps = []
    if manual_stages is None:
        manual_stages = [0] * n
    if manual_schedule:
        if len(manual_stages) != n:
            raise ValueError(f"Manual schedule has {len(manual_stages)} stages for {n} tasks")
        if any(stage < 0 for stage in manual_stages):
            raise ValueError(f"Manual schedule stages must be non-negative: {manual_stages}")
        if not enable_offset and any(stage != 0 for stage in manual_stages):
            raise ValueError("Non-zero manual stages require enable_offset=True")

    if verbose:
        print(f"[Python Z3 Loop] Starting scheduling for {n} tasks")
        print(f"[Python Z3 Loop] Latencies: {latencies}")
        print(f"[Python Z3 Loop] IIs: {iis}")
        print(f"[Python Z3 Loop] Resource flags: {resource_flags}")
        print(f"[Python Z3 Loop] Data dependencies with distances: {data_deps}")
        print(f"[Python Z3 Loop] Resource dependencies: {resource_deps}")
        print(f"[Python Z3 Loop] Owner exclusions: {owner_exclusion_deps}")
        print(f"[Python Z3 Loop] Stage-order dependencies: {stage_order_deps}")
        print(f"[Python Z3 Loop] Pipe-order dependencies: {pipe_order_deps}")
        print(f"[Python Z3 Loop] Manual schedule: {manual_schedule}, stages: {manual_stages}")
        print(f"[Python Z3 Loop] Buffer sizes: {buffer_sizes} within groups {memory_groups}")

    assert Z3_AVAILABLE, "z3-solver package is required but not installed"

    # Use a private Z3 context so this solve is isolated from any global-context
    # state accumulated by other solves in the same process. Sharing the global
    # context makes repeated identical solves return different (equally valid)
    # models run-to-run; a fresh context per call keeps the result deterministic.
    ctx = z3.Context()

    # Binary search for minimal II
    # Lower bound: 1 cycle
    # A sequential schedule must cover both dependency latency and the issue
    # intervals of tasks sharing a resource.
    ii_lower = max(iis)
    ii_upper = max(sum(latencies), sum(iis)) + 1
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
        solver = z3.Solver(ctx=ctx)
        if seed is not None:
            solver.set("random_seed", seed)
        solver.set("rlimit", 1000000)

        k_vars = [z3.Int(f"k_{i}", ctx) for i in range(n)]
        r_vars = [z3.Int(f"r_{i}", ctx) for i in range(n)]
        start_vars = [k_vars[i] * ii_mid + r_vars[i] for i in range(n)]

        for i in range(n):
            solver.add(k_vars[i] >= 0)
            solver.add(r_vars[i] >= 0)
            solver.add(r_vars[i] < ii_mid)

        if manual_schedule:
            for i, stage in enumerate(manual_stages):
                solver.add(k_vars[i] == stage)
        for previous, current in pipe_order_deps:
            solver.add(r_vars[previous] <= r_vars[current])

        buffer_vars = [z3.Int(f"buf_{i}", ctx) for i in range(len(buffer_sizes))]
        for i in range(len(buffer_sizes)):
            solver.add(buffer_vars[i] >= 1)
            solver.add(buffer_vars[i] <= num_stages)
        for group in memory_groups:
            cap = group[0]
            idxs = group[1:]
            if idxs:
                solver.add(z3.Sum([buffer_vars[i] * buffer_sizes[i] for i in idxs]) <= cap)

        # Add data dependency constraints with distance
        for u, v, distance, latency in data_deps:
            if distance >= 0:
                if verbose:
                    print(f"[Python Z3 Loop] Data dependency: task {v} - task {u} >= {latency} - {ii_mid}*{distance}")
                solver.add(start_vars[v] - start_vars[u] >= latency - ii_mid * distance)
            else:
                buffer_var_id = -distance - 1  # Convert to 0-based index
                if verbose:
                    print(
                        f"[Python Z3 Loop] Data dependency with negative distance: task {v} - task {u} >= {latency} - {ii_mid}*buf_{buffer_var_id}"
                    )
                solver.add(start_vars[v] - start_vars[u] >= latency - ii_mid * buffer_vars[buffer_var_id])

        # Add resource dependency constraints
        # For tasks i and j that use same resource, they cannot execute simultaneously
        # We create ordering variable O_i,j (True means i before j, False means j before i)
        for i, j in resource_deps:
            if i < j:  # Only consider each pair once
                ii_i = iis[i]
                ii_j = iis[j]

                # Create ordering variable
                o_ij = z3.Bool(f"O_{i}_{j}", ctx)

                solver.add(z3.Implies(o_ij, r_vars[j] - r_vars[i] >= ii_i))
                solver.add(z3.Implies(o_ij, r_vars[i] - r_vars[j] + ii_mid >= ii_j))
                solver.add(z3.Implies(z3.Not(o_ij), r_vars[i] - r_vars[j] >= ii_j))
                solver.add(z3.Implies(z3.Not(o_ij), r_vars[j] - r_vars[i] + ii_mid >= ii_i))

                if verbose:
                    print(f"[Python Z3 Loop] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

        # A physical storage ring shared by two disjoint owners is a cyclic
        # resource. Choose one modulo order and protect both the
        # forward hand-off and the wrap-around hand-off. Pipeline stage
        # offsets deliberately do not participate: owners exchange the same
        # physical slot at their issue phases, not at their logical stages.
        for i, j, latency_i_j, latency_j_i in owner_exclusion_deps:
            i_before_j = z3.Bool(f"Owner_{i}_{j}", ctx)
            solver.add(z3.Implies(i_before_j, r_vars[j] - r_vars[i] >= latency_i_j))
            solver.add(z3.Implies(i_before_j, r_vars[i] - r_vars[j] + ii_mid >= latency_j_i))
            solver.add(z3.Implies(z3.Not(i_before_j), r_vars[i] - r_vars[j] >= latency_j_i))
            solver.add(z3.Implies(z3.Not(i_before_j), r_vars[j] - r_vars[i] + ii_mid >= latency_i_j))
            if verbose:
                print(f"[Python Z3 Loop] Owner exclusion between {i} and {j}: latency_i_j={latency_i_j}, latency_j_i={latency_j_i}")

        # Add stage-order constraints for copied Let variables.
        # (u, w) requires k_u <= k_w: any unit u that reads a Let-defined var
        # must not be scheduled into a later iteration-stage than w, a unit that
        # writes the buffer the Let reads. This keeps the cloned Let (prepended
        # to the loop body in u's stage) reading the pre-write buffer value.
        for u, w in stage_order_deps:
            solver.add(k_vars[u] <= k_vars[w])
            if verbose:
                print(f"[Python Z3 Loop] Stage-order constraint: k_{u} <= k_{w}")

        # Optional offset constraint: when disabled, every task using a given
        # resource (pipe bit) must fit within a single II window, i.e.
        # max(start) - min(start) < II among that resource's tasks. For a set,
        # max - min < II is equivalent to |start_a - start_b| < II for all pairs.
        if not enable_offset:
            pipe_bits = [1, 2, 4, 8, 16, 32, 64]  # MTE1, MTE2, MTE3, Cube, Vector, Fixpipe, Scalar
            for bit in pipe_bits:
                group = [i for i in range(n) if resource_flags[i] & bit]
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        ia, ib = group[a], group[b]
                        solver.add(start_vars[ia] - start_vars[ib] < ii_mid)
                        solver.add(start_vars[ib] - start_vars[ia] < ii_mid)
                if verbose and len(group) > 1:
                    print(f"[Python Z3 Loop] Offset disabled: resource bit {bit} tasks {group} span < {ii_mid}")

        # Check feasibility
        if verbose:
            print(f"[Python Z3 Loop] Checking feasibility for II = {ii_mid}...")
        if solver.check() == z3.sat:
            if verbose:
                print(f"[Python Z3 Loop] II = {ii_mid} is feasible")
            best_ii = ii_mid
            best_model = solver.model()
            best_start_vars = start_vars
            best_k_vars = k_vars
            best_buffer_vars = buffer_vars
            # Try smaller II
            ii_upper = ii_mid
        else:
            if verbose:
                print(f"[Python Z3 Loop] II = {ii_mid} is infeasible")
            # Need larger II
            ii_lower = ii_mid + 1

    if best_model is None:
        if manual_schedule:
            raise RuntimeError(
                "Manual schedule constraints are infeasible; check per-pipe source order, T.Stage values, dependencies, and resource usage"
            )
        # No feasible II found. Fall back to a trivial sequential (non-pipelined)
        # schedule: each task starts right after the previous one finishes, in
        # index order. II = sum(latencies) makes successive loop iterations
        # non-overlapping. best_ii is clamped to >= 1 to avoid downstream
        # division by zero.
        if verbose:
            print("[Python Z3 Loop] No feasible II found; falling back to sequential schedule")
        start_times = [0] * n
        for i in range(1, n):
            start_times[i] = start_times[i - 1] + latencies[i - 1]
        best_ii = max(sum(latencies), 1)
        buffer_versions = [1] * len(buffer_sizes)
    else:
        if verbose:
            print(f"[Python Z3 Loop] Minimal feasible II = {best_ii}")

        # Extract start times from best model. Normalize by the smallest k so the
        # earliest iteration starts at 0.
        start_times = []
        k_vars = [best_model.eval(best_k_vars[i]).as_long() for i in range(n)]
        offset = min(k_vars)
        for i in range(n):
            start_time = best_model.eval(best_start_vars[i]).as_long() - offset * best_ii
            start_times.append(start_time)

        # Extract buffer versions from best model
        if recalculate_buffer_versions:
            # if recalculation is true, we recalculate based on the schedule and dependencies
            buffer_versions = len(buffer_sizes) * [1]
            for u, v, distance, latency in data_deps:
                if distance < 0:
                    buffer_var_id = -distance - 1
                    from math import ceil

                    buffer_versions[buffer_var_id] = max(
                        buffer_versions[buffer_var_id], ceil((latency + start_times[u] - start_times[v]) / best_ii)
                    )
        else:
            buffer_versions = []
            for i in range(len(buffer_sizes)):
                buffer_var = best_model.eval(best_buffer_vars[i]).as_long()
                buffer_versions.append(buffer_var)

    # Sort tasks by start time (and by index as tie-breaker)
    task_indices = list(range(n))
    task_indices.sort(key=lambda idx: (start_times[idx], idx))

    if verbose:
        print(f"[Python Z3 Loop] Scheduling completed. Minimal II = {best_ii}.")
    for i in range(n):
        idx = task_indices[i]
        if verbose:
            print(
                f"[Python Z3 Loop]   Task {idx}: start_time={start_times[idx]}, "
                f"latency={latencies[idx]}, II={iis[idx]}, "
                f"resource_flags={resource_flags[idx]:07b}"
            )
    if verbose:
        for i in range(len(buffer_sizes)):
            print(f"[Python Z3 Loop]   Buffer {i}: size={buffer_sizes[i]}, num_versions={buffer_versions[i]}")

    # Save schedule visualization when verbose is True
    if verbose:
        try:
            # Find the next available schedule number
            schedule_num = _find_next_schedule_number()
            schedule_dir = Path(f"debug/schedule_{schedule_num:03d}")
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
                "owner_exclusions": owner_exclusion_deps,
                "manual_schedule": manual_schedule,
                "manual_stages": manual_stages,
                "start_times": start_times,
                "sorted_indices": task_indices,
                "timestamp": time.time(),
            }
            if seed is not None:
                schedule_info["random_seed"] = seed

            # Save as JSON
            info_file = schedule_dir / "schedule_info.json"
            with open(info_file, "w") as f:
                json.dump(schedule_info, f, indent=2)

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

                # Create color map for resources (pipe mask values)
                def _get_pipe_color(mask):
                    if mask & 8:  # Cube
                        return "gold"
                    if mask & 16:  # Vector
                        return "green"
                    if mask & 2:  # MTE2 (GM->UB)
                        return "red"
                    if mask & 4:  # MTE3 (UB->GM)
                        return "orange"
                    if mask & 1:  # MTE1
                        return "cyan"
                    if mask & 32:  # Fixpipe
                        return "pink"
                    return "blue"

                # Plot each task as a horizontal bar with two phases
                for i in range(n):
                    color = _get_pipe_color(resource_flags[i])
                    start = start_times[i]
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
                        f"T{i}\nII={ii}",
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
                max_time = max(start_times[i] + latencies[i] for i in range(n))
                min_time = min(start_times[i] for i in range(n))
                ax.set_xlim(min_time - 5, max_time + 5)
                ax.set_ylim(-1, n)
                ax.set_xlabel("Time (cycles)")
                ax.set_ylabel("Task Index")
                ax.set_title(f"Z3 Schedule Timeline (Minimal II={best_ii})")
                ax.grid(True, alpha=0.3)

                # Add legend for resource types and phases (outside the plot)
                legend_elements = [
                    Patch(facecolor="gold", edgecolor="black", label="Cube (M)"),
                    Patch(facecolor="green", edgecolor="black", label="Vector (V)"),
                    Patch(facecolor="red", edgecolor="black", label="MTE2 (GM->)"),
                    Patch(facecolor="orange", edgecolor="black", label="MTE3 (->GM)"),
                    Patch(facecolor="cyan", edgecolor="black", label="MTE1 (L1->L0)"),
                    Patch(facecolor="pink", edgecolor="black", label="Fixpipe"),
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

    return start_times, buffer_versions, best_ii


# FFI-exposed function for loop scheduling
@tvm_ffi.register_global_func("tl.transform.z3_schedule_loop_python")
def z3_schedule_loop_ffi(
    num_stages,
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps,
    owner_exclusion_deps,
    pipe_order_deps,
    buffer_sizes,
    memory_groups,
    stage_order_deps=None,
    enable_offset=False,
    manual_stages=None,
    manual_schedule=False,
):
    """FFI wrapper for z3_schedule_loop_python.

    This function accepts TVM containers and converts them to Python lists.
    Dependency arrays are ordered as directed data dependencies, resource
    exclusions, shared-storage owner exclusions, and source pipe ordering.
    ``memory_groups`` contains ``[capacity, idx0, idx1, ...]`` entries;
    ``stage_order_deps`` contains ``(u, w)`` constraints requiring ``k_u <=
    k_w``. The remaining arguments carry the loop's offset and manual-stage
    policy from C++.
    """
    # Convert TVM containers to Python lists
    latencies_list = list(latencies)
    iis_list = list(iis)
    resource_flags_list = list(resource_flags)
    buffer_sizes_list = list(buffer_sizes)
    # Convert memory groups
    memory_groups_list = []
    if memory_groups is not None:
        for i in range(len(memory_groups)):
            memory_groups_list.append(list(memory_groups[i]))
    data_deps_list = _convert_int_tuples(data_deps, 4)
    resource_deps_list = _convert_int_tuples(resource_deps, 2)
    stage_order_deps_list = _convert_int_tuples(stage_order_deps, 2)

    manual_stages_list = []
    if manual_stages is not None:
        manual_stages_list = [int(stage) for stage in manual_stages]

    pipe_order_deps_list = _convert_int_tuples(pipe_order_deps, 2)
    owner_exclusion_deps_list = _convert_int_tuples(owner_exclusion_deps, 4)

    # Call the actual scheduler (Z3 is not thread-safe, serialize access)
    with _z3_lock:
        start_times, buffer_versions, best_ii = z3_schedule_loop_python(
            num_stages,
            latencies_list,
            iis_list,
            resource_flags_list,
            data_deps_list,
            resource_deps_list,
            owner_exclusion_deps_list,
            pipe_order_deps_list,
            buffer_sizes_list,
            memory_groups_list,
            stage_order_deps=stage_order_deps_list,
            enable_offset=bool(enable_offset),
            manual_stages=manual_stages_list,
            manual_schedule=bool(manual_schedule),
            seed=42,
        )

    # Return start_times and buffer_versions as separate arrays for easier FFI handling
    # C++ side expects a tuple of (start_times_array, buffer_versions_array, best_ii)
    return (start_times, buffer_versions, best_ii)
