"""Z3-based scheduler for auto-scheduling.

This module provides a Python implementation of the Z3 scheduler that can be
called from C++ via TVM FFI.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import tvm
import tvm_ffi

# Try to import z3, but handle missing installation gracefully
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("[Python Z3] WARNING: z3-solver package not installed. Z3 scheduling will not work.")

# Try to import mip, but handle missing installation gracefully
try:
    import mip
    MIP_AVAILABLE = True
except ImportError:
    MIP_AVAILABLE = False
    print("[Python MIP] WARNING: python-mip package not installed. MIP scheduling will not work.")

# Try to import ortools, but handle missing installation gracefully
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("[Python OR-Tools] WARNING: ortools package not installed. OR-Tools scheduling will not work.")


def z3_schedule_python(
    latencies: List[int],
    iis: List[int],
    resource_flags: List[int],
    data_deps: List[Tuple[int, int]],
    resource_deps: List[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    """Z3-based scheduler implemented in Python.

    Parameters
    ----------
    latencies : List[int]
        Latency for each task in cycles
    iis : List[int]
        Initiation interval for each task in cycles
    resource_flags : List[int]
        Resource usage flags for each task (bitmask):
        1 = uses CUDA core, 2 = uses TMA core, 4 = uses Tensor core
    data_deps : List[Tuple[int, int]]
        Data dependency pairs (i, j) where task j depends on task i
    resource_deps : List[Tuple[int, int]]
        Resource dependency pairs (i, j) where tasks i and j use same resource

    Returns
    -------
    Tuple[List[int], List[int]]
        start_times: Start time for each task
        sorted_indices: Task indices sorted by start time
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n <= 1:
        if n == 1:
            return [0], [0]
        return [], []

    print(f"[Python Z3] Starting scheduling for {n} tasks")
    print(f"[Python Z3] Latencies: {latencies}")
    print(f"[Python Z3] IIs: {iis}")
    print(f"[Python Z3] Resource flags: {resource_flags}")
    print(f"[Python Z3] Data dependencies: {data_deps}")
    print(f"[Python Z3] Resource dependencies: {resource_deps}")

    # Check if z3 is available
    if not Z3_AVAILABLE:
        print(f"[Python Z3] WARNING: z3-solver not available, using fallback scheduler")
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
                if latency_i < 0:
                    latency_i = abs(latency_i)
                solver.add(start_vars[j] >= start_vars[i] + latency_i)
                print(f"[Python Z3] Data dependency: task {j} >= task {i} + {latency_i}")

        # Add resource dependency constraints
        # For tasks i and j that use same resource, they cannot execute simultaneously
        # We create ordering variable O_i,j (True means i before j, False means j before i)
        for i, j in resource_deps:
            if i < j:  # Only consider each pair once
                ii_i = iis[i]
                ii_j = iis[j]
                if ii_i < 0:
                    ii_i = abs(ii_i)
                if ii_j < 0:
                    ii_j = abs(ii_j)

                # Create ordering variable
                o_ij = z3.Bool(f"O_{i}_{j}")

                # If o_ij is True (i before j), then start_j >= start_i + ii_i
                solver.add(z3.Implies(o_ij, start_vars[j] >= start_vars[i] + ii_i))

                # If o_ij is False (j before i), then start_i >= start_j + ii_j
                solver.add(z3.Implies(z3.Not(o_ij), start_vars[i] >= start_vars[j] + ii_j))

                print(f"[Python Z3] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

        # Objective: minimize maximum completion time (makespan)
        makespan = z3.Int("makespan")
        for i in range(n):
            latency_i = latencies[i]
            if latency_i < 0:
                latency_i = abs(latency_i)
            solver.add(makespan >= start_vars[i] + latency_i)
        solver.add(makespan >= 0)

        # Minimize makespan
        solver.minimize(makespan)

        # Check satisfiability
        print(f"[Python Z3] Checking satisfiability...")
        if solver.check() == z3.sat:
            model = solver.model()

            # Extract start times
            start_times = []
            for i in range(n):
                try:
                    start_time = model.eval(start_vars[i]).as_long()
                except:
                    # Fallback to 0 if evaluation fails
                    start_time = 0
                start_times.append(start_time)

            # Get makespan
            makespan_val = model.eval(makespan).as_long()

            # Sort tasks by start time (and by index as tie-breaker)
            task_indices = list(range(n))
            task_indices.sort(key=lambda idx: (start_times[idx], idx))

            print(f"[Python Z3] Scheduling completed. Makespan = {makespan_val}")
            for i in range(n):
                idx = task_indices[i]
                print(f"[Python Z3]   Task {idx}: start_time={start_times[idx]}, "
                      f"latency={latencies[idx]}, II={iis[idx]}, "
                      f"resource_flags={resource_flags[idx]:03b}")

            return start_times, task_indices
        else:
            print(f"[Python Z3] Z3 scheduling failed, falling back to topological sort")
            # Fallback: return tasks in original order with start times based on dependencies
            return fallback_schedule(latencies, data_deps, resource_deps)

    except Exception as e:
        print(f"[Python Z3] Error in Z3 scheduling: {e}")
        # Fallback to simple schedule
        return fallback_schedule(latencies, data_deps, resource_deps)


def fallback_schedule(
    latencies: List[int],
    data_deps: List[Tuple[int, int]],
    resource_deps: List[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
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
        print(f"[Python Z3] Cycle detected in dependencies, using original order")
        result = list(range(n))
        # Simple linear schedule
        for i in range(1, n):
            start_times[i] = start_times[i-1] + latencies[i-1]

    return start_times, result


# FFI-exposed function that matches C++ interface
@tvm_ffi.register_global_func("tl.transform.z3_schedule_python")
def z3_schedule_ffi(
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps
):
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
            if hasattr(data_deps[i], '__len__') and len(data_deps[i]) == 2:
                data_deps_list.append((int(data_deps[i][0]), int(data_deps[i][1])))

    # Convert resource dependencies
    resource_deps_list = []
    if resource_deps is not None:
        for i in range(len(resource_deps)):
            if hasattr(resource_deps[i], '__len__') and len(resource_deps[i]) == 2:
                resource_deps_list.append((int(resource_deps[i][0]), int(resource_deps[i][1])))

    # Call the actual scheduler
    start_times, _ = z3_schedule_python(
        latencies_list,
        iis_list,
        resource_flags_list,
        data_deps_list,
        resource_deps_list
    )

    # Return only start_times, C++ side will sort by start_time
    return start_times


def z3_schedule_loop_python(
    latencies: List[int],
    iis: List[int],
    resource_flags: List[int],
    data_deps: List[Tuple[int, int, int]],  # (i, j, distance)
    resource_deps: List[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
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
    latencies : List[int]
        Latency for each task in cycles
    iis : List[int]
        Initiation interval for each task in cycles
    resource_flags : List[int]
        Resource usage flags for each task (bitmask):
        1 = uses CUDA core, 2 = uses TMA core, 4 = uses Tensor core
    data_deps : List[Tuple[int, int, int]]
        Data dependency tuples (i, j, distance) where task j depends on task i
        with distance d (loop iterations distance)
    resource_deps : List[Tuple[int, int]]
        Resource dependency pairs (i, j) where tasks i and j use same resource

    Returns
    -------
    Tuple[List[int], List[int]]
        start_times: Start time for each task
        sorted_indices: Task indices sorted by start time
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n <= 1:
        if n == 1:
            return [0], [0]
        return [], []

    print(f"[Python Z3 Loop] Starting scheduling for {n} tasks")
    print(f"[Python Z3 Loop] Latencies: {latencies}")
    print(f"[Python Z3 Loop] IIs: {iis}")
    print(f"[Python Z3 Loop] Resource flags: {resource_flags}")
    print(f"[Python Z3 Loop] Data dependencies with distances: {data_deps}")
    print(f"[Python Z3 Loop] Resource dependencies: {resource_deps}")

    # Check if z3 is available
    if not Z3_AVAILABLE:
        print(f"[Python Z3 Loop] WARNING: z3-solver not available, using fallback scheduler")
        # Convert data_deps to pairs for fallback (ignore distance)
        data_deps_pairs = [(i, j) for i, j, d in data_deps]
        return fallback_schedule(latencies, data_deps_pairs, resource_deps)

    try:
        # Binary search for minimal II
        # Lower bound: 1 cycle
        # Upper bound: sum of all latencies (conservative)
        ii_lower = 1
        ii_upper = sum(abs(l) if l > 0 else 1 for l in latencies) + 1
        best_ii = ii_upper
        best_model = None
        best_start_vars = None

        print(f"[Python Z3 Loop] Binary search range: [{ii_lower}, {ii_upper})")

        while ii_lower < ii_upper:
            ii_mid = (ii_lower + ii_upper) // 2
            print(f"[Python Z3 Loop] Testing II = {ii_mid}")

            # Create solver for feasibility check
            solver = z3.Solver()
            solver.set("timeout", 100000)

            start_vars = [z3.Int(f"start_{i}") for i in range(n)]
            pro_vars = [z3.Bool(f"pro_{i}") for i in range(n)]
            begin = z3.Int("begin")

            for i in range(n):
                solver.add(start_vars[i] >= 0)
                solver.add(begin <= start_vars[i] + pro_vars[i] * ii_mid)
                latency_i = latencies[i]
                if latency_i < 0:
                    latency_i = abs(latency_i)
                solver.add(start_vars[i] + latency_i + pro_vars[i] * ii_mid - begin <= ii_mid)

            # Add data dependency constraints with distance
            for u, v, distance in data_deps:
                latency_u = latencies[u]
                if latency_u < 0:
                    latency_u = abs(latency_u)
                print(f"[Python Z3 Loop] Data dependency: task {v} - task {u} >= {latency_u} - {ii_mid}*{distance}")
                solver.add(start_vars[v] - start_vars[u] >= latency_u - ii_mid * distance)

            # Add resource dependency constraints
            # For tasks i and j that use same resource, they cannot execute simultaneously
            # We create ordering variable O_i,j (True means i before j, False means j before i)
            for i, j in resource_deps:
                if i < j:  # Only consider each pair once
                    ii_i = iis[i]
                    ii_j = iis[j]
                    if ii_i < 0:
                        ii_i = abs(ii_i)
                    if ii_j < 0:
                        ii_j = abs(ii_j)

                    # Create ordering variable
                    o_ij = z3.Bool(f"O_{i}_{j}")

                    # If o_ij is True (i before j), then start_j >= start_i + ii_i
                    solver.add(z3.Implies(o_ij, start_vars[j] >= start_vars[i] + ii_i))

                    # If o_ij is False (j before i), then start_i >= start_j + ii_j
                    solver.add(z3.Implies(z3.Not(o_ij), start_vars[i] >= start_vars[j] + ii_j))

                    print(f"[Python Z3 Loop] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

            # Check feasibility
            print(f"[Python Z3 Loop] Checking feasibility for II = {ii_mid}...")
            if solver.check() == z3.sat:
                print(f"[Python Z3 Loop] II = {ii_mid} is feasible")
                best_ii = ii_mid
                best_model = solver.model()
                best_start_vars = start_vars
                best_pro_vars = pro_vars
                # Try smaller II
                ii_upper = ii_mid
            else:
                print(f"[Python Z3 Loop] II = {ii_mid} is infeasible")
                # Need larger II
                ii_lower = ii_mid + 1

        if best_model is None:
            print(f"[Python Z3 Loop] No feasible II found in range, falling back to topological sort")
            data_deps_pairs = [(i, j) for i, j, d in data_deps]
            return fallback_schedule(latencies, data_deps_pairs, resource_deps)

        print(f"[Python Z3 Loop] Minimal feasible II = {best_ii}")

        # Extract start times from best model
        start_times = []
        promotes = []
        for i in range(n):
            try:
                start_time = best_model.eval(best_start_vars[i]).as_long()
                promote = z3.is_true(best_model.eval(best_pro_vars[i]))
            except:
                # Fallback to 0 if evaluation fails
                start_time = 0
                promote = False
            start_times.append(start_time)
            promotes.append(promote)

        # Sort tasks by start time (and by index as tie-breaker)
        task_indices = list(range(n))
        task_indices.sort(key=lambda idx: (start_times[idx], idx))

        print(f"[Python Z3 Loop] Scheduling completed. Minimal II = {best_ii}")
        for i in range(n):
            idx = task_indices[i]
            print(f"[Python Z3 Loop]   Task {idx}: start_time={start_times[idx]}, "
                  f"latency={latencies[idx]}, II={iis[idx]}, pro={promotes[idx]}, "
                  f"resource_flags={resource_flags[idx]:03b}")

        return start_times, promotes, task_indices

    except Exception as e:
        print(f"[Python Z3 Loop] Error in Z3 scheduling: {e}")
        # Fallback to simple schedule
        data_deps_pairs = [(i, j) for i, j, d in data_deps]
        return fallback_schedule(latencies, data_deps_pairs, resource_deps)


# FFI-exposed function for loop scheduling
@tvm_ffi.register_global_func("tl.transform.z3_schedule_loop_python")
def z3_schedule_loop_ffi(
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps
):
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
            if hasattr(data_deps[i], '__len__') and len(data_deps[i]) == 3:
                data_deps_list.append((int(data_deps[i][0]), int(data_deps[i][1]), int(data_deps[i][2])))

    # Convert resource dependencies (pairs)
    resource_deps_list = []
    if resource_deps is not None:
        for i in range(len(resource_deps)):
            if hasattr(resource_deps[i], '__len__') and len(resource_deps[i]) == 2:
                resource_deps_list.append((int(resource_deps[i][0]), int(resource_deps[i][1])))

    # Call the actual scheduler
    start_times, promotes, _ = z3_schedule_loop_python(
        latencies_list,
        iis_list,
        resource_flags_list,
        data_deps_list,
        resource_deps_list
    )

    # Return start_times and promotes as separate arrays for easier FFI handling
    # C++ side expects a tuple of (start_times_array, promotes_array)
    return (start_times, promotes)
