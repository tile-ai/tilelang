# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Exercise producer/consumer compiler lifecycles without executing CUDA kernels.

Run with a source build of TileLang and its development dependencies:
    python testing/python/transform/producer_consumer_ws_scenarios.py
Use --output DIR to retain the transformed IR, generated CUDA and results.
"""

import argparse
import importlib.util
import json
from functools import partial
from pathlib import Path
import traceback

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target


def make_im2col_kernel(channels=1, binding="none"):
    @T.prim_func
    def main(
        data: T.Tensor((1, 8, 8, channels), T.float16),
        weights: T.Tensor((128,), T.float32),
        weights_out: T.Tensor((4, 128), T.float32),
        data_out: T.Tensor((4, 16, channels), T.float16),
    ):
        with T.Kernel(1, threads=128):
            weight_shared = T.alloc_shared((128,), T.float32)
            data_shared = T.alloc_shared((16, channels), T.float16)
            for step in T.Pipelined(4, num_stages=2):
                T.copy(weights, weight_shared)
                if binding == "pure":
                    step_base = T.int32(step % 2)
                    nhw_step = T.int32(step_base + 1)
                    T.im2col(data, data_shared, nhw_step, 0, 3, 1, 1, 1)
                elif binding == "stateful":
                    nhw_step = T.cast(weight_shared[0], T.int32)
                    T.im2col(data, data_shared, nhw_step, 0, 3, 1, 1, 1)
                else:
                    T.im2col(data, data_shared, step % 2, 0, 3, 1, 1, 1)
                for i in T.Parallel(128):
                    weights_out[step, i] = weight_shared[i]
                for i, j in T.Parallel(16, channels):
                    data_out[step, i, j] = data_shared[i, j]

    return main


def make_overwrite_kernel(mode, destination):
    @T.prim_func
    def main(
        data: T.Tensor((1, 8, 8, 32), T.float16),
        weights: T.Tensor((128,), T.float32),
        weights_out: T.Tensor((4, 128), T.float32),
        data_out: T.Tensor((4, 16, 32), T.float16),
    ):
        with T.Kernel(1, threads=128):
            weight_shared = T.alloc_shared((128,), T.float32)
            data_shared = T.alloc_shared((16, 32), T.float16)
            for step in T.Pipelined(4, num_stages=2):
                T.copy(weights, weight_shared)
                T.im2col(data, data_shared, step % 2, 0, 3, 1, 1, 1)
                if mode == "fill":
                    if destination == "simt":
                        T.fill(data_shared, 0)
                    else:
                        T.fill(weight_shared, 0)
                else:
                    if destination == "simt":
                        for i, j in T.Parallel(16, 32):
                            data_shared[i, j] = T.float16(0)
                    else:
                        for i in T.Parallel(128):
                            weight_shared[i] = T.float32(0)
                for i in T.Parallel(128):
                    weights_out[step, i] = weight_shared[i]
                for i, j in T.Parallel(16, 32):
                    data_out[step, i, j] = data_shared[i, j]

    return main


def make_dual_kernel(binding="none", prelude=False, include_simt=True):
    @T.prim_func
    def main(
        data: T.Tensor((1, 8, 8, 32), T.float16),
        weights: T.Tensor((128,), T.float32),
        other_weights: T.Tensor((3, 128), T.float32),
        weights_out: T.Tensor((4, 128), T.float32),
        other_out: T.Tensor((4, 128), T.float32),
        data_out: T.Tensor((4, 16, 32), T.float16),
    ):
        with T.Kernel(1, threads=128):
            weight_shared = T.alloc_shared((128,), T.float32)
            other_shared = T.alloc_shared((128,), T.float32)
            data_shared = T.alloc_shared((16, 32), T.float16)
            if prelude:
                T.copy(weights, weight_shared)
            for step in T.Pipelined(4, num_stages=2):
                if not prelude:
                    T.copy(weights, weight_shared)
                if include_simt:
                    T.im2col(data, data_shared, step % 2, 0, 3, 1, 1, 1)
                if binding == "pure":
                    base_offset = T.int32(step % 2)
                    other_offset = T.int32(base_offset + 1)
                    T.copy(other_weights[other_offset, 0], other_shared)
                elif binding == "stateful":
                    other_offset = T.cast(weight_shared[0], T.int32) % 2
                    T.copy(other_weights[other_offset, 0], other_shared)
                else:
                    T.copy(other_weights[step % 2, 0], other_shared)
                for i in T.Parallel(128):
                    weight_shared[i] = T.float32(0)
                for i in T.Parallel(128):
                    other_out[step, i] = other_shared[i]
                for i in T.Parallel(128):
                    weights_out[step, i] = weight_shared[i]
                if include_simt:
                    for i, j in T.Parallel(16, 32):
                        data_out[step, i, j] = data_shared[i, j]

    return main


def make_shared_producers(order):
    @T.prim_func
    def main(data: T.Tensor((1, 8, 8, 32), T.float16), seed: T.Tensor((16, 32), T.float16), out: T.Tensor((4, 16, 32), T.float16)):
        with T.Kernel(1, threads=128):
            target_shared = T.alloc_shared((16, 32), T.float16)
            for step in T.Pipelined(4, num_stages=2):
                if order == "tma_first":
                    T.copy(seed, target_shared)
                    T.im2col(data, target_shared, step % 2, 0, 3, 1, 1, 1)
                else:
                    T.im2col(data, target_shared, step % 2, 0, 3, 1, 1, 1)
                    T.copy(seed, target_shared)
                for i, j in T.Parallel(16, 32):
                    out[step, i, j] = target_shared[i, j]

    return main


def make_preceding_consumer():
    @T.prim_func
    def main(
        a: T.Tensor((128,), T.float32),
        b: T.Tensor((128,), T.float32),
        out_a: T.Tensor((4, 128), T.float32),
        out_b: T.Tensor((4, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((128,), T.float32)
            target_shared = T.alloc_shared((128,), T.float32)
            for step in T.Pipelined(4, num_stages=2):
                T.copy(a, a_shared)
                for i in T.Parallel(128):
                    target_shared[i] = T.float32(0)
                T.copy(b, target_shared)
                for i in T.Parallel(128):
                    out_a[step, i] = a_shared[i]
                    out_b[step, i] = target_shared[i]

    return main


def touches(expr, buffer_name):
    found = []
    tvm.tirx.stmt_functor.post_order_visit(
        expr,
        lambda node: found.append(node) if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name.startswith(buffer_name) else None,
    )
    return bool(found)


def same_barrier(left, right):
    if not isinstance(left, tvm.tirx.BufferLoad) or not isinstance(right, tvm.tirx.BufferLoad):
        raise RuntimeError("expected a buffer-load barrier reference")
    return left.buffer.data.same_as(right.buffer.data) and tvm.ir.structural_equal(left.indices, right.indices, map_free_vars=True)


def target_barriers(body, producer, buffer_name, destination):
    forward = []
    phases = []
    reuse = []
    last_wait = []

    def collect_forward(node):
        if not isinstance(node, tvm.tirx.Call):
            return
        name = getattr(node.op, "name", "")
        if destination == "simt":
            if name.endswith("ptx_cp_async_barrier"):
                forward.append(node.args[0])
            if name == "tl.tileop.im2col" and touches(node.args[1], buffer_name):
                phases.append(node.annotations["tl.pipeline_mbar_phase_expr"])
        elif name == "tl.tileop.tma_copy" and touches(node.args[1], buffer_name):
            if "barrier" not in node.annotations:
                raise RuntimeError("target TMA copy has no completion barrier")
            forward.append(node.annotations["barrier"])
            phases.append(
                node.annotations["tl.pipeline_mbar_phase_expr"]
                if "tl.pipeline_mbar_phase_expr" in node.annotations
                else tvm.tirx.IntImm("int32", 0)
            )

    tvm.tirx.stmt_functor.post_order_visit(producer if destination == "simt" else body, collect_forward)

    def collect_reuse(node):
        if not isinstance(node, tvm.tirx.Call):
            return
        name = getattr(node.op, "name", "")
        if name.endswith("mbarrier_wait_parity"):
            last_wait[:] = [node.args[0]]
        matches = (destination == "tma" and name == "tl.tileop.tma_copy") or (destination == "simt" and name == "tl.tileop.im2col")
        if matches and touches(node.args[1], buffer_name):
            if not last_wait:
                raise RuntimeError("target producer has no preceding reuse wait")
            reuse.append(last_wait[0])

    tvm.tirx.stmt_functor.post_order_visit(producer, collect_reuse)
    if len(forward) != 1 or len(phases) != 1 or len(reuse) > 1:
        raise RuntimeError(f"expected one target completion barrier, got {len(forward)} forward/{len(reuse)} reuse")
    return forward[0], phases[0], reuse[0] if reuse else None


def events(stmt, buffer_name, forward, phase, reuse):
    result = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name.startswith(buffer_name):
            result.append("read")
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.name.startswith(buffer_name):
            result.append("overwrite")
        if isinstance(node, tvm.tirx.Call):
            name = getattr(node.op, "name", "")
            if name.endswith("mbarrier_wait_parity"):
                matches = same_barrier(node.args[0], forward) and tvm.ir.structural_equal(node.args[1], phase, map_free_vars=True)
                result.append("target_wait" if matches else "other_wait")
            if name.endswith("ptx_arrive_barrier") and reuse is not None and same_barrier(node.args[0], reuse):
                result.append("target_release")
            if name == "tl.tileop.fill" and touches(node.args[0], buffer_name):
                result.append("overwrite")

    tvm.tirx.stmt_functor.post_order_visit(stmt, visit)
    return result


def ordering_violations(producer, consumer, reuse):
    violations = []
    if "overwrite" in producer:
        violations.append("consumer overwrite moved into producer issue sequence")
    access_positions = [i for i, event in enumerate(consumer) if event in ("read", "overwrite")]
    if access_positions and ("target_wait" not in consumer or access_positions[0] < consumer.index("target_wait")):
        violations.append("consumer access precedes its target completion wait")
    if (
        reuse is not None
        and access_positions
        and ("target_release" not in consumer or consumer.index("target_release") < access_positions[-1])
    ):
        violations.append("target stage released before its final consumer access")
    return violations


def outer_definitions(func, split):
    def extend(defined, variables):
        result = list(defined)
        for var in variables:
            if not any(var.same_as(old) for old in result):
                result.append(var)
        return result

    defined = extend(list(func.params), [buffer.data for buffer in func.buffer_map.values()])

    def find(stmt, scope):
        if stmt.same_as(split):
            return scope
        if isinstance(stmt, tvm.tirx.AttrStmt):
            if stmt.attr_key == "thread_extent" and isinstance(stmt.node, tvm.tirx.IterVar):
                scope = extend(scope, [stmt.node.var])
            return find(stmt.body, scope)
        if isinstance(stmt, tvm.tirx.SBlockRealize):
            return find(stmt.block, scope)
        if isinstance(stmt, tvm.tirx.SBlock):
            scope = extend(
                scope,
                [axis.var for axis in stmt.iter_vars]
                + [buffer.data for buffer in stmt.alloc_buffers]
                + [match.buffer.data for match in stmt.match_buffers],
            )
            return find(stmt.body, scope)
        if isinstance(stmt, tvm.tirx.For):
            return find(stmt.body, extend(scope, [stmt.loop_var]))
        if isinstance(stmt, tvm.tirx.SeqStmt):
            for item in stmt.seq:
                found = find(item, scope)
                if found is not None:
                    return found
                if isinstance(item, tvm.tirx.Bind):
                    scope = extend(scope, [item.var])
                elif isinstance(item, tvm.tirx.AllocBuffer):
                    scope = extend(scope, [item.buffer.data])
        if isinstance(stmt, tvm.tirx.IfThenElse):
            found = find(stmt.then_case, scope)
            if found is not None:
                return found
            if stmt.else_case is not None:
                return find(stmt.else_case, scope)
        return None

    result = find(func.body, defined)
    if result is None:
        raise RuntimeError("producer split has no recognized lexical scope")
    return result


def inspect_producer_bindings(func, split):
    producer = split.then_case
    definitions = outer_definitions(func, split)
    undefined = tvm.tirx.analysis.undefined_vars(producer, definitions)
    if undefined:
        raise RuntimeError(f"producer references undefined scalar variables: {[str(var) for var in undefined]}")
    selected = []

    def move_required_bind(node):
        if not isinstance(node, tvm.tirx.SeqStmt) or selected:
            return None
        statements = list(node.seq)
        for i, stmt in enumerate(statements):
            if not isinstance(stmt, tvm.tirx.Bind) or i + 1 == len(statements):
                continue
            used = []
            tvm.tirx.stmt_functor.post_order_visit(
                tvm.tirx.SeqStmt(statements[i + 1 :]),
                lambda var, definition=stmt.var, found=used: (
                    found.append(var) if isinstance(var, tvm.tirx.Var) and var.same_as(definition) else None
                ),
            )
            if used:
                selected.append(stmt.var)
                return tvm.tirx.SeqStmt(statements[:i] + statements[i + 1 :] + [stmt])
        return None

    mutant = tvm.tirx.stmt_functor.ir_transform(producer, None, move_required_bind, ["tirx.SeqStmt"])
    if not selected:
        raise RuntimeError("pure-binding fixture has no observable definition dependency")
    rejection = None
    try:
        missing = tvm.tirx.analysis.undefined_vars(mutant, definitions)
    except Exception as exc:
        if "used before definition" not in str(exc):
            raise
        rejection = str(exc)
    else:
        if any(var.same_as(selected[0]) for var in missing):
            rejection = "moved binding is undefined at its original consumer"
    if rejection is None:
        raise RuntimeError("TVM undefined_vars accepted moving a required Bind after its use")
    return {"producer_undefined_variables": [], "binding_mutation_rejected": True, "binding_mutation_error": rejection}


def is_specialized(func):
    return "tl_tiled_ws_applied" in func.attrs


def producer_consumer_split(func):
    def has_tma(stmt):
        calls = []
        tvm.tirx.stmt_functor.post_order_visit(
            stmt,
            lambda node: (
                calls.append(node) if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.tma_copy" else None
            ),
        )
        return bool(calls)

    splits = []

    def visit(node):
        if isinstance(node, tvm.tirx.IfThenElse) and node.else_case is not None and has_tma(node.then_case):
            splits.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    if len(splits) != 1:
        raise RuntimeError(f"expected one producer/consumer split, got {len(splits)}")
    return splits[0]


def check_fallback(before, after, name):
    if is_specialized(after):
        raise RuntimeError(f"{name}: dependent producer work was warp-specialized")

    def remove_pipeline_hint(node):
        if isinstance(node, tvm.tirx.For) and "num_stages" in node.annotations:
            annotations = {key: value for key, value in node.annotations.items() if key != "num_stages"}
            return tvm.tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                thread_binding=node.thread_binding,
                annotations=annotations,
                step=node.step,
            )
        return None

    expected_body = tvm.tirx.stmt_functor.ir_transform(before.body, None, remove_pipeline_hint, ["tirx.For"])
    if not tvm.ir.structural_equal(before.with_body(expected_body), after):
        raise RuntimeError("fallback changed the original loop beyond its num_stages hint")
    return {"warp_specialized": False, "original_loop_preserved": True, "removed_annotation": "num_stages"}


def check_im2col(before, after, name):
    if name == "stateful_binding":
        return check_fallback(before, after, name)
    if not is_specialized(after):
        raise RuntimeError("mixed TMA/im2col pipeline lost warp specialization")
    split = producer_consumer_split(after)
    forward, phase, reuse = target_barriers(after.body, split.then_case, "data_shared", "simt")
    producer = events(split.then_case, "data_shared", forward, phase, reuse)
    consumer = events(split.else_case, "data_shared", forward, phase, reuse)
    violations = ordering_violations(producer, consumer, reuse)
    if violations:
        raise RuntimeError(violations)
    arrivals = []
    tvm.tirx.stmt_functor.post_order_visit(
        split.then_case,
        lambda node: (
            arrivals.append(node)
            if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "").endswith("ptx_arrive_barrier")
            else None
        ),
    )
    if not arrivals:
        raise RuntimeError("SIMT fallback has no release arrival for synchronous stores")
    binding = inspect_producer_bindings(after, split) if name == "pure_bindings" else {}
    return {"warp_specialized": True, "consumer_events": consumer, **binding}


def check_first_access(before, after, name, destination):
    applied = is_specialized(after)
    if name == "dual_tma_stateful_binding":
        return check_fallback(before, after, name)
    if not applied:
        raise RuntimeError("eligible first-access scenario lost warp specialization")
    split = producer_consumer_split(after)
    binding_result = inspect_producer_bindings(after, split) if "pure_binding" in name else {}
    buffer_name = "data_shared" if destination == "simt" else "weight_shared"
    forward, phase, reuse = target_barriers(after.body, split.then_case, buffer_name, destination)
    producer = events(split.then_case, buffer_name, forward, phase, reuse)
    consumer = events(split.else_case, buffer_name, forward, phase, reuse)
    violations = ordering_violations(producer, consumer, reuse)
    if "overwrite" not in producer and "overwrite" not in consumer:
        raise RuntimeError("scenario overwrite was not identified in the transformed IR")
    result = {
        "eligible": True,
        "forward_barrier": forward.script(),
        "forward_phase": phase.script(),
        "reuse_barrier": reuse.script() if reuse is not None else None,
        "producer_events": producer,
        "consumer_events": consumer,
        "violations": violations,
        **binding_result,
    }
    if violations:
        raise RuntimeError(violations)
    if name in ("pure_tma_pure_binding", "prelude_tma"):
        mutant = list(consumer)
        own, other, overwrite = mutant.index("target_wait"), mutant.index("other_wait"), mutant.index("overwrite")
        if not own < overwrite < other:
            raise RuntimeError("wrong-barrier mutation needs distinct waits around the overwrite")
        mutant[own], mutant[other] = mutant[other], mutant[own]
        rejected = ordering_violations(producer, mutant, reuse)
        if "consumer access precedes its target completion wait" not in rejected:
            raise RuntimeError("ordering oracle accepted the wrong-barrier event mutation")
        result["wrong_barrier_mutation"] = {"consumer_events": mutant, "violations": rejected}
    return result


def check_existing(before, after, name):
    if not is_specialized(after):
        raise RuntimeError("existing independent-buffer example lost warp specialization")
    return {"warp_specialized": True}


def run_case(name, func, check, arch, output):
    record = {"scenario": name, "arch": arch}
    target = determine_target({"kind": "cuda", "arch": arch}, return_object=True)
    try:
        mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        before = tilelang.transform.MaterializeKernelLaunch()(mod)
        after = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(before)
        if output:
            (output / f"{name}.tir").write_text(after["main"].script())
        record.update(check(before["main"], after["main"], name))
        record["ws_checked"] = True
    except Exception as exc:
        record["ws_error"] = str(exc)
        traceback.print_exc()
    try:
        with tvm.transform.PassContext(), target:
            artifact = tilelang.lower(func, target=target)
        if not artifact.kernel_source:
            raise RuntimeError("compiler returned no CUDA source")
        if output:
            (output / f"{name}.cu").write_text(artifact.kernel_source)
        record["cuda_source_bytes"] = len(artifact.kernel_source)
    except Exception as exc:
        record["cuda_source_error"] = str(exc)
        traceback.print_exc()
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, channels, binding in (
        ("scalar_fallback", 1, "none"),
        ("late_read", 32, "none"),
        ("pure_bindings", 32, "pure"),
        ("stateful_binding", 32, "stateful"),
    ):
        records.append(run_case(name, make_im2col_kernel(channels, binding), check_im2col, "sm_100", args.output))
    for mode in ("fill", "parallel"):
        for destination in ("simt", "tma"):
            records.append(
                run_case(
                    f"{mode}_{destination}",
                    make_overwrite_kernel(mode, destination),
                    partial(check_first_access, destination=destination),
                    "sm_100",
                    args.output,
                )
            )
    for name, func in (
        ("dual_tma", make_dual_kernel()),
        ("dual_tma_pure_binding", make_dual_kernel(binding="pure")),
        ("pure_tma_pure_binding", make_dual_kernel(binding="pure", include_simt=False)),
        ("prelude_tma", make_dual_kernel(prelude=True)),
        ("dual_tma_stateful_binding", make_dual_kernel(binding="stateful", include_simt=False)),
    ):
        records.append(run_case(name, func, partial(check_first_access, destination="tma"), "sm_100", args.output))
    for name, func in (
        ("tma_first", make_shared_producers("tma_first")),
        ("im2col_first", make_shared_producers("im2col_first")),
        ("consumer_before_tma", make_preceding_consumer()),
    ):
        records.append(run_case(name, func, check_fallback, "sm_100", args.output))
    project = Path(tilelang.__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "existing_ws_factories", project / "testing/python/transform/test_tilelang_transform_producer_consumer_ws.py"
    )
    factories = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factories)
    for name, func in (
        ("existing_pure_tma", factories.matmul_pipelined(128, 128, 128, 64, 32, 64, 2)),
        ("existing_prelude_tma", factories.prelude_tma_wait_sink()),
        ("existing_dual_gemm", factories.dual_gemm_shared_accumulator(64, 64, 32, 64, 64, 64, 32, 2)),
    ):
        records.append(run_case(name, func, check_existing, "sm_90", args.output))
    failures = sum(bool(row.get("ws_error") or row.get("cuda_source_error")) for row in records)
    result = {"scenarios": records, "failures": failures, "device_execution": False}
    if args.output:
        (args.output / "scenario-results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"scenarios": len(records), "failures": failures, "device_execution": False}), flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
