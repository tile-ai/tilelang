"""Grouped compilation helpers for autotuner.

This module isolates backend-aware grouped compilation logic from AutoTuner.run
so tuner.py can stay focused on orchestration.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from tilelang import tvm
from tvm.tirx import PrimFunc

from tilelang import env
from tilelang.env import resolve_pass_profile_threshold_ms
from tilelang.autotuner.param import CompileArgs
from tilelang.engine.lower import lower_to_host_device_ir, device_codegen, host_codegen
from tilelang.engine.param import CompiledArtifact
from tilelang.jit.adapter import TVMFFIKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.transform import PassConfigKey
from tilelang.utils.pass_timing import build_pass_instruments, report_pass_timing_on_exit

CompileUnitResult = tuple[int, dict[str, Any], JITKernel | None, Exception | None]


def compile_grouped_unit_tvm_ffi(
    unit_items: list[tuple[int, dict[str, Any]]],
    compile_args: CompileArgs,
    elaborate_func: Callable[..., PrimFunc],
) -> list[CompileUnitResult]:
    """Compile one grouped unit for CUDA+tvm_ffi backend.

    Flow:
    1. Elaborate each config into a PrimFunc.
    2. Lower each PrimFunc into host/device IR modules.
    3. Merge all device IR into one IRModule and compile device code once.
    4. Build host runtime module per config and import shared device module.
    5. Construct per-config JITKernel objects that share the grouped device module.
    """

    pass_configs = dict(compile_args.pass_configs) if compile_args.pass_configs else {}
    base_pass_instruments = []
    if pass_configs.get(PassConfigKey.TL_ENABLE_DUMP_IR):
        dump_ir_path = pass_configs.get(PassConfigKey.TL_DUMP_IR_DIR, "./dump_ir")
        base_pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=dump_ir_path))

    enable_profile = pass_configs.get(PassConfigKey.TL_PASS_PROFILE) or env.is_pass_profile_enabled()
    profile_threshold_ms = None
    if enable_profile:
        profile_threshold_ms = resolve_pass_profile_threshold_ms(
            pass_configs,
            PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS,
            env.get_pass_profile_threshold_ms,
        )

    def create_pass_instruments():
        return build_pass_instruments(base_pass_instruments, profile_threshold_ms)

    unit_results: list[CompileUnitResult] = []
    lowered_items: list[dict[str, Any]] = []

    for idx, config_arg in unit_items:
        try:
            program = elaborate_func(**config_arg)
            original_symbol = str(program.attrs["global_symbol"])
            unique_symbol = f"{original_symbol}_gc_{idx}"
            program = program.with_attr("global_symbol", unique_symbol)

            config_instruments, timing_inst = create_pass_instruments()

            with (
                report_pass_timing_on_exit(
                    timing_inst,
                    context=f"stage=grouped-lower, config={idx}, kernel={unique_symbol}",
                ),
                tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=config_instruments),
                compile_args.target,
            ):
                host_mod, device_mod, params, normalized_target, normalized_target_host = lower_to_host_device_ir(
                    program,
                    target=compile_args.target,
                    target_host=compile_args.target_host,
                )

            lowered_items.append(
                {
                    "idx": idx,
                    "config_arg": config_arg,
                    "program": program,
                    "host_mod": host_mod,
                    "device_mod": device_mod,
                    "params": params,
                    "target": normalized_target,
                    "target_host": normalized_target_host,
                }
            )
        except Exception as e:
            unit_results.append((idx, config_arg, None, e))

    if not lowered_items:
        return unit_results

    try:
        merged_funcs: dict[Any, Any] = {}
        merged_attrs = None
        merged_names: set[str] = set()
        for item in lowered_items:
            device_mod = item["device_mod"]
            if merged_attrs is None:
                merged_attrs = device_mod.attrs
            for global_var, func in device_mod.functions.items():
                name_hint = getattr(global_var, "name_hint", str(global_var))
                if name_hint in merged_names:
                    raise RuntimeError(
                        f"Duplicate device global symbol '{name_hint}' during grouped compilation (config index={item['idx']})."
                    )
                merged_names.add(name_hint)
                merged_funcs[global_var] = func
        merged_device_mod = tvm.IRModule(merged_funcs, attrs=merged_attrs)

        reference_target = lowered_items[0]["target"]
        device_instruments, device_timing_inst = create_pass_instruments()
        grouped_config_indices = ",".join(str(item["idx"]) for item in lowered_items)
        with (
            report_pass_timing_on_exit(
                device_timing_inst,
                context=f"stage=grouped-device, configs=[{grouped_config_indices}]",
            ),
            tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=device_instruments),
            reference_target,
        ):
            grouped_device_rt_mod = device_codegen(merged_device_mod, reference_target)

        grouped_kernel_source = grouped_device_rt_mod.inspect_source()

        for item in lowered_items:
            idx = item["idx"]
            config_arg = item["config_arg"]
            try:
                host_instruments, host_timing_inst = create_pass_instruments()
                kernel_symbol = str(item["program"].attrs["global_symbol"])
                with (
                    report_pass_timing_on_exit(
                        host_timing_inst,
                        context=f"stage=grouped-host, config={idx}, kernel={kernel_symbol}",
                    ),
                    tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=host_instruments),
                    item["target"],
                ):
                    grouped_host_rt_mod = host_codegen(item["host_mod"], item["target_host"], target=item["target"])

                grouped_host_rt_mod.import_module(grouped_device_rt_mod)

                artifact = CompiledArtifact(
                    host_mod=grouped_host_rt_mod,
                    device_mod=item["device_mod"],
                    params=item["params"],
                    kernel_source=grouped_kernel_source,
                    rt_mod=grouped_host_rt_mod,
                )

                adapter = TVMFFIKernelAdapter(
                    params=artifact.params,
                    result_idx=compile_args.out_idx,
                    target=compile_args.target,
                    func_or_mod=item["program"],
                    host_mod=artifact.host_mod,
                    device_mod=artifact.device_mod,
                    rt_mod=artifact.rt_mod,
                    device_kernel_source=artifact.kernel_source,
                    verbose=compile_args.verbose,
                    pass_configs=pass_configs,
                )

                jit_kernel = JITKernel(
                    func=item["program"],
                    out_idx=compile_args.out_idx,
                    execution_backend=compile_args.execution_backend,
                    target=compile_args.target,
                    target_host=compile_args.target_host,
                    verbose=compile_args.verbose,
                    pass_configs=pass_configs,
                    from_database=True,
                )
                jit_kernel.artifact = artifact
                jit_kernel.adapter = adapter
                jit_kernel.torch_function = adapter.func

                unit_results.append((idx, config_arg, jit_kernel, None))
            except Exception as e:
                unit_results.append((idx, config_arg, None, e))
    except Exception as e:
        for item in lowered_items:
            unit_results.append((item["idx"], item["config_arg"], None, e))

    return unit_results
