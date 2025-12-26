from tvm import tir


__all__ = [
    "pdl_trigger",
    "pdl_sync",
]


def pdl_trigger():
    return tir.call_extern(
        "int32",  # cudaError_t
        "cudaTriggerProgrammaticLaunchCompletion",
    )


def pdl_sync():
    return tir.call_extern(
        "int32",  # cudaError_t
        "cudaGridDependencySynchronize",
    )
