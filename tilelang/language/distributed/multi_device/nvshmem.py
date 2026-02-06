"""The language interface for tl programs."""

from tvm import tir


def get_pe():
    """Get the processing element (PE) ID."""
    return tir.call_intrin("int32", tir.op.Op.get("tl.GetPE"))


def get_pe_num():
    """Get the total number of processing elements (PEs)."""
    return tir.call_intrin("int32", tir.op.Op.get("tl.GetPENum"))


def int_p(dest, value, pe):
    """Put a single integer value to a remote PE with a very low latency.
    Args:
        dest: Symmetric address of the destination data object.
        value: The value to be transferred to dest.
        pe: The PE ID of the destination PE.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.IntPE"), dest, value, pe)


def barrier_all():
    """Synchronizes all processing elements (PEs),
    ensuring completion of all previously issued memory stores and remote memory updates."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.BarrierAll"))


def barrier_all_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BarrierAllBlock"), *args)


def barrier_all_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BarrierAllWarp"), *args)


def sync_all():
    """Synchronizes all processing elements (PEs).
    In contrast with `barrier_all`,
    `sync_all` only ensures completion and visibility of previously issued memory stores,
    and does not ensure completion of remote memory updates issued via NVSHMEM routines."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.SyncAll"))


def sync_all_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.SyncAllBlock"), *args)


def sync_all_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.SyncAllWarp"), *args)


def quiet():
    """Ensures completion of all operations on symmetric data objects issued by the calling PE."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.Quiet"))


def fence():
    """Ensures ordering of delivery of operations on symmetric data objects."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.Fence"))


def getmem_nbi_block(dest, src, nelems, pe):
    """Get data from remote memory to local memory at block granularity without blocking.
    Args:
        dest: Symmetric address of the destination data object.
        src: Symmetric address of the object containing the data to be copied.
        nelems: Number of elements to be transferred (in bytes).
        pe: The PE ID of the source PE.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemNbiBlock"), dest, src, nelems, pe)


def getmem_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemBlock"), *args)


def getmem_nbi_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemNbiWarp"), *args)


def getmem_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemWarp"), *args)


def getmem_nbi(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemNbi"), *args)


def getmem(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Getmem"), *args)


def putmem_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemBlock"), *args)


def putmem_nbi_block(dest, src, nelems, pe):
    """Put data from local memory to remote memory at block granularity without blocking.
    Args:
        dest: Symmetric address of the destination data object.
        src: Symmetric address of the object containing the data to be copied.
        nelems: Number of elements to be transferred (in bytes).
        pe: The PE ID of the destination PE.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemNbiBlock"), dest, src, nelems, pe)


def putmem_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemWarp"), *args)


def putmem_nbi_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemNbiWarp"), *args)


def putmem(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Putmem"), *args)


def putmem_nbi(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemNbi"), *args)


def putmem_signal(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignal"), *args)


def putmem_signal_nbi(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalNbi"), *args)


def putmem_signal_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalBlock"), *args)


def putmem_signal_nbi_block(dest, src, nelems, sig_addr, signal, sig_op, pe):
    """Put data from local memory to remote memory at block granularity without blocking,
    and update a remote flag on delivery.
    Args:
        dest: Symmetric address of the destination data object.
        src: Symmetric address of the object containing the data to be copied.
        nelems: Number of elements to be transferred (in bytes).
        sig_addr: Symmetric address of the remote flag to be updated.
        signal: The value used for updating the remote signal data object.
        sig_op: The type of update to be performed on the remote signal data object.
        pe: The PE ID of the destination PE.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalNbiBlock"), dest, src, nelems, sig_addr, signal, sig_op, pe)


def putmem_signal_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalWarp"), *args)


def putmem_signal_nbi_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalNbiWarp"), *args)


def signal_op(sig_addr, signal, sig_op, pe):
    """Atomically updates `sig_addr` with `signal` using operation `sig_op` on the specified PE.
    Args:
        sig_addr: Symmetric address of the signal word to be updated.
        signal: The value used for updating the remote signal data object.
        sig_op: The type of update to be performed on the remote signal data object.
        pe: The PE ID of the destination PE.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.SignalOp"), sig_addr, signal, sig_op, pe)


def signal_wait_until(*args):
    # TODO: handle return value(which is uint*64)?
    return tir.call_intrin("int32", tir.op.Op.get("tl.SignalWaitUntil"), *args)


def broadcast(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Broadcast"), *args)


def broadcast_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BroadcastWarp"), *args)


def broadcast_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BroadcastBlock"), *args)


def broadcastmem_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BroadcastmemBlock"), *args)


def fcollect(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Fcollect"), *args)


def fcollect_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.FcollectWarp"), *args)


def fcollect_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.FcollectBlock"), *args)
