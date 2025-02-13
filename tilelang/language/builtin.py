# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir


def CreateListofMBarrierOp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.CreateListofMBarrierOp"), *args)


def GetMBarrierOp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetMBarrierOp"), *args)


def CreateTMADescriptorOp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.CreateTMADescriptorOp"), *args)


def TMALoadOp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.TMALoadOp"), *args)


def FenceProxyAsyncOp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.FenceProxyAsyncOp"), *args)


def SetMaxNReg(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.SetMaxNReg"), *args)


def MBarrierWaitParity(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.MBarrierWaitParity"), *args)


def MBarrierExpectTX(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.MBarrierExpectTX"), *args)
