# Copyright 2018 The apache/tvm Authors. All Rights Reserved.
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Modifications Copyright (c) Microsoft.
# The code below is mostly copied from apache/tvm common_schedules.py in dlight.
"""Common schedule strategies for TIR."""

from collections.abc import Callable

from ._native_analysis import NativeSchedule, StageRef
from .analysis import BlockInfo


def get_block(
    sch: NativeSchedule,
    blocks: list[BlockInfo],
    name: str,
):
    """Get the target block from a schedule.

    Parameters
    ----------
    sch : tirx.NativeSchedule
        The TIR schedule used to get target block.
    name : str
        The name of the target block.

    Returns
    -------
    target_block : StageRef
        The target block.
    """

    target_block: StageRef = None
    for block_info in blocks:
        block = block_info.block_rv
        if sch.get(block).name_hint == name:
            target_block = block
    return target_block


def get_output_blocks(
    sch: NativeSchedule,
    blocks: list[BlockInfo],
):
    """Get the output blocks of a schedule.

    Parameters
    ----------
    sch : tirx.NativeSchedule
        The TIR schedule used to get output blocks.
    blocks : List[BlockInfo]
        The blocks to be analyzed.

    Returns
    -------
    output_blocks : List[BlockInfo]
        The output blocks.
    """

    # collect arguments buffer
    args = sch.parameter_tensors

    output_blocks = []
    for block_info in blocks:
        block = block_info.block_rv
        for write in sch.get(block).writes:
            if write.tensor in args:
                output_blocks.append(block)

    return output_blocks


def try_inline(
    sch: NativeSchedule,
    blocks: list[BlockInfo],
) -> list[BlockInfo]:
    """Try to inline as many blocks as possible, and return the remaining blocks.

    Parameters
    ----------
    sch : tirx.NativeSchedule
        The TIR schedule used to inline blocks.
    blocks : List[BlockInfo]
        The blocks to be inlined.

    Returns
    -------
    remaining : List[BlockInfo]
        The remaining blocks that cannot be inlined.
    """

    def _trial(func: Callable):
        for i, block in enumerate(blocks):
            try:
                func(block.block_rv)
            except Exception:  # pylint: disable=bare-except
                continue
            return i
        return None

    while True:
        i = _trial(sch.compute_inline)
        if i is None:
            i = _trial(sch.reverse_compute_inline)
        if i is None:
            break
        blocks.pop(i)
        for info in blocks:
            info.block_rv = sch.refresh(info.block_rv)
            for axis in info.iters:
                axis.loop_rv = sch.refresh(axis.loop_rv)
    return blocks


def try_inline_contiguous_spatial(
    sch: NativeSchedule,
    block_infos: list[BlockInfo],
) -> list[BlockInfo]:
    """Try to inline contiguous spatial blocks in a schedule

    Parameters
    ----------
    sch : tirx.NativeSchedule
        The TIR schedule used to inline blocks.
    block_infos : List[BlockInfo]
        The blocks to be try.

    Returns
    -------
    remaining : List[BlockInfo]
        The remaining blocks that cannot be inlined.
    """

    if block_infos is None:
        return None
    with sch.transaction():
        results = []
        spatial_blocks = []
        for block in block_infos:
            if block.is_injective():
                spatial_blocks.append(block)
            elif spatial_blocks:
                results.extend(try_inline(sch, spatial_blocks))
                results.append(block)
                spatial_blocks = []
            else:
                results.append(block)
        if spatial_blocks:
            results.extend(try_inline(sch, spatial_blocks))
    for info in results:
        info.block_rv = sch.refresh(info.block_rv)
        for axis in info.iters:
            axis.loop_rv = sch.refresh(axis.loop_rv)
    return results
