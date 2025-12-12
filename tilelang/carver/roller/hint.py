"""Hint definition for schedule"""

from tvm import DataType
from . import PrimFuncNode
import numpy as np
from .rasterization import *


class TensorCoreExtraConfig:
    """
    This class is used to store extra information for tensorcore
    """

    def __init__(
        self,
        AS_shape: tuple[int],
        BS_shape: tuple[int],
        AF_shape: tuple[int],
        BF_shape: tuple[int],
        tc_axis: tuple[int],
    ) -> None:
        self.AS_shape: tuple[int] = AS_shape
        self.BS_shape: tuple[int] = BS_shape
        self.AF_shape: tuple[int] = AF_shape
        self.BF_shape: tuple[int] = BF_shape
        self.tc_axis: tuple[int] = tc_axis


class Stride:
    """
    Manages stride information for a given axis of a tensor.
    """

    def __init__(self, stride: int = 1, ax: int = -1) -> None:
        # which axis to put stride on
        self._ax: int = int(ax)
        # the stride size of the axis
        self._stride: int = int(stride)

    @property
    def ax(self) -> int:
        return self._ax

    @property
    def stride(self) -> int:
        return self._stride

    def compute_strides_from_shape(self, shape: list[int]) -> list[int]:
        ndim = len(shape)
        strides = [1 for _ in shape]
        for i in range(ndim - 2, -1, -1):
            if i == self.ax:
                strides[i] = self.stride
            else:
                strides[i] = int(strides[i + 1] * shape[i + 1])
        return strides

    def compute_elements_from_shape(self, shape: list[int]) -> int:
        original_shape = np.prod(shape)
        if not self.is_valid():
            strided_elem = original_shape
        else:
            assert self.ax < len(shape)
            strided_elem = np.prod(shape[0 : self.ax + 1]) * self.stride
            assert strided_elem >= original_shape
        return int(strided_elem)

    def is_valid(self) -> bool:
        return self.ax >= 0

    def __repr__(self) -> str:
        return f"<Stride, {self._ax}, {self._stride}>"


class TileDict:
    """
    Manages tiling information and configurations for computational tasks.
    """

    def __init__(self, output_tile) -> None:
        self.output_tile = output_tile
        # schedule config
        self.tile_map = {}
        self.rstep_map = {}
        self.cached_tensors_map = {}
        self.output_strides_map = {}
        self.tensor_strides_map = {}

        # analysis
        self.traffic = -1
        self.smem_cost = -1
        self.block_per_SM = -1
        self.num_wave = -1
        self.grid_size = -1
        self.valid = True

    def get_tile(self, func) -> list[int]:
        return self.tile_map[func]

    def get_rstep(self, node) -> dict[str, int]:
        return self.rstep_map[node]

    def __hash__(self) -> int:
        return hash(tuple(self.output_tile))


class IntrinInfo:
    """
    The information of tensorcore intrinsic related information
    """

    def __init__(
        self,
        in_dtype: str,
        out_dtype: str,
        trans_b: bool,
        input_transform_kind: int = 0,
        weight_transform_kind: int = 0,
    ) -> None:
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.trans_a = False
        self.trans_b = trans_b
        self.input_transform_kind = input_transform_kind
        self.weight_transform_kind = weight_transform_kind

    def __repr__(self) -> str:
        return f"<IntrinInfo, {self.in_dtype}, {self.out_dtype}, {self.trans_b}, {self.propagate_b}>"

    def is_input_8bit(self) -> bool:
        return DataType(self.in_dtype).bits == 8

    @property
    def smooth_a(self) -> bool:
        return self.input_transform_kind >= 2

    @property
    def smooth_b(self) -> bool:
        return self.weight_transform_kind >= 2

    @property
    def inter_transform_a(self) -> bool:
        return self.input_transform_kind >= 1

    @property
    def inter_transform_b(self) -> bool:
        return self.weight_transform_kind >= 1


class Hint:
    """
    Central configuration class for managing various parameters of computational tasks.
    """

    def __init__(self) -> None:
        self.arch = None
        self.use_tc = None  # todo(lei): this should be renamed.

        # Special axes tiling info
        self.block = []
        self.thread = []
        # Special axes for MFMA
        self.warp = []
        # Reduce axes tiling info
        self.rstep = []
        self.reduce_thread = []
        self.rasterization_plan = NoRasterization()
        self.cached_tensors = []
        self.output_strides = {}
        self.schedule_stages = None
        # Config for block reduction
        self.block_reduction_depth = None  # type: int

        # TL Specific
        # Split-K factor for SM waste optimization
        self.split_k_factor: int = 1

        # Experimental
        self._raxis_order = []
        self._step = []
        self.vectorize: dict[str, int] = {}
        self.pipeline_stage = 1
        self.use_async = False
        self.opt_shapes: dict[str, int] = {}
        self.intrin_info = IntrinInfo("float16", "float16", True)
        self.shared_scope: str = "shared"
        self.pass_context: dict = {}

    def to_dict(self) -> dict:
        dic = {}
        dic["block"] = self.block
        if self.use_tc:
            dic["warp"] = self.warp
        else:
            dic["thread"] = self.thread
        dic["rstep"] = self.rstep
        if np.prod(self.reduce_thread) > 1:
            dic["reduce_thread"] = self.reduce_thread
        if self.use_tc:
            dic["use_tc"] = self.use_tc
        if self.output_strides:
            dic["strides"] = {}
            for k, stride in self.output_strides.items():
                if stride.is_valid():
                    dic["strides"][k] = stride
            if len(dic["strides"]) == 0:
                del dic["strides"]
        if np.prod(self._step) > 1:
            dic["step"] = self._step
        if self._raxis_order != []:
            dic["raxis_order"] = self._raxis_order
        if self.vectorize != {}:
            dic["vectorize"] = self.vectorize
        if self.pipeline_stage != 1:
            dic["pipeline_stage"] = self.pipeline_stage
        if self.block_reduction_depth is not None:
            dic["block_reduction_depth"] = self.block_reduction_depth
        return dic

    @classmethod
    def from_dict(cls, dic: dict) -> "Hint":
        hint = cls()
        for k, v in dic.items():
            setattr(hint, k, v)
        return hint

    def tensorcore_legalization(self):
        # only keep the last 2 axes for tensorcore
        self.warp = self.warp[-2:]
        self.block = self.block[-2:]
        return self

    @property
    def raxis_order(self) -> list[int]:
        if self._raxis_order != []:
            return self._raxis_order
        return list(range(len(self.rstep)))

    @property
    def step(self) -> list[int]:
        if self._step != []:
            return self._step
        return [1 for _ in self.block]

    def __repr__(self) -> str:
        return str(self.to_dict())

    def complete_config(self, node: PrimFuncNode):
        # analysis pass context, for int8 mma, we should merge static shared memory
        merge_static_smem = False
        # int32 and float32 accum may take too much shared memory
        if self.use_tc and self.intrin_info.out_dtype in ["float32", "int32"]:
            merge_static_smem = True
        # Always merge dynamic shared memory
        if self.shared_scope == "shared.dyn":
            merge_static_smem = True
        self.pass_context = {"tir.merge_static_smem": merge_static_smem}
        return self
