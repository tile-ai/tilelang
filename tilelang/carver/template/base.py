# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from ..arch import TileDevice, is_volta_arch, is_ampere_arch, is_cdna_arch, auto_infer_current_arch
from ..roller import Hint
from typing import List
from tvm.tir import PrimFunc


@dataclass
class BaseTemplate(ABC):

    _arch: TileDevice = field(default=auto_infer_current_arch(), init=False, repr=False)

    _func: PrimFunc = field(default=None, init=False, repr=False)

    @abstractmethod
    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10) -> List[Hint]:
        pass

    def with_arch(self, arch: TileDevice) -> "BaseTemplate":
        self._arch = arch
        return self

    def has_arch(self) -> bool:
        return self._arch is not None

    def is_volta_arch(self) -> bool:
        return is_volta_arch(self._arch) if self._arch is not None else False

    def is_ampere_arch(self) -> bool:
        return is_ampere_arch(self._arch) if self._arch is not None else False

    def is_cdna_arch(self) -> bool:
        return is_cdna_arch(self._arch) if self._arch is not None else False

    def equivalent_function(self) -> PrimFunc:
        return self._func

    def initialize_function(self) -> None:
        raise NotImplementedError("initialize_function is not implemented")

    def set_function(self, func: PrimFunc) -> "BaseTemplate":
        self._func = func
        return self

    def recommend_hints(self, topk: int = 10) -> List[Hint]:
        return self.get_hardware_aware_configs(self._arch, topk)

    @property
    def arch(self) -> TileDevice:
        return self._arch

    def __post_init__(self):
        self.initialize_function()
