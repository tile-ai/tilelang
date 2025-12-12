# Import necessary modules and classes
from abc import ABC, abstractmethod  # For defining abstract base classes
from dataclasses import dataclass, field  # For defining data classes
from ..arch import (  # Import architecture-related utilities and classes
    TileDevice,
    is_volta_arch,
    is_ampere_arch,
    is_cdna_arch,
    auto_infer_current_arch,
)
from ..roller.hint import Hint  # Import the Hint class
from ..roller.node import OutputNode  # Import the OutputNode class
from tvm.tir import PrimFunc  # Import PrimFunc for handling tensor IR functions


@dataclass
class BaseTemplate(ABC):
    """
    Base class template for hardware-aware configurations.
    This serves as an abstract base class (ABC) that defines the structure
    for subclasses implementing hardware-specific optimizations.
    """

    # The architecture of the device, inferred automatically unless explicitly set
    _arch: TileDevice = field(default=auto_infer_current_arch(), init=False, repr=False)

    # The function associated with this template, initially None
    _func: PrimFunc = field(default=None, init=False, repr=False)

    # The outputs nodes associated with this template, initially None
    _output_nodes: list[OutputNode] = field(default=None, init=False, repr=False)

    @abstractmethod
    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10) -> list[Hint]:
        """
        Abstract method that must be implemented by subclasses.
        It should return a list of hardware-aware configurations (hints)
        based on the specified architecture.

        Args:
            arch (TileDevice, optional): The target architecture. Defaults to None.
            topk (int, optional): Number of top configurations to return. Defaults to 10.

        Returns:
            List[Hint]: A list of recommended hardware-aware configurations.
        """
        pass

    def with_arch(self, arch: TileDevice) -> "BaseTemplate":
        """
        Sets the architecture for this template and returns itself.

        Args:
            arch (TileDevice): The architecture to set.

        Returns:
            BaseTemplate: The instance with the updated architecture.
        """
        self._arch = arch
        return self

    def has_arch(self) -> bool:
        """
        Checks whether the architecture is set.

        Returns:
            bool: True if the architecture is set, False otherwise.
        """
        return self._arch is not None

    def is_volta_arch(self) -> bool:
        """
        Checks if the current architecture is a Volta architecture.

        Returns:
            bool: True if the architecture is Volta, False otherwise.
        """
        return is_volta_arch(self._arch) if self._arch is not None else False

    def is_ampere_arch(self) -> bool:
        """
        Checks if the current architecture is an Ampere architecture.

        Returns:
            bool: True if the architecture is Ampere, False otherwise.
        """
        return is_ampere_arch(self._arch) if self._arch is not None else False

    def is_cdna_arch(self) -> bool:
        """
        Checks if the current architecture is a CDNA architecture.

        Returns:
            bool: True if the architecture is CDNA, False otherwise.
        """
        return is_cdna_arch(self._arch) if self._arch is not None else False

    def equivalent_function(self) -> PrimFunc:
        """
        Returns the function associated with this template.

        Returns:
            PrimFunc: The stored function.
        """
        return self._func

    def initialize_function(self) -> None:
        """
        Placeholder method that should be implemented by subclasses.
        This method is responsible for initializing the function.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("initialize_function is not implemented")

    def set_function(self, func: PrimFunc) -> "BaseTemplate":
        """
        Sets the function for this template and returns itself.

        Args:
            func (PrimFunc): The function to associate with this template.

        Returns:
            BaseTemplate: The instance with the updated function.
        """
        self._func = func
        return self

    def set_output_nodes(self, output_nodes: list[OutputNode]) -> "BaseTemplate":
        """
        Sets the output nodes for this template and returns itself.

        Args:
            output_nodes (List[OutputNode]): The output nodes to associate with this template.

        Returns:
            BaseTemplate: The instance with the updated output nodes.
        """
        self._output_nodes = output_nodes
        return self

    def recommend_hints(self, topk: int = 10) -> list[Hint]:
        """
        Provides a list of recommended hardware-aware configurations.

        Args:
            topk (int, optional): Number of top configurations to return. Defaults to 10.

        Returns:
            List[Hint]: A list of recommended configurations.
        """
        return self.get_hardware_aware_configs(self._arch, topk)

    @property
    def arch(self) -> TileDevice:
        """
        Returns the current architecture.

        Returns:
            TileDevice: The architecture of this template.
        """
        return self._arch

    @property
    def output_nodes(self) -> list[OutputNode]:
        """
        Returns the output nodes associated with this template.

        Returns:
            List[OutputNode]: The output nodes.
        """
        return self._output_nodes

    def __post_init__(self):
        """
        Post-initialization method that is called after the data class is created.
        Ensures that the function is initialized.
        """
        self.initialize_function()
