"""Metal backend registration module."""

from tilelang.backend.module import BackendModule, register_backend_module

from . import codegen as codegen  # noqa: F401
from . import execution_backend as execution_backend  # noqa: F401
from . import pipeline as pipeline  # noqa: F401

BACKEND_MODULE = register_backend_module(BackendModule("metal", ("metal",)))
