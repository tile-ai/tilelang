from setuptools import setup
from mlx.extension import CMakeBuild, CMakeExtension

setup(
    name="tilelang-mlx-runtime",
    version="0.1.0",
    packages=[],
    install_requires=["mlx==0.32.4"],
    ext_modules=[CMakeExtension("_tilelang_mlx", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
)
