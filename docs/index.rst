👋 Welcome to Tile Language
===========================

`GitHub <https://github.com/tile-ai/tilelang>`_

Tile Language (tile-lang) is a concise domain-specific language designed to streamline 
the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). 
By employing a Pythonic syntax with an underlying compiler infrastructure on top of TVM, 
tile-lang allows developers to focus on productivity without sacrificing the 
low-level optimizations necessary for state-of-the-art performance.

.. toctree::
   :maxdepth: 2
   :caption: GET STARTED

   get_started/Installation.rst
   get_started/overview.rst

.. toctree::
   :maxdepth: 1
   :caption: TUTORIALS

   tutorials/writing_kernels_with_tilelibrary.rst
   tutorials/writint_kernels_with_thread_primitives.rst
   tutorials/annotate_memory_layout.rst
   tutorials/debug_tools_for_tilelang.rst
   tutorials/auto_tuning.rst
   tutorials/jit_compilation.rst
   tutorials/pipelining_computations_and_data_movements.rst


.. toctree::
   :maxdepth: 1
   :caption: DEEP LEARNING OPERATORS

   deeplearning_operators/elementwise.rst
   deeplearning_operators/gemv.rst
   deeplearning_operators/matmul.rst
   deeplearning_operators/matmul_dequant.rst
   deeplearning_operators/flash_attention.rst
   deeplearning_operators/flash_linear_attention.rst
   deeplearning_operators/convolution.rst
   deeplearning_operators/tmac_gpu.rst

.. toctree::
   :maxdepth: 2
   :caption: LANGUAGE REFERENCE

   language_ref/ast.rst
   language_ref/primitives.rst
   language_ref/tilelibrary.rst
   


.. toctree::
   :maxdepth: 1
   :caption: Privacy

   privacy.rst
