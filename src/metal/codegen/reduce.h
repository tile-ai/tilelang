/*!
 * \file metal/codegen/reduce.h
 * \brief Self-contained Metal helpers for the shared GPU reduction lowerer.
 */
#ifndef TILELANG_METAL_CODEGEN_REDUCE_H_
#define TILELANG_METAL_CODEGEN_REDUCE_H_

namespace tvm {
namespace codegen {

constexpr const char *kMetalReduceSource = R"(
namespace tl {
struct SumOp {
  template <typename T> T operator()(T x, T y) const { return x + y; }
};
struct MaxOp {
  template <typename T> T operator()(T x, T y) const { return metal::max(x, y); }
};
struct MinOp {
  template <typename T> T operator()(T x, T y) const { return metal::min(x, y); }
};
struct BitAndOp {
  template <typename T> T operator()(T x, T y) const { return x & y; }
};
struct BitOrOp {
  template <typename T> T operator()(T x, T y) const { return x | y; }
};
struct BitXorOp {
  template <typename T> T operator()(T x, T y) const { return x ^ y; }
};

template <class Reducer, int Threads, int Scale, int ThreadOffset = 0,
          int Batch = 1, int WorkspaceStride = 0>
struct AllReduce {
  template <typename T>
  static T run(T value, uint thread_index) {
    static_assert(Threads <= 32, "A cross-SIMD reduction needs shared storage");
    for (int offset = Threads / 2; offset >= Scale; offset /= 2) {
      value = Reducer()(value, simd_shuffle_xor(value, ushort(offset)));
    }
    return value;
  }

  template <typename T>
  static T run(T value, threadgroup T *workspace, uint thread_index) {
    for (int offset = Threads / 2; offset >= Scale; offset /= 2) {
      if (offset >= 32) {
        uint index = thread_index - ThreadOffset;
        workspace[index] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        value = Reducer()(value, workspace[index ^ offset]);
        // All readers must finish before another reduction reuses the workspace.
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        value = Reducer()(value, simd_shuffle_xor(value, ushort(offset)));
      }
    }
    return value;
  }

  template <typename T>
  static void run_batch(thread T *values, uint thread_index) {
    for (int i = 0; i < Batch; ++i) values[i] = run(values[i], thread_index);
  }

  template <typename T>
  static void run_batch(thread T *values, threadgroup T *workspace, uint thread_index) {
    for (int i = 0; i < Batch; ++i) {
      values[i] = run(values[i], workspace + i * WorkspaceStride, thread_index);
    }
  }
};
} // namespace tl
)";

} // namespace codegen
} // namespace tvm

#endif // TILELANG_METAL_CODEGEN_REDUCE_H_
