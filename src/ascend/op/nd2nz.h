/*!
 * \file ascend/op/nd2nz.h
 * \brief Shared geometry for compact NZ scatter and UB-to-L1 post-copy.
 */
#ifndef TVM_TL_ASCEND_OP_ND2NZ_H_
#define TVM_TL_ASCEND_OP_ND2NZ_H_

#include <cstdint>
#include <string>
#include <tvm/tirx/expr.h>

namespace tvm::tl {

inline int64_t Nd2NzElementBytes(const PrimExpr &expr) {
  const auto *dtype = expr.as<tirx::StringImmNode>();
  if (dtype && dtype->value == "float")
    return 4;
  if (dtype && (dtype->value == "half" || dtype->value == "bfloat16_t"))
    return 2;
  TVM_FFI_THROW(ValueError)
      << "ND2NZ dtype must be half, bfloat16_t, or float, got " << expr;
  return 0;
}

struct Nd2NzGeometry {
  int64_t rows, cols, src_pitch, dst_pitch, src_bytes, dst_bytes;

  bool Empty() const { return rows == 0 || cols == 0; }
  int64_t Bursts() const { return cols / (32 / dst_bytes); }
  int64_t SpanBytes(int64_t pitch) const {
    return Empty() ? 0 : ((Bursts() - 1) * pitch + rows) * 32;
  }
};

inline Nd2NzGeometry ParseNd2NzGeometry(const tirx::CallNode *op,
                                        bool scatter) {
  if (op->args.size() != 7)
    TVM_FFI_THROW(ValueError)
        << "ND2NZ expects 7 arguments with explicit pitches";
  auto integer = [&](size_t index, const char *name) {
    const auto *value = op->args[index].as<tirx::IntImmNode>();
    if (!value || value->value < 0 || value->value > INT32_MAX)
      TVM_FFI_THROW(ValueError)
          << "ND2NZ " << name << " must be a non-negative static int32";
    return value->value;
  };
  Nd2NzGeometry g{integer(2, "rows"),
                  integer(3, "cols"),
                  scatter ? 0 : integer(4, "src_pitch"),
                  integer(scatter ? 4 : 5, "dst_pitch"),
                  Nd2NzElementBytes(op->args[scatter ? 5 : 6]),
                  Nd2NzElementBytes(op->args[6])};
  if (g.Empty())
    return g;
  if (g.cols % (32 / g.dst_bytes) != 0 || g.dst_pitch < g.rows ||
      (!scatter && g.src_pitch < g.rows))
    TVM_FFI_THROW(ValueError)
        << "ND2NZ requires C0-aligned columns and pitches >= rows";
  // asc_copy_ub2l1 uses uint16_t lengths/gaps. vsstb encodes its block
  // stride in the upper 16 bits. Bound inputs before computing byte spans.
  if (g.rows > UINT16_MAX || g.Bursts() > (scatter ? UINT16_MAX : 4095) ||
      (scatter ? g.dst_pitch > UINT16_MAX
               : (g.src_pitch - g.rows > UINT16_MAX ||
                  g.dst_pitch - g.rows > UINT16_MAX)))
    TVM_FFI_THROW(ValueError)
        << "ND2NZ geometry exceeds instruction field range";
  if (scatter &&
      (g.rows % 16 != 0 ||
       (g.src_bytes == 4 && g.dst_bytes == 2 && g.cols % 64 != 0) ||
       (g.src_bytes == 2 && g.dst_bytes == 4 && g.cols % 128 != 0) ||
       (op->args[5].as<tirx::StringImmNode>()->value !=
            op->args[6].as<tirx::StringImmNode>()->value &&
        !(g.src_bytes == 4 &&
          op->args[6].as<tirx::StringImmNode>()->value == "bfloat16_t") &&
        !(g.dst_bytes == 4 &&
          op->args[5].as<tirx::StringImmNode>()->value == "bfloat16_t"))))
    TVM_FFI_THROW(ValueError)
        << "Unsupported ND2NZ scatter shape or dtype conversion";
  return g;
}

} // namespace tvm::tl
#endif // TVM_TL_ASCEND_OP_ND2NZ_H_
