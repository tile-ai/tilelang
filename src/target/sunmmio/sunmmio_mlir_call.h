#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_CALL_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_CALL_H_

#include "codegen_sunmmio.h"
#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

class SunmmioMlirCall {
public:
  explicit SunmmioMlirCall(SunmmioMlirContext &ctx);

  SunMMIOValue Call(const std::string &result_name, const std::string &callee,
                    const std::vector<SunMMIOValue> &operands,
                    const SunMMIOCallAttrs &attrs, const std::string &category,
                    DataType ret_dtype, const SunMMIOType &ret_type);

  SunMMIOValue RegionCall(const std::string &result_name,
                          const std::string &buffer_handle,
                          const std::vector<SunMMIOValue> &mins,
                          const std::vector<int64_t> &extents,
                          DataType ret_dtype, const SunMMIOType &ret_type,
                          int64_t byte_offset = 0,
                          bool preserve_unit_dims = false);

  std::pair<SunMMIOValue, SunMMIOValue>
  MXUnpack(const std::string &scale_name, const std::string &data_name,
           const SunMMIOValue &mx, DataType scale_dtype, DataType data_dtype);

private:
  SunmmioMlirContext &ctx_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_CALL_H_
