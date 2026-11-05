/*!
 * \file output_module.h
 * \brief Serializable public entries for callee-allocated native kernels.
 */
#ifndef TVM_TL_RUNTIME_OUTPUT_MODULE_H_
#define TVM_TL_RUNTIME_OUTPUT_MODULE_H_

#include "output_frame.h"

#include <tvm/ffi/extra/module.h>

#include <string>
#include <vector>

namespace tvm::tl {

inline constexpr const char *kOutputModuleSchemaAttr =
    "tl.output_module_schema";
struct OutputEntrySpec {
  std::string public_name;
  std::string allocate_name;
  std::string compute_name;
  int parameter_count;
  int device_type;
  bool is_main;
  std::vector<OutputSlotSpec> outputs;
};

/*! \brief Encode only static ABI data; expressions remain in typed workers. */
ffi::Bytes EncodeOutputSchema(const std::vector<OutputEntrySpec> &entries);
/*! \brief Validate schema and wrap one compiler module, retaining its imports.
 */
ffi::Module MakeOutputModule(ffi::Module worker, ffi::Bytes schema);

} // namespace tvm::tl
#endif // TVM_TL_RUNTIME_OUTPUT_MODULE_H_
