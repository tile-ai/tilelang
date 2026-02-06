/*!
 * \file tilescale_cuda_module.h
 * \brief TileScale extended CUDA module with distributed table initialization
 * support.
 */
#ifndef TILESCALE_RUNTIME_TILESCALE_CUDA_MODULE_H_
#define TILESCALE_RUNTIME_TILESCALE_CUDA_MODULE_H_

#include <tvm/runtime/module.h>

#include <string>
#include <unordered_map>

#include "runtime/meta_data.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Create a TileScale extended CUDA module from data.
 *
 * This module extends TVM's CUDAModule with additional functionality:
 * - __tilescale_init_distributed_table: Initialize distributed table by copying
 *   host data to the device's meta_data symbol for distributed kernels.
 *
 * \param data The module data, can be ptx, cubin
 * \param fmt The format of the data, can be "ptx", "cubin"
 * \param fmap The map function information map of each function.
 * \param cuda_source Optional, cuda source file
 */
ffi::Module
TileScaleCUDAModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source);

} // namespace runtime
} // namespace tvm

#endif // TILESCALE_RUNTIME_TILESCALE_CUDA_MODULE_H_
