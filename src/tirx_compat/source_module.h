#ifndef TL_TIRX_COMPAT_SOURCE_MODULE_H_
#define TL_TIRX_COMPAT_SOURCE_MODULE_H_

#include "tirx_compat/metadata.h"
#include <functional>
#include <tvm/codegen/codegen.h>
#include <unordered_map>

namespace tvm::codegen {
/*!
 * \brief Create a source module for viewing.
 * \param code The code to be viewed.
 * \param fmt The code. format.
 */
ffi::Module SourceModuleCreate(std::string code, std::string fmt);

/*!
 * \brief Create a C source module for viewing and compiling GCC code.
 * \param code The code to be viewed.
 * \param fmt The code format.
 * \param func_names The name of functions inside the runtime module.
 * \param const_vars. The constant variables that the c source module needs.
 * \return The created module.
 */
ffi::Module CSourceModuleCreate(const ffi::String &code, const ffi::String &fmt,
                                const ffi::Array<ffi::String> &func_names,
                                const ffi::Array<ffi::String> &const_vars = {});

/*!
 * \brief Wrap the submodules in a metadata module.
 * \param params The variable to constant mapping that is collected by the host
 *        module.
 * \param target_module The main TIR-lowered internal runtime module
 * \param modules All the external modules that needs to be imported inside the
 * metadata module(s).
 * \param target The target that all the modules are compiled for
 * \return The wrapped module.
 */
ffi::Module
CreateMetadataModule(const std::unordered_map<std::string, ffi::Tensor> &params,
                     ffi::Module target_module,
                     const ffi::Array<ffi::Module> &ext_modules, Target target);

/*!
 * \brief Create a source module for viewing and limited saving for device.
 * \param data The code data to be viewed.
 * \param fmt The code. format.
 * \param fmap The map function information map of each function.
 * \param type_key The type_key of the runtime module of this source code
 * \param fget_source a closure to replace default get source behavior.
 */
ffi::Module DeviceSourceModuleCreate(
    std::string data, std::string fmt,
    ffi::Map<ffi::String, runtime::FunctionInfo> fmap, std::string type_key,
    std::function<std::string(const std::string &)> fget_source = nullptr);

} // namespace tvm::codegen
#endif
