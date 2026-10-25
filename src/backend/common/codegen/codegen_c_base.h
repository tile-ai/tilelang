#ifndef TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_
#define TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_

#include <tvm/codegen/c.h>

#include <sstream>
#include <string>

namespace tvm::codegen {

class CodeGenTileLangCBase : public CodeGenC {
public:
  void SetEmitLineDirectives(bool enable) { emit_line_directives_ = enable; }

  std::string Finish() final {
    std::ostringstream code;
    code << preamble_stream_.str();
    EmitPreamble_(code);
    code << CodeGenC::Finish();
    return code.str();
  }

protected:
  virtual void EmitPreamble_(std::ostream &) const {}
  std::ostringstream preamble_stream_;
};

} // namespace tvm::codegen

#endif
