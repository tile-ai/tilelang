/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file split_host_device.cc
 * \brief Split device function from host.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "common/assume.h"
#include "common/attr.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tvm/node/cast.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {
using namespace ffi;
namespace tir = tvm::tir;

// This pass traverses the AST, split the target function into host part and
// device part and copies all assume attribute statements to the device side.

// 1. Traverse AST and collect all assume statements into host_assumes_.
// 2. Until the first AttrStmtNode with tvm::attr::kTarget.
// 3. Call SplitDeviceFunc, which will create a new device function and replace
//    the original body with a call to that function.
class HostDeviceSplitter : public tir::StmtMutator {
public:
  explicit HostDeviceSplitter(IRModule *device_mod,
                              std::function<GlobalVar()> var_supply)
      : device_mod_(device_mod), var_supply_(std::move(var_supply)) {}

  void SetNonRestrictParams(Optional<Array<tir::Var>> params) {
    for (auto param : params.value()) {
      non_restrict_params_.push_back(param);
    }
  }

  void SetClusterDims(Array<Integer> cluster_dims) {
    cluster_dims_ = std::move(cluster_dims);
  }

  void SetHostFuncSignature(const tir::PrimFunc &func) {
    host_params_ = func->params;
    host_buffer_map_ = func->buffer_map;
  }

  tir::Stmt VisitStmt_(const tir::AttrStmtNode *op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      found_device_region_ = true;
      auto device_target = op->node.as<tvm::Target>().value().WithoutHost();
      return SplitDeviceFunc(op->body, device_target);
    } else if (op->attr_key == tir::attr::tilelang_assume) {
      // NOTE(chaofan): the assumes collected here must be in host-side.
      //    This is because when the collector reaches the split region,
      //    it will start to split and return. For safety, we add a check here.
      ICHECK(!found_device_region_)
          << "Assumes collection should not be in device region.";
      // We first push back the outside assume, then visit the child.
      // So when moving assumes to device side, we need to do the building
      // process in a reverse order.
      host_assumes_.push_back(op);
    }
    return tir::StmtMutator::VisitStmt_(op);
  }

  tir::Stmt VisitStmt_(const tir::EvaluateNode *op) final {
    auto stmt = GetRef<tir::Stmt>(op);
    // There should be no assume in evaluate form after InjectAssumes.
    ICHECK(!IsAssumeInEvaluateForm(stmt))
        << "Unexpected assume in evaluate form. Please run InjectAssumes pass "
           "first.";
    return tir::StmtMutator::VisitStmt_(op);
  }

  tir::Stmt ForceSplit(tir::Stmt body, tvm::Target device_target) {
    return SplitDeviceFunc(std::move(body), std::move(device_target));
  }

  bool found_device_region() const { return found_device_region_; }

private:
  bool found_device_region_{false};
  Array<tir::Var> host_params_;
  Map<tir::Var, tir::Buffer> host_buffer_map_;
  Array<tir::Var> non_restrict_params_;
  Optional<Array<Integer>> cluster_dims_{std::nullopt};
  Optional<String> code_block_source_{std::nullopt};
  Optional<String> code_block_entry_name_{std::nullopt};
  mutable Optional<String> resolved_code_block_entry_name_{std::nullopt};

  static void SortDeviceParams(std::vector<tir::Var> *params) {
    std::sort(params->begin(), params->end(),
              [](const tir::Var &a, const tir::Var &b) {
                auto sort_key = [](const tir::Var &var) {
                  return std::tuple{
                      !var->dtype.is_handle(),
                      var->name_hint,
                  };
                };
                return sort_key(a) < sort_key(b);
              });
  }

  struct ExternalKernelParam {
    std::string name;
    bool is_pointer{false};
  };

  struct ExternalKernelSignature {
    std::string name;
    std::vector<ExternalKernelParam> params;
  };

  struct SourceKernelCandidate {
    tir::Var var;
    std::vector<std::string> names;
  };

  mutable std::vector<ExternalKernelSignature>
      parsed_external_kernel_signatures_;

  static std::string Trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    auto begin = std::find_if(value.begin(), value.end(), not_space);
    auto end = std::find_if(value.rbegin(), value.rend(), not_space).base();
    if (begin >= end) {
      return "";
    }
    return std::string(begin, end);
  }

  static std::string ToLower(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
  }

  static bool IsIdentifierChar(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
  }

  static std::vector<std::string>
  SplitTopLevelCommaList(const std::string &params) {
    std::vector<std::string> fields;
    size_t field_start = 0;
    int paren_depth = 0;
    int angle_depth = 0;
    int bracket_depth = 0;

    for (size_t i = 0; i < params.size(); ++i) {
      char ch = params[i];
      switch (ch) {
      case '(':
        ++paren_depth;
        break;
      case ')':
        --paren_depth;
        break;
      case '<':
        ++angle_depth;
        break;
      case '>':
        if (angle_depth > 0)
          --angle_depth;
        break;
      case '[':
        ++bracket_depth;
        break;
      case ']':
        if (bracket_depth > 0)
          --bracket_depth;
        break;
      case ',':
        if (paren_depth == 0 && angle_depth == 0 && bracket_depth == 0) {
          fields.push_back(params.substr(field_start, i - field_start));
          field_start = i + 1;
        }
        break;
      default:
        break;
      }
    }

    fields.push_back(params.substr(field_start));
    return fields;
  }

  static ExternalKernelParam
  ParseExternalKernelParam(const std::string &param_decl) {
    std::string trimmed = Trim(param_decl);
    if (trimmed.empty() || trimmed == "void") {
      return {};
    }

    bool is_pointer = trimmed.find('*') != std::string::npos ||
                      trimmed.find('[') != std::string::npos;

    size_t cursor = trimmed.size();
    while (cursor > 0 &&
           std::isspace(static_cast<unsigned char>(trimmed[cursor - 1]))) {
      --cursor;
    }

    while (cursor > 0 && trimmed[cursor - 1] == ']') {
      int bracket_depth = 1;
      --cursor;
      while (cursor > 0 && bracket_depth > 0) {
        --cursor;
        if (trimmed[cursor] == ']') {
          ++bracket_depth;
        } else if (trimmed[cursor] == '[') {
          --bracket_depth;
        }
      }
      while (cursor > 0 &&
             std::isspace(static_cast<unsigned char>(trimmed[cursor - 1]))) {
        --cursor;
      }
    }

    size_t ident_end = cursor;
    while (ident_end > 0 && !IsIdentifierChar(trimmed[ident_end - 1])) {
      --ident_end;
    }
    size_t ident_begin = ident_end;
    while (ident_begin > 0 && IsIdentifierChar(trimmed[ident_begin - 1])) {
      --ident_begin;
    }

    ICHECK_LT(ident_begin, ident_end)
        << "Failed to parse external CUDA kernel parameter name from `"
        << param_decl << "`";

    return {
        trimmed.substr(ident_begin, ident_end - ident_begin),
        is_pointer,
    };
  }

  static std::vector<ExternalKernelSignature>
  ParseExternalKernelSignatures(const std::string &source) {
    static const std::regex kKernelPattern(
        R"((?:extern\s+"C"\s+)?__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?(\w+))");

    std::vector<ExternalKernelSignature> signatures;
    std::unordered_set<std::string> seen_names;

    for (auto it =
             std::sregex_iterator(source.begin(), source.end(), kKernelPattern);
         it != std::sregex_iterator(); ++it) {
      const std::smatch &match = *it;
      std::string kernel_name = match[1].str();
      if (!seen_names.insert(kernel_name).second) {
        continue;
      }

      size_t cursor = static_cast<size_t>(match.position(0) + match.length(0));
      while (cursor < source.size() &&
             std::isspace(static_cast<unsigned char>(source[cursor]))) {
        ++cursor;
      }
      ICHECK_LT(cursor, source.size())
          << "Malformed external CUDA kernel declaration for `" << kernel_name
          << "`";
      ICHECK_EQ(source[cursor], '(')
          << "Expected parameter list for external CUDA kernel `" << kernel_name
          << "`";

      size_t params_begin = cursor + 1;
      int paren_depth = 1;
      ++cursor;
      while (cursor < source.size() && paren_depth > 0) {
        if (source[cursor] == '(') {
          ++paren_depth;
        } else if (source[cursor] == ')') {
          --paren_depth;
        }
        ++cursor;
      }
      ICHECK_EQ(paren_depth, 0)
          << "Unterminated parameter list for external CUDA kernel `"
          << kernel_name << "`";

      std::string params_text =
          source.substr(params_begin, cursor - params_begin - 1);
      ExternalKernelSignature signature;
      signature.name = kernel_name;
      for (const std::string &field : SplitTopLevelCommaList(params_text)) {
        ExternalKernelParam param = ParseExternalKernelParam(field);
        if (!param.name.empty()) {
          signature.params.push_back(std::move(param));
        }
      }
      signatures.push_back(std::move(signature));
    }

    ICHECK(!signatures.empty()) << "T.CUDASourceCodeKernel expects external "
                                   "CUDA source to declare at least one "
                                   "__global__ kernel";
    return signatures;
  }

  const ExternalKernelSignature &
  ResolveExternalKernelSignature(const std::string &source) const {
    parsed_external_kernel_signatures_ = ParseExternalKernelSignatures(source);

    auto choose_by_name =
        [&](const std::string &kernel_name) -> const ExternalKernelSignature * {
      for (const auto &signature : parsed_external_kernel_signatures_) {
        if (signature.name == kernel_name) {
          return &signature;
        }
      }
      return nullptr;
    };

    if (code_block_entry_name_) {
      const std::string requested_name = code_block_entry_name_.value();
      const auto *signature = choose_by_name(requested_name);
      ICHECK(signature)
          << "T.CUDASourceCodeKernel expected entry_name=`" << requested_name
          << "` to match a __global__ kernel in the provided CUDA source";
      resolved_code_block_entry_name_ = String(signature->name);
      return *signature;
    }

    if (parsed_external_kernel_signatures_.size() == 1U) {
      resolved_code_block_entry_name_ =
          String(parsed_external_kernel_signatures_.front().name);
      return parsed_external_kernel_signatures_.front();
    }

    if (const auto *signature = choose_by_name("main_kernel")) {
      resolved_code_block_entry_name_ = String(signature->name);
      return *signature;
    }

    LOG(FATAL) << "T.CUDASourceCodeKernel expects external CUDA source to "
                  "either contain exactly one __global__ kernel, define a "
                  "`main_kernel` entry, or specify entry_name explicitly";
  }

  static std::string
  DescribeCandidates(const std::vector<SourceKernelCandidate> &candidates,
                     const std::vector<bool> &used) {
    std::ostringstream os;
    bool first = true;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (used[i])
        continue;
      if (!first) {
        os << ", ";
      }
      first = false;
      os << candidates[i].var->name_hint;
    }
    return first ? std::string("<none>") : os.str();
  }

  static int
  MatchExternalKernelParam(const ExternalKernelParam &param,
                           const std::vector<SourceKernelCandidate> &candidates,
                           const std::vector<bool> &used) {
    auto try_match = [&](bool case_sensitive) -> int {
      int matched_index = -1;
      std::string needle = case_sensitive ? param.name : ToLower(param.name);
      for (size_t i = 0; i < candidates.size(); ++i) {
        if (used[i])
          continue;
        for (const std::string &candidate_name : candidates[i].names) {
          std::string haystack =
              case_sensitive ? candidate_name : ToLower(candidate_name);
          if (haystack == needle) {
            if (matched_index != -1) {
              return -2;
            }
            matched_index = static_cast<int>(i);
            break;
          }
        }
      }
      return matched_index;
    };

    int exact_match = try_match(true);
    if (exact_match >= 0) {
      return exact_match;
    }
    if (exact_match == -2) {
      return exact_match;
    }

    int insensitive_match = try_match(false);
    if (insensitive_match >= 0) {
      return insensitive_match;
    }
    if (insensitive_match == -2) {
      return insensitive_match;
    }

    int unresolved_count = 0;
    int last_unresolved = -1;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (!used[i]) {
        ++unresolved_count;
        last_unresolved = static_cast<int>(i);
      }
    }

    if (unresolved_count == 1) {
      return last_unresolved;
    }
    return -1;
  }

  std::tuple<Array<tir::Var>, Array<tir::Buffer>>
  CollectSourceKernelSignature() const {
    ICHECK(code_block_source_)
        << "CollectSourceKernelSignature requires code_block_source";

    const ExternalKernelSignature &signature =
        ResolveExternalKernelSignature(code_block_source_.value());

    std::vector<SourceKernelCandidate> pointer_candidates;
    std::vector<SourceKernelCandidate> scalar_candidates;
    std::unordered_set<std::string> seen_pointer_vars;
    std::unordered_set<std::string> seen_scalar_vars;

    auto add_pointer_candidate = [&](const tir::Buffer &buf,
                                     const tir::Var &host_param) {
      const std::string &var_name = buf->data->name_hint;
      if (!seen_pointer_vars.insert(var_name).second) {
        return;
      }
      std::vector<std::string> names = {var_name, buf->name};
      if (host_param.defined()) {
        std::string host_name = host_param->name_hint;
        constexpr const char *kHandleSuffix = "_handle";
        size_t suffix_len = std::strlen(kHandleSuffix);
        if (host_name.size() > suffix_len &&
            host_name.compare(host_name.size() - suffix_len, suffix_len,
                              kHandleSuffix) == 0) {
          host_name.erase(host_name.size() - suffix_len);
        }
        names.push_back(host_name);
      }
      pointer_candidates.push_back({buf->data, names});
    };

    auto add_scalar_candidate = [&](const tir::Var &var) {
      const std::string &var_name = var->name_hint;
      if (!seen_scalar_vars.insert(var_name).second) {
        return;
      }
      scalar_candidates.push_back({var, {var_name}});
    };

    for (const auto &param : host_params_) {
      if (auto opt = host_buffer_map_.Get(param)) {
        add_pointer_candidate(opt.value(), param);
      } else {
        add_scalar_candidate(param);
      }
    }

    for (const auto &param : host_params_) {
      auto opt = host_buffer_map_.Get(param);
      if (!opt) {
        continue;
      }
      const tir::Buffer &buf = opt.value();
      auto record_symbol = [&](const PrimExpr &expr) {
        if (const auto *var = expr.as<tir::VarNode>()) {
          add_scalar_candidate(GetRef<tir::Var>(var));
        }
      };
      for (const PrimExpr &dim : buf->shape) {
        record_symbol(dim);
      }
      for (const PrimExpr &stride : buf->strides) {
        record_symbol(stride);
      }
      record_symbol(buf->elem_offset);
    }

    std::vector<bool> used_pointer(pointer_candidates.size(), false);
    std::vector<bool> used_scalar(scalar_candidates.size(), false);
    std::vector<tir::Var> params;

    for (const ExternalKernelParam &param : signature.params) {
      const auto &candidates =
          param.is_pointer ? pointer_candidates : scalar_candidates;
      const auto &used = param.is_pointer ? used_pointer : used_scalar;
      int matched_index = MatchExternalKernelParam(param, candidates, used);

      ICHECK_GE(matched_index, 0)
          << "Unable to infer binding for external CUDA kernel parameter `"
          << param.name << "` while building source-kernel ABI. Available "
          << (param.is_pointer ? "pointer" : "scalar")
          << " symbols: " << DescribeCandidates(candidates, used);

      if (param.is_pointer) {
        used_pointer[matched_index] = true;
      } else {
        used_scalar[matched_index] = true;
      }
      params.push_back(candidates[matched_index].var);
    }

    // Source kernels use the external CUDA entry as the source of truth for
    // the device ABI, so we intentionally avoid introducing extra DeclBuffer
    // carriers that would force additional shape/stride parameters into the
    // device signature.
    return {Array<tir::Var>(params.begin(), params.end()),
            Array<tir::Buffer>()};
  }

  class SourceKernelAttrExtractor : public tir::StmtMutator {
  public:
    static Stmt Extract(Stmt body, Optional<String> *code_block_source,
                        Optional<String> *code_block_entry_name) {
      SourceKernelAttrExtractor extractor(code_block_source,
                                          code_block_entry_name);
      return extractor(std::move(body));
    }

  private:
    explicit SourceKernelAttrExtractor(Optional<String> *code_block_source,
                                       Optional<String> *code_block_entry_name)
        : code_block_source_(code_block_source),
          code_block_entry_name_(code_block_entry_name) {}

    void ExtractStringAnnotation(const String &attr_key,
                                 const ffi::Any &value) {
      Optional<String> *target = nullptr;
      if (attr_key == tl::attr::kCodeBlockSource) {
        target = code_block_source_;
      } else if (attr_key == tl::attr::kCodeBlockEntryName) {
        target = code_block_entry_name_;
      } else {
        LOG(FATAL) << "Unexpected source-kernel annotation key: " << attr_key;
      }

      if (auto str = value.try_cast<String>()) {
        *target = str.value();
      } else if (auto expr = value.try_cast<PrimExpr>()) {
        if (auto str = expr.value().as<StringImmNode>()) {
          *target = str->value;
        } else {
          LOG(FATAL) << "Expected annotation `" << attr_key
                     << "` to carry a StringImm value, but got "
                     << expr.value()->GetTypeKey();
        }
      } else {
        LOG(FATAL) << "Expected annotation `" << attr_key
                   << "` to carry a StringImm value, but got "
                   << value.GetTypeKey();
      }
    }

    Stmt VisitStmt_(const tir::BlockNode *op) final {
      tir::Block block = Downcast<tir::Block>(tir::StmtMutator::VisitStmt_(op));
      bool has_source = block->annotations.count(tl::attr::kCodeBlockSource);
      bool has_entry_name =
          block->annotations.count(tl::attr::kCodeBlockEntryName);
      if (!has_source && !has_entry_name) {
        return std::move(block);
      }

      if (has_source) {
        ExtractStringAnnotation(
            tl::attr::kCodeBlockSource,
            block->annotations.Get(tl::attr::kCodeBlockSource).value());
      }
      if (has_entry_name) {
        ExtractStringAnnotation(
            tl::attr::kCodeBlockEntryName,
            block->annotations.Get(tl::attr::kCodeBlockEntryName).value());
      }

      tir::BlockNode *block_ptr = block.CopyOnWrite();
      if (has_source) {
        block_ptr->annotations.erase(tl::attr::kCodeBlockSource);
      }
      if (has_entry_name) {
        block_ptr->annotations.erase(tl::attr::kCodeBlockEntryName);
      }
      return std::move(block);
    }

    Stmt VisitStmt_(const tir::AttrStmtNode *op) final {
      if (op->attr_key == tl::attr::kCodeBlockSource ||
          op->attr_key == tl::attr::kCodeBlockEntryName) {
        ExtractStringAnnotation(op->attr_key, op->value);
        return VisitStmt(op->body);
      }
      return tir::StmtMutator::VisitStmt_(op);
    }

    Optional<String> *code_block_source_;
    Optional<String> *code_block_entry_name_;
  };

  // Wrap body with assumes, substituting variables in assumes with the
  // corresponding variables in the device body based on name_hint matching.
  // This substitution is necessary because host-side assume variables may be
  // different Var objects from device-side parameters, even if they have the
  // same name. We always perform substitution to ensure ConvertSSA sees
  // consistent variable references.
  Stmt wrapBodyWithHostSideAssumes(
      Stmt body, const std::unordered_map<std::string, tir::Var> &name_to_var) {
    // Build substitution map: assume_var -> body_var
    // Always substitute if we find a matching name, regardless of whether
    // it's the same object. This ensures ConvertSSA treats them as the same
    // variable.
    auto substitute_func =
        [&name_to_var](const tir::Var &var) -> Optional<PrimExpr> {
      auto it = name_to_var.find(var->name_hint);
      if (it != name_to_var.end()) {
        return it->second;
      }
      return Optional<PrimExpr>();
    };

    for (auto it = host_assumes_.rbegin(); it != host_assumes_.rend(); ++it) {
      // Substitute variables in the assume condition
      PrimExpr original_node = Downcast<PrimExpr>((*it)->node);
      PrimExpr substituted_node =
          tir::Substitute(original_node, substitute_func);
      body = AttrStmt(substituted_node, tir::attr::tilelang_assume,
                      (*it)->value, body);
    }
    return body;
  }

  tir::Stmt SplitDeviceFunc(tir::Stmt body, tvm::Target device_target) {
    code_block_source_ = std::nullopt;
    code_block_entry_name_ = std::nullopt;
    resolved_code_block_entry_name_ = std::nullopt;
    body = SourceKernelAttrExtractor::Extract(
        std::move(body), &code_block_source_, &code_block_entry_name_);

    // Normal kernels infer device parameters from use-def of the device body.
    // Source kernels have no meaningful DSL body, so their device signature
    // must be reconstructed explicitly from the host PrimFunc signature and
    // buffer metadata.
    auto [old_params, buffers_to_declare] =
        [&]() -> std::tuple<Array<tir::Var>, Array<tir::Buffer>> {
      if (code_block_source_) {
        return CollectSourceKernelSignature();
      }

      tir::VarUseDefAnalyzer use_def(/*defined_vars=*/{},
                                     /*visit_thread_extent=*/true);
      use_def(body);

      std::vector<tir::Var> params{use_def.undefined_.begin(),
                                   use_def.undefined_.end()};
      SortDeviceParams(&params);
      return {Array<tir::Var>(params.begin(), params.end()),
              use_def.undefined_buffers_};
    }();

    // Create new parameter variables for the device function to avoid sharing
    // Var objects with the host function. This prevents ConvertSSA from
    // incorrectly renaming variables when it processes multiple functions.
    Array<tir::Var> params;
    Map<tir::Var, PrimExpr> var_remap;
    std::unordered_map<std::string, tir::Var> name_to_var;
    for (const auto &old_var : old_params) {
      tir::Var new_var(old_var->name_hint, old_var->type_annotation);
      params.push_back(new_var);
      var_remap.Set(old_var, new_var);
      name_to_var[old_var->name_hint] = new_var;
    }

    // Substitute old variables with new ones in the body
    body = tir::Substitute(body, var_remap);

    // Also remap buffers to use new variables
    Array<tir::Buffer> new_buffers_to_declare;
    for (const auto &buf : buffers_to_declare) {
      auto new_shape = buf->shape.Map(
          [&](const PrimExpr &e) { return tir::Substitute(e, var_remap); });
      auto new_strides = buf->strides.Map(
          [&](const PrimExpr &e) { return tir::Substitute(e, var_remap); });
      auto new_elem_offset = tir::Substitute(buf->elem_offset, var_remap);
      auto new_data = var_remap.count(buf->data)
                          ? Downcast<tir::Var>(var_remap[buf->data])
                          : buf->data;
      tir::Buffer new_buf(new_data, buf->dtype, new_shape, new_strides,
                          new_elem_offset, buf->name, buf->data_alignment,
                          buf->offset_factor, buf->buffer_type,
                          buf->axis_separators, buf->span);
      new_buffers_to_declare.push_back(new_buf);
    }
    buffers_to_declare = new_buffers_to_declare;

    // CodeGenCPU is used for some device-side targets, such as
    // "ext_dev", and expects to be able to return a int32_t status
    // code.

    bool can_propagate_errors = [&]() {
      auto kind = device_target->GetTargetDeviceType();
      return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
    }();
    IntImm success(DataType::Int(32), 0);
    Type kernel_ret_type;
    if (can_propagate_errors) {
      kernel_ret_type = PrimType(DataType::Int(32));
      body = tir::SeqStmt::Flatten(body, tir::Evaluate(ret(success)));
    } else {
      kernel_ret_type = VoidType();
    }

    // Declare necessary buffers for the device side.
    for (tir::Buffer buf : buffers_to_declare) {
      body = tir::DeclBuffer(buf, std::move(body));
    }

    // Copy assumes from host-side to device-side, with variable substitution.
    // This must be done after DeclBuffer so that assumes are at the outermost
    // level of the function body. This ensures ConvertSSA correctly identifies
    // that assume variables refer to function parameters.
    body = wrapBodyWithHostSideAssumes(body, name_to_var);

    // Remap non_restrict_params to use new parameter variables
    Array<tir::Var> remapped_non_restrict_params;
    for (const auto &old_var : non_restrict_params_) {
      if (var_remap.count(old_var)) {
        remapped_non_restrict_params.push_back(
            Downcast<tir::Var>(var_remap[old_var]));
      } else {
        remapped_non_restrict_params.push_back(old_var);
      }
    }

    tir::PrimFunc device_func(params, body, kernel_ret_type);
    Map<String, ffi::Any> device_attrs = {
        {tvm::attr::kTarget, device_target},
        {tir::attr::kNoAlias, true},
        {tir::attr::kIsGlobalFunc, true},
        {tl::attr::kNonRestrictParams, remapped_non_restrict_params}};
    if (cluster_dims_.defined()) {
      device_attrs.Set("cluster_dims", cluster_dims_.value());
    }
    if (code_block_source_) {
      device_attrs.Set(tl::attr::kCodeBlockSource, code_block_source_.value());
      ICHECK(resolved_code_block_entry_name_)
          << "T.CUDASourceCodeKernel expects SplitHostDevice to resolve an "
             "external CUDA entry name before building the device PrimFunc";
      device_attrs.Set(tvm::attr::kGlobalSymbol,
                       resolved_code_block_entry_name_.value());
    }
    if (code_block_entry_name_) {
      device_attrs.Set(tl::attr::kCodeBlockEntryName,
                       code_block_entry_name_.value());
    }
    device_func = WithAttrs(std::move(device_func), device_attrs);

    GlobalVar kernel_symbol_global = var_supply_();
    (*device_mod_)->Add(kernel_symbol_global, device_func);
    // Use old_params as call arguments (host-side variables)
    Array<PrimExpr> args =
        old_params.Map([](const tir::Var &var) -> PrimExpr { return var; });

    if (can_propagate_errors) {
      tir::Var kernel_error_code("kernel_error_code", success->dtype);
      tir::Call kernel_call(success->dtype, kernel_symbol_global, args);
      tir::AssertStmt assert_success(
          kernel_error_code == success,
          tir::StringImm("Error executing compute kernel"), tir::Evaluate(0));
      tir::LetStmt let_check(kernel_error_code, kernel_call, assert_success);

      return let_check;

    } else {
      return tir::Evaluate(
          tir::Call(DataType::Void(), kernel_symbol_global, args));
    }
  }

  // target ir module
  IRModule *device_mod_;
  // Generate new GlobalVar for the kernel
  std::function<GlobalVar()> var_supply_;
  // Collect assumes in host side
  Array<const tir::AttrStmtNode *> host_assumes_;
};

tir::PrimFunc SplitHostDevice(tir::PrimFunc func, IRModule *device_mod,
                              std::function<GlobalVar()> var_supply) {
  HostDeviceSplitter splitter(device_mod, std::move(var_supply));
  splitter.SetHostFuncSignature(func);
  // Propagate non-restrict parameter list from host func to device kernels
  if (auto opt = func->GetAttr<Array<tir::Var>>(tl::attr::kNonRestrictParams)) {
    splitter.SetNonRestrictParams(opt.value());
    // Remove the attribute from host-side PrimFunc; it only matters for device
    // codegen.
    func = tvm::WithoutAttr(std::move(func), tl::attr::kNonRestrictParams);
  }
  // Propagate cluster_dims from host func to device kernel.
  // LowerOpaqueBlock sets this attr on the pre-split kernel; after splitting
  // it must live on the device side so the codegen can emit a cluster launch.
  if (auto opt = func->GetAttr<Array<Integer>>("cluster_dims")) {
    splitter.SetClusterDims(opt.value());
    func = tvm::WithoutAttr(std::move(func), "cluster_dims");
  }

  if (auto body = splitter(func->body); !body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  } else if (!splitter.found_device_region()) {
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      auto device_target = target.value().WithoutHost();
      if (device_target.defined() &&
          func->HasNonzeroAttr(tir::attr::kIsEntryFunc) &&
          tir::is_no_op(func->body)) {
        if (auto forced = splitter.ForceSplit(func->body, device_target);
            !forced.same_as(func->body)) {
          func.CopyOnWrite()->body = forced;
        }
      }
    }
  }
  return func;
}

namespace transform {

tvm::transform::Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    tvm::GlobalVarSupply global_var_supply(mod);

    IRModule device_mod = IRModule(Map<GlobalVar, BaseFunc>({}));
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));

    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<tir::PrimFunc>()) {
        tir::PrimFunc func = opt.value();

        auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        auto name_prefix = global_symbol.value_or(gvar->name_hint);
        auto kernel_name = name_prefix + "_kernel";
        auto var_supply = [&global_var_supply, &kernel_name]() -> GlobalVar {
          return global_var_supply->FreshGlobal(kernel_name, false);
        };

        func = ::tvm::tl::SplitHostDevice(std::move(func), &device_mod,
                                          var_supply);
        if (!func.same_as(base_func)) {
          updates->Add(gvar, func);
        }
      }
    }
    mod->Update(updates);
    mod->Update(device_mod);
    return tir::transform::ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tl.SplitHostDevice",
                                          {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SplitHostDevice", SplitHostDevice);
}

} // namespace transform
} // namespace tl
} // namespace tvm
