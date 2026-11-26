/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/transform.h>

#include "auto_schedule/kernel_rewriter.h"
#include "core_usage.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ascend;
namespace {
bool IsPTOTarget(const PrimFunc &func) {
  if (auto target = func->GetAttr<Target>(tvm::attr::kTarget))
    for (const auto &key : target.value()->keys)
      if (key == "pto")
        return true;
  return false;
}

} // namespace

tvm::transform::Pass LowerAscendCoreRegions() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    bool pto = IsPTOTarget(func);
    func = RewriteTilelangKernels(
        std::move(func), "LowerAscendCoreRegions",
        [pto](const TilelangKernelContext &context) {
          SBlock root = context.root;
          Stmt input = root->body;
          AscendCoreUsage usage = AscendCoreUsageAnalyzer::Analyze(root);
          bool has_cube = usage.explicit_cores & kCoreCube;
          bool has_vector = usage.explicit_cores & kCoreVector;
          ICHECK(!usage.uses_ss || !pto)
              << "PTO does not support T.alloc_ss/shared.ss scalar accesses; "
                 "use "
                 "the AscendC backend, or redesign communication using "
                 "supported "
                 "storage and explicit cross-core synchronization.";
          ICHECK(!usage.uses_ss || !usage.vector_count.has_value() ||
                 usage.vector_count.value()->value == 2)
              << "T.alloc_ss shared communication requires two active AIVs "
                 "(sids=2)";
          CoreMask streams = usage.execution_cores | usage.explicit_cores;
          if (!streams)
            streams =
                usage.scalar_candidates == kCoreCube ? kCoreCube : kCoreVector;
          bool grouped = usage.vector_count.has_value();
          int count = grouped ? usage.vector_count.value()->value : 2;
          bool mixed = usage.requires_mix || streams == kCoreAll ||
                       (!pto && grouped && (streams & kCoreVector));
          Stmt body = input;
          if (usage.requires_mix || streams == kCoreAll || has_cube ||
              has_vector) {
            ICHECK(!usage.unplaced_work)
                << "Manual core placement is incomplete "
                   "(tl.enable_auto_schedule=False): "
                << (has_cube || has_vector
                        ? "executable work remains outside explicit core "
                          "regions. "
                        : "mixed execution or cross-core hardware "
                          "participation "
                          "requires explicit core regions. ")
                << "Put all computation, memory accesses and synchronization "
                   "inside the appropriate T.Cube/T.Vector region. Example:\n"
                   "  with T.Cube():\n"
                   "    # Cube operations and Cube-side synchronization\n"
                   "    ...\n"
                   "  with T.Vector(G):\n"
                   "    # Vector operations and Vector-side synchronization\n"
                   "    ...\n"
                   "Use the same G as MixedKernel(sids=G), if specified. "
                   "Alternatively set tl.enable_auto_schedule=True. "
                   "Pure flat Cube/Vector programs remain supported.";
          } else {
            // Pure flat programs need only one whole-body region, not
            // scheduling.
            ffi::Map<ffi::String, ffi::Any> annotations;
            if (streams == kCoreVector && mixed)
              annotations.Set("vector_count", Integer(count));
            body = SBlock({}, {}, {}, streams == kCoreCube ? "CUBE" : "VECTOR",
                          body, std::nullopt, {}, {}, annotations);
            has_cube = streams == kCoreCube;
            has_vector = streams == kCoreVector;
          }
          // Only add the missing hardware participant. Existing sections and
          // their lexical containers stay in place.
          if (mixed && !has_cube)
            body = SeqStmt::Flatten(SBlock({}, {}, {}, "CUBE", Evaluate(0),
                                           std::nullopt, {}, {}, {}),
                                    body);
          if (mixed && !has_vector)
            body = SeqStmt::Flatten(
                body, SBlock({}, {}, {}, "VECTOR", Evaluate(0), std::nullopt,
                             {}, {}, {{"vector_count", Integer(count)}}));
          auto *writer = root.CopyOnWrite();
          writer->body = body;
          if (usage.vector_count.has_value())
            writer->annotations.Set("vector_count", usage.vector_count.value());
          return root;
        },
        false);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.LowerAscendCoreRegions", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerAscendCoreRegions",
                        LowerAscendCoreRegions);
}
} // namespace tl
} // namespace tvm
