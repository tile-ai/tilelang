#include "support/check.h"
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

/*!
 * \file annotate_layout.cc
 * \brief Implements the AnnotateLayout schedule primitive for tilelang.
 *
 * Annotates a buffer (identified by name) with a Layout object by
 * adding a "layout_map" entry to the specified block's annotations.
 * The LayoutInference pass reads this annotation as the initial seed
 * for BFS-based layout propagation.
 */

#include <tvm/tirx/stmt_functor.h>

#include "s_tir/schedule/transform.h"
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/utils.h>

#include "../../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using namespace tvm::s_tir;

// ---------------------------------------------------------------------------
// Helper: walk the IR tree to find a buffer by name
// ---------------------------------------------------------------------------
struct BufferFinder : public StmtVisitor {
  String target_name;
  Optional<Buffer> found;

  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &buf : op->alloc_buffers) {
      if (buf->name == target_name) {
        found = buf;
        return;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
};

// ---------------------------------------------------------------------------
// AnnotateLayout: main entry point
// ---------------------------------------------------------------------------
static void AnnotateLayout(s_tir::ScheduleState self,
                           const StmtSRef &block_sref,
                           const String &buffer_name, const Layout &layout) {
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);

  // Walk the block's subtree to find the buffer by name.
  BufferFinder finder;
  finder.target_name = buffer_name;
  finder.VisitStmt_(block);

  ICHECK(finder.found.defined())
      << "AnnotateLayout: buffer \"" << buffer_name
      << "\" not found in alloc_buffers of the block subtree";

  Buffer target_buf = finder.found.value();

  // Build or update the layout_map annotation (Map<Var, Layout>).
  ffi::Map<Var, Layout> layout_map;
  if (block->annotations.count(attr::kLayoutMap)) {
    layout_map = block->annotations.Get(attr::kLayoutMap)
                     ->as<ffi::Map<Var, Layout>>()
                     .value();
  }
  layout_map.Set(target_buf->data, layout);

  // Replace the block with updated annotations.
  tvm::ffi::ObjectPtr<SBlockNode> new_block =
      ffi::make_object<SBlockNode>(*block);
  new_block->annotations.Set(attr::kLayoutMap, layout_map);

  ffi::Map<SBlock, SBlock> block_sref_reuse;
  block_sref_reuse.Set(ffi::GetRef<SBlock>(block), SBlock(new_block));
  self->Replace(block_sref, SBlock(new_block), block_sref_reuse);
}

// ---------------------------------------------------------------------------
// FFI Registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.schedule.ScheduleAnnotateLayout",
                        [](s_tir::Schedule self, const SBlockRV &block_rv,
                           const String &buffer_name, const Layout &layout) {
                          AnnotateLayout(self->state(), self->GetSRef(block_rv),
                                         buffer_name, layout);
                        });
}

} // namespace tl
} // namespace tvm
