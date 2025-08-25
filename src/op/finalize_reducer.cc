/*!
 * \file src/op/finalize_reducer.cc
 *
 * Define finalize_reducer operator.
 */

#include "finalize_reducer.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

FinalizeReducer::FinalizeReducer(Array<PrimExpr> args, BufferMap vmap)
    : reducer_(vmap[GetVarFromAccessPtr(args[0])]),
      op_((ReducerOpType)*as_const_int(args[1])) {}

Stmt FinalizeReducer::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  auto buffer = T.buffer_remap[reducer_];
  auto opt_layout = T.layout_map.Get(reducer_);
  ICHECK(opt_layout);
  ICHECK(opt_layout->as<Fragment>());
  auto layout = opt_layout->as<Fragment>().value();
  Array<PrimExpr> indices_0;
  indices_0.reserve(layout->OutputDim());
  for (int i = 0; i < layout->OutputDim(); ++i)
    indices_0.push_back(Var("__finred_" + std::to_string(i)));

  const int64_t *p_extent = as_const_int(layout->ReplicateExtent());
  ICHECK(p_extent);
  int extent = *p_extent, scale = 1;
  ICHECK(extent == 1 || extent == *as_const_int(T.thread_bounds->extent))
      << "Illegal finalize_reducer: extent=" << extent
      << "; T.thread_bounds=" << T.thread_bounds;

  if (extent == 1)
    return Evaluate(0);

  std::array op_names{"tl::SumOp", "tl::MaxOp", "tl::MinOp"};
  auto op_str = op_names[(int)op_];

  // adopted from ReduceOp
  int reducing_threads = extent;
  std::stringstream ss;
  auto thread_offset = T.thread_bounds->min;
  if (TargetIsHopper(T.target) || TargetIsSm100(T.target)) {
    auto all_threads = T.thread_bounds->extent;
    ss << "tl::AllReduce<" << op_str << ", " << reducing_threads << ", " << 1
       << ", " << thread_offset << ", " << all_threads << ">::run_hopper";
  } else {
    ss << "tl::AllReduce<" << op_str << ", " << reducing_threads << ", " << 1
       << ", " << thread_offset << ">::run";
  }
  Array<PrimExpr> thread_reduce_args = {StringImm(ss.str()),
                                        BufferLoad(buffer, indices_0)};
  if (reducing_threads >= 32) {
    PrimExpr workspace =
        T.AddWorkspace(*as_const_int(T.thread_bounds->extent), buffer->dtype);
    thread_reduce_args.push_back(workspace);
  }
  auto call = Call(buffer->dtype, builtin::call_extern(), thread_reduce_args);
  Stmt body = BufferStore(buffer, call, indices_0);

  // make the outer spatial loop
  for (int i = layout->OutputDim() - 1; i >= 0; i--) {
    body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
               ForKind::kParallel, body);
  }

  return body;
}

TIR_REGISTER_TL_OP(FinalizeReducer, finalize_reducer)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm
