/*!
 * \file tl/backend/common/op/arg_reduce.h
 * \brief Joint GPU arg reduction and shared-memory redistribution.
 */
#ifndef TVM_TL_BACKEND_COMMON_OP_ARG_REDUCE_H_
#define TVM_TL_BACKEND_COMMON_OP_ARG_REDUCE_H_

#include "backend/common/op/reduce.h"
#include "op/arg_reduce.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {
namespace backend {
namespace arg_reduce {

inline Array<PrimExpr> Shape(const BufferRegion &region) {
  arith::Analyzer analyzer;
  return region->region.Map(
      [&](const Range &r) { return analyzer.Simplify(r->extent); });
}

inline int64_t Size(const Array<PrimExpr> &shape) {
  int64_t n = 1;
  for (const auto &s : shape) {
    const auto *constant = as_const_int(s);
    CHECK(constant != nullptr, ValueError)
        << "arg_reduce requires static tile extents";
    int64_t extent = *constant;
    CHECK(extent > 0 && n <= std::numeric_limits<int>::max() / extent,
          ValueError)
        << "arg_reduce tile is too large for GPU workspace indexing";
    n *= extent;
  }
  return n;
}

inline Array<PrimExpr> Unflatten(PrimExpr flat, const Array<PrimExpr> &shape) {
  Array<PrimExpr> indices(shape.size(), PrimExpr());
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    indices.Set(i, FloorMod(flat, shape[i]));
    flat = FloorDiv(flat, shape[i]);
  }
  return indices;
}

inline Array<PrimExpr> Offset(Array<PrimExpr> indices,
                              const BufferRegion &region) {
  for (size_t i = 0; i < indices.size(); ++i)
    indices.Set(i, indices[i] + region->region[i]->min);
  return indices;
}

inline PrimExpr Address(const Buffer &buffer, const Array<PrimExpr> &indices) {
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(buffer, indices)});
}

inline Stmt Update(const Buffer &values, const Buffer &indices,
                   const Array<PrimExpr> &position, PrimExpr value,
                   PrimExpr index, bool is_max) {
  return Evaluate(Call(DataType::Handle(), tl::arg_reduce_update(),
                       {Address(values, position), Address(indices, position),
                        value, cast(indices->dtype, index), Bool(is_max)}));
}

inline Stmt Parallel(const Array<PrimExpr> &vars, const Array<PrimExpr> &shape,
                     Stmt body, const Fragment &layout, const LowerArgs &args,
                     arith::Analyzer *analyzer) {
  for (int i = static_cast<int>(vars.size()) - 1; i >= 0; --i) {
    body = For(Downcast<Var>(vars[i]), 0, shape[i], ForKind::kParallel, body);
  }
  return PragmaUnrollLoop(PartitionLoop(Downcast<For>(body), args.thread_index,
                                        analyzer, layout, true));
}

inline Array<PrimExpr> Variables(const Array<PrimExpr> &shape) {
  Array<PrimExpr> vars;
  for (size_t i = 0; i < shape.size(); ++i)
    vars.push_back(Var("arg_i" + std::to_string(i)));
  return vars;
}

inline PrimExpr ThreadBase(const Fragment &layout) {
  return layout->ThreadRange().defined() ? layout->ThreadRange()->min
                                         : Integer(0);
}

// Validate the *physical* XOR topology, including padded fibers. A power-of-two
// logical reduction extent alone says nothing about which fiber a peer owns.
inline bool CheckTopology(const Fragment &layout,
                          const std::vector<reduce::ThreadReduceStep> &steps,
                          int threads, int warp_size) {
  for (const auto &step : steps) {
    int width = step.ReducingThreads();
    if ((width & (width - 1)) || (step.scale & (step.scale - 1)) ||
        threads % width != 0)
      return false;
  }
  int64_t slots = Size(layout->OutputShape());
  // Bound compile-time enumeration; large or exotic layouts use redistribution.
  if (slots * threads > 65536)
    return false;
  auto inverse = layout->InverseWithLevel(true).first;
  std::vector<std::vector<int64_t>> fibers(slots * threads);
  for (int64_t s = 0; s < slots; ++s) {
    for (int t = 0; t < threads; ++t) {
      auto physical = Unflatten(Integer(s), layout->OutputShape());
      physical.push_back(Integer(t));
      auto logical = inverse->Forward(physical);
      bool valid = true;
      for (size_t i = 0; i < logical.size(); ++i) {
        auto v = EvaluateConstantInteger(logical[i]);
        if (!v)
          return false;
        int64_t extent =
            *as_const_int(i < layout->InputDim() ? layout->InputShape()[i]
                                                 : layout->ReplicateExtent());
        valid &= *v >= 0 && *v < extent;
        if (i < layout->InputDim())
          fibers[s * threads + t].push_back(*v);
      }
      PrimExpr replica = logical.back();
      logical.pop_back();
      auto owner =
          EvaluateConstantInteger(layout->ForwardThread(logical, replica));
      if (!owner)
        return false;
      valid &= *owner == t;
      if (!valid)
        fibers[s * threads + t].clear();
    }
  }
  for (int64_t s = 0; s < slots; ++s) {
    for (int t = 0; t < threads; ++t) {
      for (const auto &step : steps) {
        if (step.ReducingThreads() > warp_size &&
            fibers[s * threads + t].empty())
          return false;
        for (int delta = step.scale; delta < step.ReducingThreads();
             delta *= 2) {
          if (fibers[s * threads + t] != fibers[s * threads + (t ^ delta)])
            return false;
        }
      }
    }
  }
  return true;
}

} // namespace arg_reduce

template <typename Impl> struct ArgReduceLowerer {
  static Stmt Collective(const ArgReduceOpNode &op, const LowerArgs &args,
                         const Buffer &values, const Buffer &indices,
                         const Array<PrimExpr> &position, int width,
                         int scale) {
    Array<PrimExpr> call_args = {arg_reduce::Address(values, position),
                                 arg_reduce::Address(indices, position),
                                 Bool(op.is_max),
                                 Integer(width),
                                 Integer(scale),
                                 args.thread_bounds->min,
                                 args.thread_bounds->extent};
    if (width > Impl::WarpSize(args.target)) {
      int threads = *as_const_int(args.thread_bounds->extent);
      call_args.push_back(args.add_workspace(threads, values->dtype));
      call_args.push_back(args.add_workspace(threads, indices->dtype));
    }
    return Evaluate(Call(DataType::Handle(), tl::arg_reduce_all(), call_args));
  }

  static std::optional<Stmt> TryFragment(const ArgReduceOpNode &op,
                                         const LowerArgs &args,
                                         arith::Analyzer *analyzer) {
    using namespace arg_reduce;
    Buffer src = op.src_region->buffer, dst = op.dst_region->buffer;
    if (!IsFragmentBuffer(src) || !IsFragmentBuffer(dst))
      return std::nullopt;
    Fragment src_layout = Downcast<Fragment>(args.layout_map[src]);
    Fragment red_layout = ComputeReducerLayout(src_layout, op.dim);
    Fragment dst_layout = Downcast<Fragment>(
        args.layout_map[dst]->Reshape(red_layout->InputShape(), analyzer));
    if (!analyzer->CanProveEqual(ThreadBase(src_layout),
                                 args.thread_bounds->min) ||
        !analyzer->CanProveEqual(ThreadBase(red_layout),
                                 args.thread_bounds->min))
      return std::nullopt;

    auto vars = Variables(red_layout->InputShape());
    arith::Analyzer proof;
    for (size_t i = 0; i < vars.size(); ++i)
      proof.Bind(Downcast<Var>(vars[i]), Range(0, red_layout->InputShape()[i]));
    if (!ProveFragmentContains(dst_layout, red_layout, vars, vars, proof))
      return std::nullopt;
    Array<IterVar> src_vars;
    Var rv("arg_r");
    int j = 0;
    for (size_t i = 0; i < src_layout->InputDim(); ++i) {
      Var v = i == op.dim ? rv : Downcast<Var>(vars[j++]);
      src_vars.push_back(
          IterVar(Range(0, src->shape[i]), v, IterVarType::kDataPar));
    }
    auto logical =
        src_vars.Map([](const IterVar &v) { return PrimExpr(v->var); });
    auto src_thread = src_layout->ForwardThread(logical, {});
    reduce::ReduceOwnershipPlan plan;
    // Iterator compression is an optional optimization. Layouts outside its
    // affine ownership model still have the shared-memory implementation.
    try {
      plan = reduce::MakeReduceOwnershipPlan(
          src_layout->Forward(logical), src_thread, src_vars, rv, analyzer);
    } catch (const Error &) {
      return std::nullopt;
    } catch (const NormalizeIterException &) {
      return std::nullopt;
    }
    // Every thread-dependent part of the reduction axis must be represented
    // by a step. Mixed/non-affine terms cannot silently become a local reduce.
    PrimExpr reconstructed = Substitute(src_thread, {{rv, Integer(0)}});
    for (const auto &step : plan.thread_steps) {
      reconstructed =
          reconstructed + FloorMod(FloorDiv(rv, Integer(step.lower_factor)),
                                   Integer(step.extent)) *
                              step.scale;
    }
    proof.Bind(rv, Range(0, src->shape[op.dim]));
    if (!proof.CanProveEqual(src_thread, reconstructed))
      return std::nullopt;
    int threads = *as_const_int(args.thread_bounds->extent);
    if (!CheckTopology(red_layout, plan.thread_steps, threads,
                       Impl::WarpSize(args.target)))
      return std::nullopt;

    auto src_physical = plan.local_src_indices;
    src_physical.push_back(args.thread_index - ThreadBase(src_layout));
    auto recovered =
        src_layout->InverseWithLevel(true).first->Forward(src_physical);
    PrimExpr valid = const_true();
    j = 0;
    for (size_t i = 0; i < src_layout->InputDim(); ++i) {
      valid = valid && recovered[i] >= 0 && recovered[i] < src->shape[i];
      if (i != op.dim)
        valid = valid && recovered[i] == vars[j++];
    }
    valid = valid && recovered.back() >= 0 &&
            recovered.back() < src_layout->ReplicateExtent();
    auto recovered_logical = recovered;
    recovered_logical.pop_back();
    valid = valid &&
            src_layout->ForwardThread(recovered_logical, recovered.back()) ==
                args.thread_index - ThreadBase(src_layout);
    Buffer values = decl_buffer(red_layout->OutputShape(), src->dtype,
                                "arg_value", "local");
    Buffer indices = decl_buffer(red_layout->OutputShape(), dst->dtype,
                                 "arg_index", "local");
    auto position = red_layout->Forward(vars);
    Buffer physical_src = args.buffer_remap.Get(src).value_or(src);
    Stmt local =
        IfThenElse(analyzer->Simplify(valid),
                   Update(values, indices, position,
                          BufferLoad(physical_src, plan.local_src_indices),
                          recovered[op.dim], op.is_max));
    for (int i = static_cast<int>(plan.local_reduce_vars.size()) - 1; i >= 0;
         --i) {
      IterVar iv = plan.local_reduce_vars[i];
      local = For(iv->var, 0, iv->dom->extent, ForKind::kUnrolled, local);
    }
    Array<Stmt> body = {
        BufferStore(values, make_zero(src->dtype), position),
        BufferStore(indices, make_const(dst->dtype, -1), position), local};
    for (const auto &step : plan.thread_steps) {
      body.push_back(Collective(op, args, values, indices, position,
                                step.ReducingThreads(), step.scale));
    }
    Stmt reduce_body = Parallel(vars, red_layout->InputShape(), SeqStmt(body),
                                red_layout, args, analyzer);
    Buffer physical_dst = args.buffer_remap.Get(dst).value_or(dst);
    Stmt store = BufferStore(physical_dst, BufferLoad(indices, position),
                             dst_layout->Forward(vars));
    Stmt copy = Parallel(vars, dst_layout->InputShape(), store, dst_layout,
                         args, analyzer);
    return SeqStmt(
        {AllocBuffer(values), AllocBuffer(indices), reduce_body, copy});
  }

  static Stmt Shared(const ArgReduceOpNode &op, const LowerArgs &args,
                     arith::Analyzer *analyzer) {
    using namespace arg_reduce;
    auto shape = Shape(op.src_region), out_shape = Shape(op.dst_region);
    int threads = *as_const_int(args.thread_bounds->extent);
    int warp = Impl::WarpSize(args.target), groups = threads / warp;
    PrimExpr tid = args.thread_index - args.thread_bounds->min;
    Buffer src = op.src_region->buffer, dst = op.dst_region->buffer;
    Buffer input = src;
    Array<Stmt> body;
    if (IsFragmentBuffer(src)) {
      input = decl_buffer(shape, src->dtype, "arg_staging", "shared.dyn");
      body.push_back(AllocBuffer(input));
      auto vars = Variables(shape);
      Fragment layout = Downcast<Fragment>(args.layout_map[src]);
      // Replicas are identical; exactly one owner writes each shared element.
      Stmt store = IfThenElse(args.thread_index ==
                                  layout->ForwardThread(vars, Integer(0)) +
                                      ThreadBase(layout),
                              BufferStore(input, BufferLoad(src, vars), vars));
      body.push_back(Parallel(vars, shape, store, layout, args, analyzer));
      body.push_back(Impl::Sync(args));
    }
    int outputs = Size(out_shape);
    Buffer result =
        decl_buffer({Integer(outputs)}, dst->dtype, "arg_result", "shared.dyn");
    Buffer value = decl_buffer({Integer(1)}, src->dtype, "arg_value", "local");
    Buffer index = decl_buffer({Integer(1)}, dst->dtype, "arg_index", "local");
    body.push_back(AllocBuffer(result));
    body.push_back(AllocBuffer(value));
    body.push_back(AllocBuffer(index));
    Var batch("arg_batch"), r("arg_r");
    PrimExpr output = batch * groups + FloorDiv(tid, warp);
    PrimExpr axis_index = r * warp + FloorMod(tid, warp);
    Array<PrimExpr> reduced_shape;
    for (size_t i = 0; i < shape.size(); ++i)
      if (i != op.dim)
        reduced_shape.push_back(shape[i]);
    auto spatial = Unflatten(output, reduced_shape);
    spatial.insert(spatial.begin() + op.dim, axis_index);
    auto source_indices =
        IsFragmentBuffer(src) ? spatial : Offset(spatial, op.src_region);
    Stmt update = IfThenElse(output < outputs && axis_index < shape[op.dim],
                             Update(value, index, {Integer(0)},
                                    BufferLoad(input, source_indices),
                                    axis_index, op.is_max));
    Stmt loop = For(r, 0, FloorDiv(shape[op.dim] + warp - 1, warp),
                    ForKind::kSerial, update);
    Stmt work = SeqStmt(
        {BufferStore(value, make_zero(src->dtype), {Integer(0)}),
         BufferStore(index, make_const(dst->dtype, -1), {Integer(0)}), loop,
         Collective(op, args, value, index, {Integer(0)}, warp, 1),
         IfThenElse(
             output < outputs && FloorMod(tid, warp) == 0,
             BufferStore(result, BufferLoad(index, {Integer(0)}), {output}))});
    body.push_back(For(batch, 0, Integer((outputs + groups - 1) / groups),
                       ForKind::kSerial, work));
    body.push_back(Impl::Sync(args));
    if (IsFragmentBuffer(dst)) {
      auto vars = Variables(out_shape);
      PrimExpr flat = Integer(0);
      for (size_t i = 0; i < vars.size(); ++i)
        flat = flat * out_shape[i] + vars[i];
      Stmt store = BufferStore(dst, BufferLoad(result, {flat}), vars);
      body.push_back(Parallel(vars, out_shape, store,
                              Downcast<Fragment>(args.layout_map[dst]), args,
                              analyzer));
    } else {
      Var i("arg_copy");
      PrimExpr flat = i * threads + tid;
      Stmt store = IfThenElse(
          flat < outputs,
          BufferStore(dst, BufferLoad(result, {flat}),
                      Offset(Unflatten(flat, out_shape), op.dst_region)));
      body.push_back(For(i, 0, Integer((outputs + threads - 1) / threads),
                         ForKind::kSerial, store));
    }
    // Scratch may be reused by the next TileOp after every reader finishes.
    body.push_back(Impl::Sync(args));
    return SeqStmt(body);
  }

  static Stmt Lower(const ArgReduceOpNode &op, const LowerArgs &args,
                    arith::Analyzer *analyzer) {
    int warp = Impl::WarpSize(args.target);
    CHECK(analyzer->CanProveEqual(FloorMod(args.thread_bounds->min, warp), 0) &&
              analyzer->CanProveEqual(
                  FloorMod(args.thread_bounds->extent, warp), 0),
          ValueError)
        << "arg_reduce requires a warp-aligned participating thread range, got "
        << args.thread_bounds;
    arg_reduce::Size(arg_reduce::Shape(op.src_region));
    if (auto fast = TryFragment(op, args, analyzer))
      return *fast;
    return Shared(op, args, analyzer);
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm
#endif // TVM_TL_BACKEND_COMMON_OP_ARG_REDUCE_H_
