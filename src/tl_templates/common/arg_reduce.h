#pragma once

namespace tl {

template <typename T> TL_DEVICE T arg_reduce_compare_value(T value) {
  return value;
}
// Some native half != operators are ordered comparisons (false for NaN).
// Widen only these floating types; integer values retain their full precision.
TL_DEVICE float arg_reduce_compare_value(half_t value) { return float(value); }
TL_DEVICE float arg_reduce_compare_value(bfloat16_t value) {
  return float(value);
}

// A negative index is the identity, so finite extrema and all-NaN tiles need
// no value sentinel. Both components always come from the same candidate.
template <bool IsMax, typename T, typename I, typename J>
TL_DEVICE void arg_reduce_update(T *value, I *index, T candidate,
                                 J candidate_offset) {
  const I candidate_index = static_cast<I>(candidate_offset);
  if (candidate_index < 0)
    return;
  const auto candidate_value = arg_reduce_compare_value(candidate);
  const auto current_value = arg_reduce_compare_value(*value);
  const bool candidate_nan = candidate_value != candidate_value;
  const bool current_nan = current_value != current_value;
  const bool better =
      IsMax ? candidate_value > current_value : candidate_value < current_value;
  const bool equal =
      candidate_value == current_value || (candidate_nan && current_nan);
  if (*index < 0 || (candidate_nan && !current_nan) ||
      (!current_nan && better) || (equal && candidate_index < *index)) {
    *value = candidate;
    *index = candidate_index;
  }
}

} // namespace tl
