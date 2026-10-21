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
 * \file buffer_lifetime_analysis.h
 * \brief Iteration-order and generation-lifetime queries for AutoSchedule
 * buffers.
 */

#pragma once

#include <tvm/tirx/op.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace ascend {

// The query's scope and the site's epoch domain identify the clock. Offset
// counts iterations of that clock, never pipeline stages or flag slots.
struct IterationEvent {
  size_t site;
  int64_t offset{0};

  bool operator<(const IterationEvent &other) const {
    return std::pair(offset, site) < std::pair(other.offset, other.site);
  }
  bool operator==(const IterationEvent &other) const {
    return site == other.site && offset == other.offset;
  }
};

inline std::optional<int64_t> AddIterationOffsets(int64_t lhs, int64_t rhs) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return std::nullopt;
  return lhs + rhs;
}

// Nonnegative weighted reachability, without a distance-enumeration horizon.
// Strong edges certify completion order; weak edges only certify issue order.
// Queries require a path containing at least one strong edge. Clock/guard
// compatibility is checked by the producer, not inferred from numeric IDs here.
// Nonnegative edges also keep intermediate iterations between the endpoints,
// so a path does not rely on events outside a finite loop's boundaries.
class IterationOrder {
public:
  explicit IterationOrder(size_t sites = 0) : outgoing_(2 * sites) {}

  bool AddEdge(size_t src, size_t dst, int64_t distance, bool strict = true) {
    if (src >= outgoing_.size() / 2 || dst >= outgoing_.size() / 2 ||
        distance < 0)
      return false;
    // State 0 has only issue order so far; state 1 has crossed a completion
    // edge. Keep both distances so a short weak path cannot hide a longer
    // strong one. Weak edges remain usable before and after synchronization.
    for (size_t completed = 0; completed < 2; ++completed) {
      size_t next = 2 * dst + (completed || strict);
      auto [it, inserted] =
          outgoing_[2 * src + completed].emplace(next, distance);
      if (inserted || distance < it->second) {
        it->second = distance;
        distances_.clear();
      }
    }
    return true;
  }

  std::optional<int64_t> MinimumDistance(size_t src, size_t dst) const {
    if (src >= outgoing_.size() / 2 || dst >= outgoing_.size() / 2)
      return std::nullopt;
    auto cached = distances_.find(src);
    if (cached == distances_.end()) {
      std::vector<std::optional<int64_t>> distances(outgoing_.size());
      using Entry = std::pair<int64_t, size_t>;
      std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> work;
      // Start before any completion edge, including for src == dst.
      for (const auto &[next, distance] : outgoing_[2 * src]) {
        distances[next] = distance;
        work.emplace(distance, next);
      }
      while (!work.empty()) {
        auto [distance, site] = work.top();
        work.pop();
        if (distances[site] != distance)
          continue;
        for (const auto &[next, delta] : outgoing_[site]) {
          auto candidate = AddIterationOffsets(distance, delta);
          if (candidate &&
              (!distances[next] || *candidate < *distances[next])) {
            distances[next] = *candidate;
            work.emplace(*candidate, next);
          }
        }
      }
      cached = distances_.emplace(src, std::move(distances)).first;
    }
    return cached->second[2 * dst + 1];
  }

private:
  std::vector<std::unordered_map<size_t, int64_t>> outgoing_;
  mutable std::unordered_map<size_t, std::vector<std::optional<int64_t>>>
      distances_;
};

using MinimumIterationDistance =
    std::function<std::optional<int64_t>(size_t, size_t)>;

// The minimum query must certify *every* distance at or above its result,
// using destination-event recurrence in one compatible clock. This is a
// stronger contract than finding one exact weighted path.
inline std::optional<int64_t>
SeparateIterationEvents(const std::vector<IterationEvent> &ends,
                        const std::vector<IterationEvent> &begins,
                        const MinimumIterationDistance &minimum_distance) {
  if (ends.empty() || begins.empty())
    return std::nullopt;
  int64_t shift = std::numeric_limits<int64_t>::min();
  for (const IterationEvent &end : ends) {
    for (const IterationEvent &begin : begins) {
      auto distance = minimum_distance(end.site, begin.site);
      if (!distance || begin.offset == std::numeric_limits<int64_t>::min())
        return std::nullopt;
      auto offset = AddIterationOffsets(end.offset, -begin.offset);
      auto required =
          offset ? AddIterationOffsets(*offset, *distance) : std::nullopt;
      if (!required)
        return std::nullopt;
      shift = std::max(shift, *required);
    }
  }
  return shift;
}

// A separating cut q covers all relative generations without enumerating
// iterations:
//
//   ends(A(k))     < begins(B(k+q))
//   ends(B(k+q-1)) < begins(A(k))
//
// Monotonic event recurrence extends the first relation to every later B,
// and the second to every earlier B. Thus q is a proof certificate, not a
// heuristic unroll bound. Whole-allocation aliasing checks every generation,
// conservatively including generations using different physical ring slots.
template <typename Lifetime>
bool CanSeparatePeriodicLifetimes(
    const Lifetime &lhs, const Lifetime &rhs,
    const MinimumIterationDistance &minimum_distance) {
  auto forward =
      SeparateIterationEvents(lhs.end_sites, rhs.begin_sites, minimum_distance);
  auto reverse =
      SeparateIterationEvents(rhs.end_sites, lhs.begin_sites, minimum_distance);
  if (!forward || !reverse)
    return false;
  auto sum = AddIterationOffsets(*forward, *reverse);
  return sum && *sum <= 1;
}

struct GenerationLifetime {
  std::vector<IterationEvent> begin_sites;
  std::vector<IterationEvent> end_sites;
  PrimExpr guard;
  bool guard_iteration_invariant{false};
};

using HappensBeforeQuery =
    std::function<bool(size_t src, size_t dst, int64_t distance)>;

inline std::vector<IterationEvent>
MakeIterationEvents(const std::vector<size_t> &sites, int64_t offset = 0) {
  std::vector<IterationEvent> events;
  events.reserve(sites.size());
  for (size_t site : sites)
    events.push_back({site, offset});
  return events;
}

inline bool IsLocalGeneration(const GenerationLifetime &generation) {
  auto local = [](const IterationEvent &event) { return event.offset == 0; };
  return std::all_of(generation.begin_sites.begin(),
                     generation.begin_sites.end(), local) &&
         std::all_of(generation.end_sites.begin(), generation.end_sites.end(),
                     local);
}

inline bool AllEventsBefore(const std::vector<IterationEvent> &ends,
                            const std::vector<IterationEvent> &begins,
                            int64_t shift,
                            const HappensBeforeQuery &happens_before) {
  if (ends.empty() || begins.empty())
    return false;
  for (const IterationEvent &end : ends) {
    for (const IterationEvent &begin : begins) {
      if (end.offset == std::numeric_limits<int64_t>::min())
        return false;
      auto destination = AddIterationOffsets(begin.offset, shift);
      auto distance = destination
                          ? AddIterationOffsets(*destination, -end.offset)
                          : std::nullopt;
      if (!distance || !happens_before(end.site, begin.site, *distance))
        return false;
    }
  }
  return true;
}

using MutuallyExclusiveQuery =
    std::function<bool(const PrimExpr &lhs, const PrimExpr &rhs)>;

inline bool AllHappenBefore(const std::vector<size_t> &sources,
                            const std::vector<size_t> &destinations,
                            int64_t distance,
                            const HappensBeforeQuery &happens_before) {
  for (size_t source : sources) {
    for (size_t destination : destinations) {
      if (!happens_before(source, destination, distance))
        return false;
    }
  }
  return true;
}

inline bool
CanAliasGenerations(const std::vector<GenerationLifetime> &lhs,
                    const std::vector<GenerationLifetime> &rhs, bool periodic,
                    const HappensBeforeQuery &happens_before,
                    const MinimumIterationDistance &minimum_distance,
                    const MutuallyExclusiveQuery &mutually_exclusive) {
  ICHECK(!lhs.empty() && !rhs.empty());
  for (const GenerationLifetime &a : lhs) {
    for (const GenerationLifetime &b : rhs) {
      bool invariant =
          a.guard_iteration_invariant && b.guard_iteration_invariant;
      if ((!periodic || invariant) && a.guard.defined() && b.guard.defined() &&
          mutually_exclusive(a.guard, b.guard))
        continue;
      if (periodic) {
        // A changing predicate cannot be shifted between generations without
        // an explicit clock/guard mapping. In particular, local exclusion is
        // not exclusion between adjacent active epochs.
        if (!invariant || !CanSeparatePeriodicLifetimes(a, b, minimum_distance))
          return false;
      } else {
        bool a_before_b =
            AllEventsBefore(a.end_sites, b.begin_sites, 0, happens_before);
        bool b_before_a =
            AllEventsBefore(b.end_sites, a.begin_sites, 0, happens_before);
        if (a_before_b == b_before_a)
          return false;
      }
    }
  }
  return true;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
