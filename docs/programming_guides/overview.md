# Programming Guides Overview

This section provides a practical guide to writing high‑performance kernels with Tile Language (tile‑lang).
It mirrors the structure of a similar guide in another project and adapts it to tile‑lang concepts and APIs.

- Audience: Developers implementing custom GPU/CPU kernels with tile‑lang
- Prereqs: Basic Python, NumPy/Tensor concepts, and familiarity with GPU programming notions
- Scope: Language basics, control flow, instructions, layouts, reducers,
  software pipelining, warp specialization, TMA, synchronization, autotuning,
  and the type system

## What You’ll Learn
- How to structure kernels with TileLang’s core DSL constructs
- How to move data across global/shared/fragment and pipeline compute
- How layout replication differs from reducer partials
- How to use manual software-pipeline annotations when inference is not enough
- How to reason about TMA lifetimes, barriers, and proxy fences
- How to apply autotuning to tile sizes and schedules
- How to specify and work with dtypes in kernels

## Suggested Reading Order
1. Language Basics
2. Control Flow
3. Instructions
4. Advanced Semantics
5. Software Pipelines
6. Autotuning
7. Type System

## Related Docs

- Tutorials: see existing guides in `tutorials/`
- Operators: examples in `deeplearning_operators/`
