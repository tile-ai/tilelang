// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once
#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "gemm_sm90.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 890))
#include "gemm_sm89.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 750))
#include "gemm_sm80.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 700))
#include "gemm_sm70.h"
#else

#endif
