#pragma once

#include <cute/arch/mma_sm89.hpp>

#include "cuda_fp8.h"

#include "gemm_mma.h"

namespace tl {
using tl_mma::gemm_rs;
using tl_mma::gemm_sr;
using tl_mma::gemm_ss;
} // namespace tl
