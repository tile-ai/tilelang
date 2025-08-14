#pragma once

#include <cute/arch/mma_sm120.hpp>
#include "cuda_fp8.h"

#define JUST_POC_FOR_SM120

// SM120 supports same instruction set as SM80 for WMMA
#include "gemm_sm80.h"
