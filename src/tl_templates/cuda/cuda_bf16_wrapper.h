// Downloaded from from FasterTransformer v5.2.1
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.2.1_tag/src/fastertransformer/utils/cuda_bf16_wrapper.h
/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_bf16.h>
#include "common.h"

TL_DEVICE bfloat16_t min(const bfloat16_t a, const bfloat16_t b) {
    return (a < b) ? a : b;
}

TL_DEVICE bfloat16_t max(const bfloat16_t a, const bfloat16_t b) {
    return (a > b) ? a : b;
}

