#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDA_FUNCTIONS{
    float partitioned_reduction(float* input_array, size_t dataSize);
}
