#pragma once
#include "BuddhabrotParameters.h"
#include <cuda_runtime_api.h>

cudaError_t computeBuddhabrotCUDA(const BuddhabrotParameters& parameters, const BuddhabrotViewport& viewport, uint32_t* buffer);
