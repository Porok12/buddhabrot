#pragma once
#include "BuddhabrotParameters.h"
#include <cuda_runtime_api.h>

cudaError_t computeBuddhabrotCUDA(const BuddhabrotParameters& parameters,
    const BuddhabrotViewport& viewport, uint32_t* dev_buddhabrot);

cudaError_t computeLayerCUDA(const BuddhabrotParameters& parameters,
    const BuddhabrotViewport& viewport, uint32_t* &dev_buddhabrot,
    float* &dev_image, float r, float g, float b,
    float correction_a, float correction_gamma,
    bool subtractive_blending, float background_r,
    float background_g, float background_b);

cudaError_t getImageCUDA(uint8_t* &image, float* &dev_image,
    const BuddhabrotViewport& viewport,
    float background_r, float background_g, float background_b);

cudaError_t allocateMemoryCUDA(uint32_t*& dev_buddhabrot, float*& dev_image,
    const BuddhabrotViewport& viewport);

void freeMemoryCUDA(uint32_t*& dev_buddhabrot, float*& dev_image);

cudaError_t clearBufferCUDA(uint32_t*& dev_buddhabrot, float*& dev_image,
    const BuddhabrotViewport& viewport);
