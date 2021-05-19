#pragma once
#include "BuddhabrotParameters.h"
#include <cuda_runtime_api.h>

cudaError_t computeBuddhabrotCUDA(const BuddhabrotParameters& parameters, const BuddhabrotViewport& viewport, uint32_t* dev_buddhabrot);
cudaError_t computeLayerCUDA(const BuddhabrotParameters& parameters, const BuddhabrotViewport& viewport, uint32_t* &dev_buddhabrot, float* &dev_image, float r, float g, float b, float correction_a, float correction_gamma);
cudaError_t getImageCuda(uint8_t* &image, float* &dev_image, const BuddhabrotViewport& viewport);
cudaError_t initCUDA(uint32_t* &dev_buddhabrot, float* &dev_image, const BuddhabrotViewport& viewport);
void freeCUDA(uint32_t* &dev_buddhabrot, float* &dev_image);