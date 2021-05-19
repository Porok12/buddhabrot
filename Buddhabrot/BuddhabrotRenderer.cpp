#include "BuddhabrotRenderer.h"
#include "BuddhabrotCuda.cuh"
#include <iostream>
#include <algorithm>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

BuddhabrotRenderer::BuddhabrotRenderer(const BuddhabrotViewport& viewport,
	bool subtractive_blending,
	float background_r, float background_g, float background_b,
	float start_re, float start_im)
	    :   viewport(viewport),
            subtractive_blending(subtractive_blending),
            background_r(background_r),
            background_g(background_g),
            background_b(background_b),
            start_re(start_re),
            start_im(start_im) {
    allocateMemoryCUDA(dev_buddhabrot, dev_image, viewport);
}

BuddhabrotRenderer::~BuddhabrotRenderer() {
    freeMemoryCUDA(dev_buddhabrot, dev_image);
}

void BuddhabrotRenderer::addLayer(uint64_t samples, uint32_t iterations,
    float r, float g, float b,
    float correction_a, float correction_gamma) {
    BuddhabrotParameters parameters;
    parameters.samples = samples;
    parameters.iterations = iterations;
    parameters.start_re = start_re;
    parameters.start_im = start_im;
    size_t num_pixels = viewport.width * viewport.height;
    cudaError_t error;
    
    error = clearBufferCUDA(dev_buddhabrot, dev_image, viewport);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: while initialize with CUDA\n";
        goto Error;
    }

    error = computeBuddhabrotCUDA(parameters, viewport, dev_buddhabrot);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: while computing buddhabrot with CUDA\n";
        goto Error;
    }
    
    error = computeLayerCUDA(parameters, viewport, dev_buddhabrot, dev_image,
        r, g, b, correction_a, correction_gamma,
        subtractive_blending, background_r, background_g, background_b);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: while computing layer with CUDA\n";
        goto Error;
    }
Error:
    return;
}

void BuddhabrotRenderer::saveImage(const std::string& filename) {
    std::string actual_filename = filename + ".png";
    size_t num_pixels = viewport.width * viewport.height;
    uint8_t* image = new uint8_t[num_pixels * 3];
    cudaError_t error = getImageCUDA(image, dev_image, viewport,
        background_r, background_g, background_b);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error while getting image with CUDA\n";
        goto Error;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int status = stbi_write_png(actual_filename.c_str(), viewport.width, viewport.height, 3, image, viewport.width * 3);
    if (status == 0) {
        std::cerr << "Error while saving image to file\n";
        goto Error;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\Saving image: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

Error:
    delete[] image;
}

