#include "BuddhabrotRenderer.h"
#include "BuddhabrotCuda.cuh"
#include <iostream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

BuddhabrotRenderer::BuddhabrotRenderer(const BuddhabrotViewport& viewport,
	bool subtractive_blending,
	float background_r, float background_g, float background_b,
	float start_re, float start_im)
	    :   viewport(viewport),
             subtractive_blending(subtractive_blending),
	         start_re(start_re),
             start_im(start_im) {
    background_color.r = background_r;
    background_color.g = background_g;
    background_color.b = background_b;
    size_t num_pixels = viewport.width * viewport.height;
    image_buffer.insert(image_buffer.begin(), num_pixels, background_color);
    compute_buffer.resize(num_pixels);
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
    
    error = initCUDA(dev_buddhabrot, dev_image, viewport);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: while initialize with CUDA\n";
    }

    error = computeBuddhabrotCUDA(parameters, viewport, dev_buddhabrot);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: while computing buddhabrot with CUDA\n";
        goto Error;
    }
    
    error = computeLayerCUDA(parameters, viewport, dev_buddhabrot, dev_image, r, g, b, correction_a, correction_gamma);
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
    cudaError_t error = getImageCuda(image, dev_image, viewport);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error while getting image with CUDA\n";
        goto Error;
    }

    int status = stbi_write_png(actual_filename.c_str(), viewport.width, viewport.height, 3, image, viewport.width * 3);
    if (status == 0) {
        std::cerr << "Error while saving image to file\n";
    }

Error:
    freeCUDA(dev_buddhabrot, dev_image);
    delete[] image;
}

