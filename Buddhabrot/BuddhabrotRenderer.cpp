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
    cudaError_t error = computeBuddhabrotCUDA(parameters, viewport, compute_buffer.data());
    if (error != cudaSuccess) {
        std::cout << "CUDA error while computing buddhabrot with CUDA\n";
    }

    // adding rendering results to the image, this could be CUDA too
    uint32_t max_value = 0;
    for (uint32_t value : compute_buffer) {
        max_value = std::max(max_value, value);
    }
    std::cout << "\tlayer max: " << max_value << "\n";
    if (max_value > 0) {
        float norm_mul = 1.f / max_value;
        if (subtractive_blending) {
            r = background_color.r - r;
            g = background_color.g - g;
            b = background_color.b - b;
            for (size_t i = 0; i < num_pixels; ++i) {
                float normalized = compute_buffer[i] * norm_mul;
                float corrected = correction_a * std::pow(normalized, correction_gamma);
                FloatRGB& output = image_buffer[i];
                output.r -= corrected * r;
                output.g -= corrected * g;
                output.b -= corrected * b;
            }
        }
        else {
            for (size_t i = 0; i < num_pixels; ++i) {
                float normalized = compute_buffer[i] * norm_mul;
                float corrected = correction_a * std::pow(normalized, correction_gamma);
                FloatRGB& output = image_buffer[i];
                output.r += corrected * r;
                output.g += corrected * g;
                output.b += corrected * b;
            }
        }
    }
}

void BuddhabrotRenderer::saveImage(const std::string& filename) {
    std::string actual_filename = filename + ".png";
    size_t num_pixels = viewport.width * viewport.height;
    uint8_t* image = new uint8_t[num_pixels * 3];
    for (size_t i = 0; i < num_pixels; ++i) {
        image[i * 3 + 0] = std::min(255.f, std::max(0.f, image_buffer[i].r * 255.f));
        image[i * 3 + 1] = std::min(255.f, std::max(0.f, image_buffer[i].g * 255.f));
        image[i * 3 + 2] = std::min(255.f, std::max(0.f, image_buffer[i].b * 255.f));
    }
    int status = stbi_write_png(actual_filename.c_str(), viewport.width, viewport.height, 3, image, viewport.width * 3);
    if (status == 0) {
        std::cerr << "Error while saving image to file\n";
    }
    delete[] image;
}

