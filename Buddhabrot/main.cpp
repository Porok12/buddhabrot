#define _USE_MATH_DEFINES
#include <cmath>
#include "cuda_runtime.h"
#include <string>
#include <stdio.h>
#include "BuddhabrotRenderer.h"

cudaError_t initCuda() {
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}
	return cudaSuccess;
}

int main() {
	if (initCuda() != cudaSuccess) {
		return 1;
	}
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint64_t samples = 100000000;
	float start_re = 0.f;
	float start_im = 0.f;
	float background_r = 0.f;
	float background_g = 0.f;
	float background_b = 0.f;
	bool subtractive_blending = false;
	float center_re = -0.5f;
	float center_im = 0.f;
	float scale = 3.f;
	float rotation = 90.f * M_PI / 180.f;
	std::string filename = "buddhabrot";
    BuddhabrotViewport viewport(width, height, center_re, center_im, scale, rotation);
    BuddhabrotRenderer renderer(viewport, false, 0, 0, 0, start_re, start_im);
    renderer.addLayer(samples, 800, 1, 0, 0, 2, 1);
    renderer.addLayer(samples, 200, 0, 1, 0, 2, 1);
    renderer.addLayer(samples, 50, 0, 0, 1, 2, 1);
    renderer.saveImage(filename);
    return 0;
}
