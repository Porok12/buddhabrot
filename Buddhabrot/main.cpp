#define _USE_MATH_DEFINES
#include <cmath>
#include "cuda_runtime.h"
#include <string>
#include <stdio.h>
#include "BuddhabrotRenderer.h"
#include "SettingsReader.h"

cudaError_t initCuda() {
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}
	return cudaSuccess;
}

int main(int argc, char* argv[]) {
	json j = argc == 2 ? load_config(argv[1]) : load_config();

	if (initCuda() != cudaSuccess) {
		return EXIT_FAILURE;
	}

	uint32_t width = j.value("width", 1280);
    uint32_t height = j.value("height", 720);
    uint64_t samples = j.value("samples", 10000000);
	uint64_t max_repeats_per_thread = j.value("mrpt", 256);
	uint64_t blocks_per_multiprocessor = j.value("bpm", 8);
	float start_re = j.value("start_re", 0);
	float start_im = j.value("start_im", 0);
	float background_r = j.value("background_r", 0);
	float background_g = j.value("background_g", 0);
	float background_b = j.value("background_b", 0);
	bool subtractive_blending = j.value("subtractive_blending", false);
	float center_re = j.value("center_re", -0.5f);
	float center_im = j.value("center_im", 0.f);
	float scale = j.value("scale", 3.f);
	float rotation = j.value("rotation", 90.f) * M_PI / 180.f;
	std::string filename = j.value("filename", "buddhabrot");
    BuddhabrotViewport viewport(width, height, center_re, center_im, scale, rotation, max_repeats_per_thread, blocks_per_multiprocessor);
    BuddhabrotRenderer renderer(viewport, false, 0, 0, 0, start_re, start_im);
    renderer.addLayer(samples, j.value("iterations_r", 800), 1, 0, 0, 2, j.value("gamma_r", 1));
    //renderer.addLayer(samples, j.value("iterations_g", 200), 0, 1, 0, 2, j.value("gamma_g", 1));
    //renderer.addLayer(samples, j.value("iterations_b", 50), 0, 0, 1, 2, j.value("gamma_b", 1));
	/*max_kernel(g_data, max_value, size)
	float norm_mul = 1.f / max_value;
	layer_kernel(g_data, dev_image, norm_mul, 1, 0, 0, 2, 1);
	layer_kernel(g_data, dev_image, norm_mul, 0, 1, 0, 2, 1);
	layer_kernel(g_data, dev_image, norm_mul, 0, 0, 1, 2, 1);*/
    renderer.saveImage(filename);
    return EXIT_SUCCESS;
}
