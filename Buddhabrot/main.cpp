#define _USE_MATH_DEFINES
#include <cmath>
#include "cuda_runtime.h"
#include <string>
#include <stdio.h>
#include <chrono>
#include "BuddhabrotRenderer.h"
#include "SettingsReader.h"

/**
 * @brief inicjalizacja CUDA-y
 * @return status initcjalizacji
*/
cudaError_t initCuda() {
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}
	return cudaSuccess;
}

/**
 * @brief pomocnicza funkcja, odczytuj¹ca kolor z json_array do tablicy
 * @param result tablica do której bêd¹ za³adowane dane
 * @param json_array obiekt z którego maj¹ byæ pobrane kolory
*/
inline void readColor(float* result, json& json_array) {
    int counter = 0;
    for (auto& el : json_array) {
        if (counter < 3) {
            el.get_to(result[counter]);
        }
        counter++;
    }
}

/**
 * @brief g³ówna funkcja
 * @param argc iloœæ argumentów
 * @param argv lista argumentów
 * @return kod zakoñczenia dzia³ania
*/
int main(int argc, char* argv[]) {
	json j = argc >= 2 ? load_config(argv[1]) : load_config();

	if (initCuda() != cudaSuccess) {
		return EXIT_FAILURE;
	}

	uint32_t width = j.value("width", 1280);
    uint32_t height = j.value("height", 720);
    uint64_t samples = j.value("samples", 10000000);
	uint64_t max_repeats_per_thread = j.value("mrpt", 256);
	uint64_t blocks_per_multiprocessor = j.value("bpm", 8);
	float start_re = j.value("start_re", 0.f);
	float start_im = j.value("start_im", 0.f);

    std::string blending = j.value("blending", "add");
    bool subtractive_blending = blending == "sub";
	float center_re = j.value("center_re", -0.5f);
	float center_im = j.value("center_im", 0.f);
	float scale = j.value("scale", 3.f);
	float rotation = j.value("rotation", 0.f) * M_PI / 180.f;
	std::string filename = j.value("filename", "buddhabrot");

    json background_json = j.at("background");
    float background_color[3] = { 0.f, 0.f, 0.f };
    if (background_json.is_array()) {
        readColor(background_color, background_json);
    }

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    BuddhabrotViewport viewport(width, height, center_re, center_im, scale, rotation, max_repeats_per_thread, blocks_per_multiprocessor);
    BuddhabrotRenderer renderer(viewport, subtractive_blending,
        background_color[0], background_color[1], background_color[2],
        start_re, start_im);

    json layers = j.at("layers");
    if (layers.is_array()) {
        for (auto& layer : layers) {
            float color[3] = { 1.f, 1.f, 1.f };
            json color_json = layer.at("color");
            if (color_json.is_array()) {
                readColor(color, color_json);
            }
            renderer.addLayer(samples, layer.value("iterations", 50),
                color[0], color[1], color[2],
                layer.value("correction_a", 2.f), layer.value("correction_gamma", 1.f));
        }
    } else {
        renderer.addLayer(samples, 800, 1, 0, 0, 2, 1);
        renderer.addLayer(samples, 200, 0, 1, 0, 2, 1);
        renderer.addLayer(samples, 50, 0, 0, 1, 2, 1);
    }

    renderer.saveImage(filename);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Total time (CPU): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    return EXIT_SUCCESS;
}
