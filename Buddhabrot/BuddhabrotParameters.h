#pragma once

#include <cstdint>

/**
 * @brief struktura opisuj¹ca parametry fraktala
*/
struct BuddhabrotParameters {
	float start_re;
	float start_im;
    uint64_t samples;
	uint32_t iterations;
};

/**
 * @brief struktura opisuj¹ca obraz fraktala
*/
struct BuddhabrotViewport {
	BuddhabrotViewport(uint32_t width, uint32_t height, float center_re, float center_im, float scale, float rotation, uint64_t max_repeats_per_thread, uint64_t blocks_per_multiprocessor);
	uint32_t width;
	uint32_t height;
    uint64_t max_repeats_per_thread;
    uint64_t blocks_per_multiprocessor;
    // transfromation for converting complex position into pixel position
    float a11;
    float a12;
    float a21;
    float a22;
    float b1;
    float b2;
};