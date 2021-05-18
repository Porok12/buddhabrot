#pragma once

#include <cstdint>

struct BuddhabrotParameters {
	float start_re;
	float start_im;
    uint64_t samples;
	uint32_t iterations;
};

struct BuddhabrotViewport {
	BuddhabrotViewport(uint32_t width, uint32_t height, float center_re, float center_im, float scale, float rotation);
	uint32_t width;
	uint32_t height;
    // transfromation for converting complex position into pixel position
    float a11;
    float a12;
    float a21;
    float a22;
    float b1;
    float b2;
};