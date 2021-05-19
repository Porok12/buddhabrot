#pragma once
#include <vector>
#include <string>
#include "BuddhabrotParameters.h"

class BuddhabrotRenderer
{
    struct FloatRGB {
        float r;
        float g;
        float b;
    };
public:
	BuddhabrotRenderer(const BuddhabrotViewport& viewport,
		bool subtractive_blending,
		float background_r, float background_g, float background_b,
		float start_re, float start_im);

	void addLayer(uint64_t samples, uint32_t iterations,
		float r, float g, float b,
		float correction_a, float correction_gamma);

	void saveImage(const std::string& filename);

private:
	BuddhabrotViewport viewport;
	bool subtractive_blending;
    FloatRGB background_color;
    float start_re;
    float start_im;
	std::vector<FloatRGB> image_buffer;
    std::vector<uint32_t> compute_buffer;
	float* dev_image = NULL;
	uint32_t* dev_buddhabrot = NULL;
};

