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
    ~BuddhabrotRenderer();
    BuddhabrotRenderer(const BuddhabrotRenderer& other) = delete;
    BuddhabrotRenderer& operator=(const BuddhabrotRenderer& other) = delete;
    BuddhabrotRenderer(BuddhabrotRenderer&& other) noexcept = delete;
    BuddhabrotRenderer& operator=(BuddhabrotRenderer&& other) noexcept  = delete;

	void addLayer(uint64_t samples, uint32_t iterations,
		float r, float g, float b,
		float correction_a, float correction_gamma);

	void saveImage(const std::string& filename);

private:
	BuddhabrotViewport viewport;
	bool subtractive_blending;
    float background_r;
    float background_g;
    float background_b;
    float start_re;
    float start_im;
	float* dev_image = NULL;
	uint32_t* dev_buddhabrot = NULL;
};

