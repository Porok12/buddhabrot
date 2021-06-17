#pragma once
#include <vector>
#include <string>
#include "BuddhabrotParameters.h"

/**
 * @brief Klasa przeznaczona do renderowania fraktala
*/
class BuddhabrotRenderer
{
    struct FloatRGB {
        float r;
        float g;
        float b;
    };
public:
	/**
	 * @brief Konstruktor alokuj¹cy pamiêæ
	 * @param viewport obiekt zawieraj¹cy parametry obrazu
	 * @param subtractive_blending typ blendingu
	 * @param background_r mno¿nik sk³adowej czerwonej
	 * @param background_g mno¿nik sk³adowej zielonej
	 * @param background_b mno¿nik sk³adowej niebieskiej
	 * @param start_re startowa sk³adowa rzeczywista
	 * @param start_im startowa sk³adowa urojona
	*/
	BuddhabrotRenderer(const BuddhabrotViewport& viewport,
		bool subtractive_blending,
		float background_r, float background_g, float background_b,
		float start_re, float start_im);
	/**
	* @brief Destruktor zwalniaj¹cy pamiêæ
	*/
    ~BuddhabrotRenderer();
    BuddhabrotRenderer(const BuddhabrotRenderer& other) = delete;
    BuddhabrotRenderer& operator=(const BuddhabrotRenderer& other) = delete;
    BuddhabrotRenderer(BuddhabrotRenderer&& other) noexcept = delete;
    BuddhabrotRenderer& operator=(BuddhabrotRenderer&& other) noexcept  = delete;

	/**
	 * @brief obliczenie warstwy fraktala
	 * @param samples iloœæ próbek
	 * @param iterations iloœæ iteracji warstwy
	 * @param r mno¿nik sk³adowej czerwonej
	 * @param g mno¿nik sk³adowej zielonej
	 * @param b mno¿nik sk³adowej niebieskiej
	 * @param correction_a wartoœæ wspó³czynnika a
	 * @param correction_gamma wartoœæ korekcji gamma
	*/
	void addLayer(uint64_t samples, uint32_t iterations,
		float r, float g, float b,
		float correction_a, float correction_gamma);

	/**
	 * @brief zapis fraktala do pliku 
	 * @param filename nazwa pliku pod jak¹ ma byæ zapisany obraz
	*/
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

