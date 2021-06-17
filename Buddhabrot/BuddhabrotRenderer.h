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
	 * @brief Konstruktor alokuj�cy pami��
	 * @param viewport obiekt zawieraj�cy parametry obrazu
	 * @param subtractive_blending typ blendingu
	 * @param background_r mno�nik sk�adowej czerwonej
	 * @param background_g mno�nik sk�adowej zielonej
	 * @param background_b mno�nik sk�adowej niebieskiej
	 * @param start_re startowa sk�adowa rzeczywista
	 * @param start_im startowa sk�adowa urojona
	*/
	BuddhabrotRenderer(const BuddhabrotViewport& viewport,
		bool subtractive_blending,
		float background_r, float background_g, float background_b,
		float start_re, float start_im);
	/**
	* @brief Destruktor zwalniaj�cy pami��
	*/
    ~BuddhabrotRenderer();
    BuddhabrotRenderer(const BuddhabrotRenderer& other) = delete;
    BuddhabrotRenderer& operator=(const BuddhabrotRenderer& other) = delete;
    BuddhabrotRenderer(BuddhabrotRenderer&& other) noexcept = delete;
    BuddhabrotRenderer& operator=(BuddhabrotRenderer&& other) noexcept  = delete;

	/**
	 * @brief obliczenie warstwy fraktala
	 * @param samples ilo�� pr�bek
	 * @param iterations ilo�� iteracji warstwy
	 * @param r mno�nik sk�adowej czerwonej
	 * @param g mno�nik sk�adowej zielonej
	 * @param b mno�nik sk�adowej niebieskiej
	 * @param correction_a warto�� wsp�czynnika a
	 * @param correction_gamma warto�� korekcji gamma
	*/
	void addLayer(uint64_t samples, uint32_t iterations,
		float r, float g, float b,
		float correction_a, float correction_gamma);

	/**
	 * @brief zapis fraktala do pliku 
	 * @param filename nazwa pliku pod jak� ma by� zapisany obraz
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

