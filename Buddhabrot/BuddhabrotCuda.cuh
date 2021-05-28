/**
* @file BuddhabrotCuda.cuh
*/

#pragma once
#include "BuddhabrotParameters.h"
#include <cuda_runtime_api.h>

/**
 * @brief wywołanie odpowiedniego kernela
 * @param grid_dim rozmiar siatki
 * @param block_dim rozmiar bloku
 * @param smem_size wielkość pamięci współdzielonej
 * @param dev_vec tablica z danymi
 * @param dev_tmp tymczasowa tablica
 * @param size rozmiar tablicy dev_vec
 * @param threads ilość wątków na blok
*/
void max_kernel_helper(dim3 grid_dim, dim3 block_dim, uint32_t smem_size, 
    uint32_t*& dev_vec, uint32_t*& dev_tmp, uint32_t size, uint32_t threads);

/**
 * @brief wywołanie kernela obliczającego fraktal
 * @param parameters paramatry fraktala
 * @param viewport parametry obrazu
 * @param dev_buddhabrot tablica z fraktalem na GPU
 * @return status operacji
*/
cudaError_t computeBuddhabrotCUDA(const BuddhabrotParameters& parameters,
    const BuddhabrotViewport& viewport, uint32_t* dev_buddhabrot);

/**
 * @brief obliczenie obrazu
 * @param parameters parametry fraktala
 * @param viewport parametry obrazu
 * @param dev_buddhabrot tablica z fraktalem na GPU
 * @param dev_image tablica z obrazem na GPU
 * @param r mnożnik składowej czerwonej
 * @param g mnożnik składowej zielonej
 * @param b mnożnik składowej niebieskiej
 * @param correction_a wartość współczynnika a
 * @param correction_gamma wartość korekcji gamma
 * @param subtractive_blending tryb blendingu
 * @param background_r mnożnik składowej czerwonej
 * @param background_g mnożnik składowej zielonej
 * @param background_b mnożnik składowej niebieskiej
 * @return status operacji
*/
cudaError_t computeLayerCUDA(const BuddhabrotParameters& parameters,
    const BuddhabrotViewport& viewport, uint32_t* &dev_buddhabrot,
    float* &dev_image, float r, float g, float b,
    float correction_a, float correction_gamma,
    bool subtractive_blending, float background_r,
    float background_g, float background_b);

/**
 * @brief pobranie obrazu z GPU
 * @param image tablica z obrazem
 * @param dev_image tablica z obrazem na GPU
 * @param viewport parametry obrazu
 * @param background_r mnożnik składowej czerwonej
 * @param background_g mnożnik składowej zielonej
 * @param background_b mnożnik składowej niebieskiej
 * @return status operacji
*/
cudaError_t getImageCUDA(uint8_t* &image, float* &dev_image,
    const BuddhabrotViewport& viewport,
    float background_r, float background_g, float background_b);

/**
 * @brief alokacja pamięci na GPU
 * @param dev_buddhabrot tablica z fraktalem na GPU
 * @param dev_image tablica z obrazem na GPU
 * @param viewport parametry obrazu
 * @return status operacji
*/
cudaError_t allocateMemoryCUDA(uint32_t*& dev_buddhabrot, float*& dev_image,
    const BuddhabrotViewport& viewport);

/**
 * @brief zwalnanie pamięci na GPU
 * @param dev_buddhabrot 
 * @param dev_image 
*/
void freeMemoryCUDA(uint32_t*& dev_buddhabrot, float*& dev_image);

/**
 * @brief wyczyszczenie buforów
 * @param dev_buddhabrot tablica z obrazem na GPU
 * @param dev_image tablica z obrazem na GPU
 * @param viewport parametry obrazu
 * @return status operacji
*/
cudaError_t clearBufferCUDA(uint32_t*& dev_buddhabrot, float*& dev_image,
    const BuddhabrotViewport& viewport);
