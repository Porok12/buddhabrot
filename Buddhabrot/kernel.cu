
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mandelbrot.h"

//TODO: Usunąć nieużywane
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <complex>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define USE_GPU

using uint = unsigned int;
using uint8 = unsigned char;

cudaError_t buddhabrotCuda(
    float* data,
    const uint width,
    const uint height,
    const float minX,
    const float minY,
    const float maxX,
    const float maxY,
    const float ratioX,
    const float ratioY,
    const uint iterations,
    const uint samples,
    const uint loop = 0);

uint levelHost(
    const float p_re,
    const float p_im,
    const uint nIter,
    float* seqX,
    float* seqY);

__global__ void setup_kernel(curandState* state, uint64_t seed) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__device__ float getRandom(uint64_t seed, int tid, int threadCallCount) {
    curandState s;
    //curand_init(seed + tid + threadCallCount, 0, 0, &s);
    curand_init(clock() + tid + threadCallCount, 0, 0, &s);
    return curand_uniform(&s);
}

__device__ int level(
    const float p_re, 
    const float p_im,
    const unsigned int nIter,
    float *seqX,
    float *seqY) {
    float z_re = 0;
    float z_im = 0;
    int iteration = 0;
    float tmp_re, tmp_im;

    do {
        tmp_re = z_re * z_re - z_im * z_im + p_re;
        tmp_im = 2 * z_re * z_im + p_im;
        z_re = tmp_re;
        z_im = tmp_im;
        seqX[iteration] = tmp_im;
        seqY[iteration] = tmp_re;
        iteration++;
    } while ((z_re * z_re + z_im * z_im) < 4 && iteration < nIter);

    if (iteration >= nIter || iteration < 0) {
        iteration = 0;
    }

    return iteration;
}

[[deprecated]]
__global__ void buddhabrotKernelXXX(
    float* data,
    const int width,
    const int height,
    const float minX,
    const float minY,
    const float maxX,
    const float maxY,
    const float ratioX, // (maxX - minX) / width;
    const float ratioY, // (maxY - minY) / height;
    const unsigned int iterations,
    float* sX,
    float* sY,
    const uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /*int xx = tid % width;
    int yy = tid / width;

    //data[xx + yy * width] = yy * 255 / width;
    data[tid] = getRandom(seed, 2 * tid, 0) * 255;
    return;*/

    float* seqX = sX + (tid * iterations);
    float* seqY = sY + (tid * iterations);

    float p_re = -2 + (2 - -2) * getRandom(seed, 2 * tid, 0);
    float p_im = -2 + (2 - -2) * getRandom(seed, 2 * tid, 1);
    //float* seqX = new float[iterations+1];
    //float* seqY = new float[iterations+1];
    int l = level(p_re, p_im, iterations, seqX, seqY);

    int x, y;
    for (int j = 0; j < l; j++) {
        if (minX < seqX[j] && maxX > seqX[j] && minY < seqY[j] && maxY > seqY[j])
        {
            x = (seqX[j] - minX) / ratioX;
            y = (seqY[j] - minY) / ratioY;
            atomicAdd(&data[x + y * width], 1.0f);
        }
    }

    //delete[] seqX;
    //delete[] seqY;
}

__global__ void buddhabrot_kernel(
    float* data,
    const int width,
    const int height,
    const float minX,
    const float minY,
    const float maxX,
    const float maxY,
    const float ratioX,
    const float ratioY,
    const unsigned int iterations,
    float* sX,
    float* sY,
    curandState* globalState,
    int skip) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float* seqX = sX + (tid * iterations);
    float* seqY = sY + (tid * iterations);
    curandState localState = globalState[tid];
    for (int i = 0; i < skip * 2; i++) {
        curand_uniform(&localState);
    }
    float p_re = -2 + (2 - -2)*curand_uniform(&localState);
    float p_im = -2 + (2 - -2)*curand_uniform(&localState);
    int l = level(p_re, p_im, iterations, seqX, seqY);

    int x, y;
    for (int j = 0; j < l; j++) {
        if (minX < seqX[j] && maxX > seqX[j] && minY < seqY[j] && maxY > seqY[j])
        {
            x = (seqX[j] - minX) / ratioX;
            y = (seqY[j] - minY) / ratioY;
            atomicAdd(&data[x + y * width], 1.0f);
        }
    }
}

void gpu_heatmap(
    float* data,
    const unsigned int width,
    const unsigned int height,
    const unsigned int iterations,
    int samples) {
    std::cout << "--------------------------------" << std::endl;

    float minX = -1.5;
    float minY = -2.0;
    float maxX = 1.5;
    float maxY = 1.0;

    float ratioX = (maxX - minX) / width;
    float ratioY = (maxY - minY) / height;

    float min_r = -2;
    float min_i = -2;
    float max_r = 2;
    float max_i = 2;

    // karta 4GB obsługuje sampleLimit = 400000 (dla max iteracji 800); dla mniejszych trzeba zmniejszyć
    uint sampleLimit = 100000;
    if (samples > sampleLimit) {
        uint loops = samples / sampleLimit;
        buddhabrotCuda(data, width, height, minX, minY, maxX, maxY, ratioX, ratioY, iterations, sampleLimit, loops);
        /*while (samples > 0) {
            std::cout << "(" << loops - samples / sampleLimit << "/" << loops << ") " << std::endl;
            buddhabrotCuda(data, width, height, minX, minY, maxX, maxY, ratioX, ratioY, iterations, sampleLimit);
            samples -= sampleLimit;
        }*/
    }
    else {
        buddhabrotCuda(data, width, height, minX, minY, maxX, maxY, ratioX, ratioY, iterations, samples);
    }
}

float* cpu_heatmap(
    float* data,
    const unsigned int width,
    const unsigned int height,
    const unsigned int iterations,
    const unsigned int samples);

uint8* generateBudhabrot(
    uint width,
    uint height,
    uint red_iter,
    uint blue_iter,
    uint green_iter,
    uint samples) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    float* h_red = new float[width * height];
    std::fill(h_red, h_red + width * height, 0);

    float* h_green = new float[width * height];
    std::fill(h_green, h_green + width * height, 0);

    float* h_blue = new float[width * height];
    std::fill(h_blue, h_blue + width * height, 0);

#ifdef USE_GPU
    gpu_heatmap(h_red, width, height, red_iter, samples);
    gpu_heatmap(h_green, width, height, green_iter, samples);
    gpu_heatmap(h_blue, width, height, blue_iter, samples);
#else
    cpu_heatmap(h_red, width, height, red_iter, samples);
    cpu_heatmap(h_green, width, height, green_iter, samples);
    cpu_heatmap(h_blue, width, height, blue_iter, samples);
#endif

    auto biggest = h_red[0];
    auto smalest = h_red[0];
    auto r_max = h_red[0];
    auto b_max = h_blue[0];
    auto g_max = h_green[0];
    auto r_min = h_red[0];
    auto b_min = h_blue[0];
    auto g_min = h_green[0];
    for (int i = 0; i < width * height; i++) {
        if (h_red[i] > biggest) { biggest = h_red[i]; }
        if (h_red[i] < smalest) { smalest = h_red[i]; }
        if (h_green[i] > biggest) { biggest = h_green[i]; }
        if (h_green[i] < smalest) { smalest = h_green[i]; }
        if (h_blue[i] > biggest) { biggest = h_blue[i]; }
        if (h_blue[i] < smalest) { smalest = h_blue[i]; }
        if (h_red[i] > r_max) { r_max = h_red[i]; }
        if (h_blue[i] > b_max) { b_max = h_blue[i]; }
        if (h_green[i] > g_max) { g_max = h_green[i]; }
        if (h_red[i] < r_min) { r_min = h_red[i]; }
        if (h_blue[i] < b_min) { b_min = h_blue[i]; }
        if (h_green[i] < g_min) { g_min = h_green[i]; }
    }

    std::cout << "biggest: " << biggest << ", smalest: " << smalest << std::endl;
    std::cout << "r_max: " << r_max << ", r_min: " << r_min << std::endl;
    std::cout << "b_max: " << b_max << ", b_min: " << b_min << std::endl;
    std::cout << "g_max: " << g_max << ", g_min: " << g_min << std::endl;

    int index, lvl;
    uint8* image = new uint8[width * height * 3];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            index = (i * width + j) * 3;

            image[index + 0] = ((h_red[j + i * width] - r_min) / (r_max - r_min)) * 255.0f * 2;
            image[index + 1] = ((h_green[j + i * width] - g_min) / (g_max - g_min)) * 255.0f * 2;
            image[index + 2] = ((h_blue[j + i * width] - b_min) / (b_max - g_min)) * 255.0f * 2;

            image[index + 0] = image[index + 0] > 255 ? 255 : image[index + 0];
            image[index + 1] = image[index + 1] > 255 ? 255 : image[index + 1];
            image[index + 2] = image[index + 2] > 255 ? 255 : image[index + 2];
        }
    }

    delete[] h_red;
    delete[] h_green;
    delete[] h_blue;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    return image;
}

void buddhabrot_to_file(
    std::string file_name,
    uint img_width, 
    uint img_height,
    uint red_iter,
    uint blue_iter,
    uint green_iter,
    uint samples) {

    file_name.append(".png");
    uint8* image = generateBudhabrot(
        img_width, 
        img_height, 
        red_iter, 
        blue_iter, 
        green_iter, 
        samples);
    int status3 = stbi_write_png(file_name.c_str(), img_width, img_height, 3, image, img_width * 3);
    delete[] image;
}

int main(int argc, char* argv[])
{
    for (int i = 0; i < argc; i++)
    {
        fprintf(stdout, "%d: %s\n", i, argv[i]);
    }

    uint img_width = 600;
    uint img_height = 600;
    std::string file_name = "buddhabrot";
    uint red_iter = 800;
    uint blue_iter = 50;
    uint green_iter = 200;
    uint samples = 5000000;

    buddhabrot_to_file(
        file_name, 
        img_width, 
        img_height,
        red_iter,
        blue_iter,
        green_iter,
        samples);

    return EXIT_SUCCESS;
}

cudaError_t buddhabrotCuda(
    float* data,
    const uint width,
    const uint height,
    const float minX,
    const float minY,
    const float maxX,
    const float maxY,
    const float ratioX,
    const float ratioY,
    const uint iterations,
    const uint samples,
    const uint loops)
{
    float* dev_data = 0;
    float* dev_seqX = 0;
    float* dev_seqY = 0;
    curandState* dev_curand_states;
    cudaError_t cudaStatus;
    uint data_size = width * height * sizeof(float);
    uint seq_size = (iterations + 1) * samples * sizeof(float);
    uint threads = 10;
    uint blocks = samples / 10;
    uint threadCount = blocks * threads;

    cudaStream_t computeStream;
    cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking); //cudaStreamNonBlocking cudaStreamDefault

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Ustawienie seed-a
    cudaStatus = cudaMalloc(&dev_curand_states, threadCount * sizeof(curandState));
    setup_kernel << <blocks, threads >> > (dev_curand_states, time(NULL));

    // Alokowanie
    cudaStatus = cudaMalloc((void**) &dev_data, data_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &dev_seqX, seq_size);
    cudaStatus = cudaMalloc((void**) &dev_seqY, seq_size);

    // Ustawienie wartości
    cudaStatus = cudaMemset(dev_seqX, 0, seq_size);
    cudaStatus = cudaMemset(dev_seqY, 0, seq_size);
    cudaStatus = cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed\n!");
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (uint i = 0; i <= loops; i++)
    {
        //setup_kernel << <blocks, threads >> > (dev_curand_states, time(NULL));
        buddhabrot_kernel << <blocks, threads >> > (
            dev_data,
            width,
            height,
            minX,
            minY,
            maxX,
            maxY,
            ratioX,
            ratioY,
            iterations,
            dev_seqX,
            dev_seqY,
            dev_curand_states,
            i);
        //cudaDeviceSynchronize();
        //cudaStreamSynchronize(computeStream);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    fprintf(stdout, "\tTime (gpu): %f ms\n", milliseconds);

    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "buddhabrotKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching buddhabrotKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(data, dev_data, data_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\tTime (cpu): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

Error:
    cudaFree(dev_data);
    cudaFree(dev_seqX);
    cudaFree(dev_seqY);
    cudaFree(dev_curand_states);

    return cudaStatus;
}

uint levelHost(
    const float p_re,
    const float p_im,
    const uint nIter,
    float* seqX,
    float* seqY) {
    float z_re = 0;
    float z_im = 0;
    int iteration = 0;
    float tmp_re, tmp_im;

    do {
        tmp_re = z_re * z_re - z_im * z_im + p_re;
        tmp_im = 2 * z_re * z_im + p_im;
        z_re = tmp_re;
        z_im = tmp_im;
        seqX[iteration] = tmp_im;
        seqY[iteration] = tmp_re;
        iteration++;
    } while ((z_re * z_re + z_im * z_im) < 4 && iteration < nIter);

    if (iteration >= nIter) {
        iteration = 0;
    }

    return iteration;
}

float* cpu_heatmap(
    float* data,
    const unsigned int width,
    const unsigned int height,
    const unsigned int iterations,
    const unsigned int samples) {
    float minX = -1.5;
    float minY = -2.0;
    float maxX = 1.5;
    float maxY = 1.0;

    float ratioX = (maxX - minX) / width;
    float ratioY = (maxY - minY) / height;

    float min_r = -2;
    float min_i = -2;
    float max_r = 2;
    float max_i = 2;

    std::random_device rd;
    std::uniform_real_distribution<float> dist_r(min_r, max_r);
    std::uniform_real_distribution<float> dist_i(min_i, max_i);

    float* seqX = new float[iterations];
    std::fill(seqX, seqX + iterations, 0);
    float* seqY = new float[iterations];
    std::fill(seqY, seqY + iterations, 0);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int x, y;
    float x_dist, y_dist;
    for (int i = 0; i < samples; i++) {
        float p_re = dist_r(rd);
        float p_im = dist_i(rd);
        int l = levelHost(p_re, p_im, iterations, seqX, seqY);

        for (int j = 0; j < l; j++) {
            if (minX < seqX[j] && maxX > seqX[j] && minY < seqY[j] && maxY > seqY[j])
            {
                x = std::floor((seqX[j] - minX) / ratioX);
                y = std::floor((seqY[j] - minY) / ratioY);
                data[x + y * width] += 1;
            }
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time (cpu): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    delete[] seqX;
    delete[] seqY;

    return NULL;
}