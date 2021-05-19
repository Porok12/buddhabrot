#include "buddhabrotCuda.cuh"

#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <chrono>
#include <iostream>
#include <algorithm>

__global__ void setup_kernel(curandState* states, uint64_t seed) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void buddhabrot_kernel(uint32_t* buffer, curandState* states,
    uint32_t repeats, BuddhabrotParameters parameters, BuddhabrotViewport viewport) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState rng_state = states[tid];
    float c_re, c_im, z_re, z_im, tmp_re, tmp_im;
    uint32_t iterations;
    // iterations loop
    for (uint32_t i = 0; i < repeats; ++i) {
        c_re = -2 + 4 * curand_uniform(&rng_state);
        c_im = -2 + 4 * curand_uniform(&rng_state);
        iterations = 0;
        z_re = parameters.start_re;
        z_im = parameters.start_im;
        // first iteration without affecting the global memory
        // checking if it escapes
        while ((z_re * z_re + z_im * z_im) < 4 && iterations < parameters.iterations) {
            tmp_re = z_re * z_re - z_im * z_im + c_re;
            tmp_im = 2 * z_re * z_im + c_im;
            z_re = tmp_re;
            z_im = tmp_im;
            iterations++;
        }
        // if interation escapes, increment counters in the buffer
        if (iterations < parameters.iterations) {
            z_re = parameters.start_re;
            z_im = parameters.start_im;
            for (uint32_t i = 0; i < iterations; ++i) {
                tmp_re = z_re * z_re - z_im * z_im + c_re;
                tmp_im = 2 * z_re * z_im + c_im;
                z_re = tmp_re;
                z_im = tmp_im;
                // transform complex plane into pixel coordinates
                float xb_re = z_re - viewport.b1;
                float xb_im = z_im - viewport.b2;
                int pixel_x = xb_re * viewport.a11 + xb_im * viewport.a12;
                int pixel_y = xb_re * viewport.a21 + xb_im * viewport.a22;
                // check if resulting position is in bounds and add the value to counter
                if (pixel_x >= 0 && pixel_y >= 0 &&
                    pixel_x < viewport.width && pixel_y < viewport.height) {
                    atomicAdd(&buffer[pixel_x + pixel_y * viewport.width], 1.0f);
                }
            }
        }
    }
    
    // save rng back to global so that next call can reuse it
    states[tid] = rng_state;
}


template<unsigned int blockSize>
__device__ void wrapReduce(volatile uint32_t* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

template<unsigned int blockSize>
__global__ void max_kernel(uint32_t* g_data, uint32_t* g_tmp, uint32_t size) {
    extern __shared__ uint32_t sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    uint32_t x1 = (i < size ? g_data[i] : 0);
    uint32_t x2 = (i + blockDim.x < size ? g_data[i + blockDim.x] : 0);
    sdata[tid] = max(x1, x2);
    __syncthreads();

    /*for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }*/
    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); }
        __syncthreads();
    }

    if (tid < 32) wrapReduce<blockSize>(sdata, tid);

    if (tid == 0) {
        g_tmp[blockIdx.x] = sdata[0];
    }
}

__global__ void layer_kernel(uint32_t* g_data, float* image, uint32_t size, float norm_mul, float r, float g, float b, float correction, float gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    float normalized = g_data[i / 3] * norm_mul;
    float corrected = correction * powf(normalized, gamma);

    switch (i % 3) {
    case 0:
        image[i] += corrected * r;
        break;
    case 1:
        image[i] += corrected * g;
        break;
    case 2:
        image[i] += corrected * b;
        break;
    }
}

__global__ void save_kernel(uint8_t* image, const float* data, const uint32_t size,
    float background_r, float background_g, float background_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    switch (i % 3) {
    case 0:
        image[i] = min(255.f, max(0.f, (background_r + data[i]) * 255.f));
        break;
    case 1:
        image[i] = min(255.f, max(0.f, (background_g + data[i]) * 255.f));
        break;
    case 2:
        image[i] = min(255.f, max(0.f, (background_b + data[i]) * 255.f));
        break;
    }
}


void max_kernel_helper(dim3 grid_dim, dim3 block_dim, uint32_t smem_size, uint32_t*& dev_vec, uint32_t*& dev_tmp, uint32_t size, uint32_t threads) {
    switch (threads) {
    case 1024:
        max_kernel<1024> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 512:
        max_kernel< 512> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 256:
        max_kernel< 256> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 128:
        max_kernel< 128> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 64:
        max_kernel<  64> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 32:
        max_kernel<  32> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 16:
        max_kernel<  16> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 8:
        max_kernel<   8> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 4:
        max_kernel<   4> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 2:
        max_kernel<   2> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    case 1:
        max_kernel<   1> << <grid_dim, block_dim, smem_size >> > (dev_vec, dev_tmp, size);
        break;
    default:
        std::cerr << "Unhadled threads" << std::endl;
    }
}

BuddhabrotViewport::BuddhabrotViewport(uint32_t width, uint32_t height,
    float center_re, float center_im, float scale, float rotation, 
    uint64_t max_repeats_per_thread, uint64_t blocks_per_multiprocessor)
	: width(width), height(height), 
    max_repeats_per_thread(max_repeats_per_thread), blocks_per_multiprocessor(blocks_per_multiprocessor) {
	float viewport_ratio = (float) width / height;
	float viewport_ratio_sqrt = std::sqrt(viewport_ratio);
	float mul_x = scale * viewport_ratio_sqrt;
	float mul_y = scale / viewport_ratio_sqrt;
	float rot_cos = std::cos(rotation);
	float rot_sin = std::sin(rotation);
	float x_span_re = rot_cos * mul_x;
	float x_span_im = -rot_sin * mul_x;
	float y_span_re = rot_sin * mul_y;
	float y_span_im = rot_cos * mul_y;
	float base_re = center_re - 0.5f * (x_span_re + y_span_re);
    float base_im = center_im - 0.5f * (x_span_im + y_span_im);
    float increment_x_re = x_span_re / width;
    float increment_x_im = x_span_im / width;
    float increment_y_re = y_span_re / height;
    float increment_y_im = y_span_im / height;
    float inv_det = 1.f / (increment_y_im * increment_x_re - increment_y_re * increment_x_im);
    a11 = increment_y_im * inv_det;
    a12 = -increment_y_re * inv_det;
    a21 = -increment_x_im * inv_det;
    a22 = increment_x_re * inv_det;
    b1 = base_re;
    b2 = base_im;
}

cudaError_t computeBuddhabrotCUDA(const BuddhabrotParameters& parameters,
    const BuddhabrotViewport& viewport, uint32_t* dev_buddhabrot) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cudaError_t cudaStatus;
    curandState* dev_curand_states;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Computing a layer, iterations: " << parameters.iterations << ", samples: " << parameters.samples << "\n";

    // Computing amount of threads, blocks and repeats

    // These 3 parameters make computation faster when they're bigger
    // but only to some extent. Obviously threads per block can't be too big,
    uint64_t threads_per_block = 256;
    uint64_t max_repeats_per_thread = viewport.max_repeats_per_thread;
    uint64_t blocks_per_multiprocessor = viewport.blocks_per_multiprocessor;

    int num_multiprocessors;
    cudaStatus = cudaDeviceGetAttribute(&num_multiprocessors,
        cudaDevAttrMultiProcessorCount, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed!");
        goto Error;
    }

    uint64_t num_blocks = num_multiprocessors * blocks_per_multiprocessor;
    uint64_t num_threads = num_blocks * threads_per_block;
    uint64_t num_samples = (parameters.samples / num_threads) * num_threads;
    std::cout << "\t\tnum blocks: " << num_blocks << "\n";
    std::cout << "\t\tnum threads: " << num_threads << "\n";
    std::cout << "\t\tnum samples: " << num_samples << "\n";


    // RNG allocation
    cudaStatus = cudaMalloc(&dev_curand_states, num_threads * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    float milliseconds = 0;

    cudaEventRecord(start);
    setup_kernel << <num_blocks, threads_per_block>> > (
        dev_curand_states, time(NULL));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "setup_kernel failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    std::cout << "\tsetup_kernel completed in " << milliseconds << "ms\n";

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setup_kernel!\n", cudaStatus);
        goto Error;
    }

    // Computing the fractal
    uint64_t samples_left_to_compute = num_samples;
    uint64_t num_launches = 0;
    cudaEventRecord(start);
    while (samples_left_to_compute > 0) {
        uint64_t needed_repeats = samples_left_to_compute / num_threads;
        needed_repeats = std::min(needed_repeats, max_repeats_per_thread);
        num_launches++;
        samples_left_to_compute -= needed_repeats * num_threads;
        buddhabrot_kernel << <num_blocks, threads_per_block >> > (
            dev_buddhabrot, dev_curand_states, needed_repeats, parameters, viewport
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "buddhabrot_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    std::cout << "\tbuddhabrot_kernel completed in " << milliseconds << "ms (launched " << num_launches << " times)\n";
    
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching buddhabrot_kernel!\n", cudaStatus);
        goto Error;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\tComputing layer (CPU time): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

Error:
    cudaFree(dev_curand_states);
    return cudaStatus;
}

cudaError_t computeLayerCUDA(
    const BuddhabrotParameters& parameters,
    const BuddhabrotViewport& viewport, 
    uint32_t* &dev_buddhabrot, 
    float* &dev_image,
    float r, float g, float b,
    float correction_a, float correction_gamma,
    bool subtractive_blending, float background_r,
    float background_g, float background_b
) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cudaError_t cudaStatus;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    uint32_t image_size = viewport.width * viewport.height * 3;

    uint32_t max_value;
    uint32_t size = viewport.width * viewport.height;
    uint32_t tmp_size = size / 2;
    uint32_t TPB = 512;
    uint32_t TPBx2 = 2 * TPB;

    // Calculate max value using kernel
    uint32_t* dev_tmp;
    cudaStatus = cudaMalloc((void**) &dev_tmp, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }
    cudaStatus = cudaMemset(dev_tmp, 0, size * sizeof(uint32_t));

    dim3 gridDim((size + TPBx2 - 1) / TPBx2);
    dim3 blockDim(TPB);
    uint32_t sharedmem = TPB * sizeof(uint32_t);

    cudaEventRecord(start);
    max_kernel_helper(gridDim, blockDim, sharedmem, dev_buddhabrot, dev_tmp, size, TPB);
    for (uint32_t i = TPB; i < size; i *= TPB) {
        max_kernel_helper(gridDim, blockDim, sharedmem, dev_tmp, dev_tmp, size, TPB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "max_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    std::cout << "\tmax_kernel completed in " << milliseconds << "ms\n";
    
    cudaStatus = cudaMemcpy(&max_value, dev_tmp, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "\tlayer max: " << max_value << "\n";

    // Calculate layer
    if (max_value > 0) {
        float norm_mul = 1.f / max_value;
        dim3 gridDim2((image_size + TPB - 1) / TPB);
        dim3 blockDim2(TPB);
        cudaEventRecord(start);
        if (subtractive_blending) {
            r = r - background_r;
            g = g - background_g;
            b = b - background_b;
            layer_kernel<<< gridDim2, blockDim2 >>> (
                dev_buddhabrot, dev_image, image_size, norm_mul, r, g, b, correction_a, correction_gamma
                );
        }
        else {
            layer_kernel<<< gridDim2, blockDim2 >>> (
                dev_buddhabrot, dev_image, image_size, norm_mul, r, g, b, correction_a, correction_gamma
            );
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "layer_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        std::cout << "\layer_kernel completed in " << milliseconds << "ms\n";
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\tComputing layer colors (CPU time): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

Error:
    cudaFree(dev_tmp);
    return cudaStatus;
}

cudaError_t getImageCUDA(uint8_t* &image, float* &dev_image,
    const BuddhabrotViewport& viewport,
    float background_r, float background_g, float background_b) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    uint32_t TPB = 512;
    cudaError_t cudaStatus;
    uint32_t image_size = viewport.width * viewport.height * 3;

    uint8_t* dev_tmp;
    cudaStatus = cudaMalloc((void**) &dev_tmp, image_size * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    dim3 gridDim((image_size + TPB - 1) / TPB);
    dim3 blockDim(TPB);
    cudaEventRecord(start);
    save_kernel << <gridDim, blockDim >> > (dev_tmp, dev_image, image_size,
        background_r, background_g, background_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "save_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    std::cout << "\tsave_kernel completed in " << milliseconds << "ms\n";

    cudaStatus = cudaMemcpy(image, dev_tmp, image_size * sizeof(uint8_t),
        cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\tComputing image (CPU time): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

Error:
    cudaFree(dev_tmp);
    return cudaStatus;
}

cudaError_t allocateMemoryCUDA(uint32_t*& dev_buddhabrot, float*& dev_image,
    const BuddhabrotViewport& viewport) {
    cudaError_t cudaStatus;
    uint32_t size = viewport.width * viewport.height;
    uint32_t channels = 3;

    cudaStatus = cudaMalloc((void**)&dev_image, size * channels * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_image failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_image, 0, size * channels * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset dev_image failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_buddhabrot, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_buddhabrot failed!\n");
        goto Error;
    }

Error:
    return cudaStatus;
}

void freeMemoryCUDA(uint32_t*& dev_buddhabrot, float*& dev_image) {
    cudaFree(dev_buddhabrot);
    cudaFree(dev_image);
}

cudaError_t clearBufferCUDA(uint32_t* &dev_buddhabrot, float* &dev_image, const BuddhabrotViewport& viewport) {
    cudaError_t cudaStatus; 
    uint32_t size = viewport.width * viewport.height;
    uint32_t channels = 3;

    cudaStatus = cudaMemset(dev_buddhabrot, 0, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset dev_buddhabrot failed!\n");
        goto Error;
    }
Error:
    return cudaStatus;
}
