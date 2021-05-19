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

#include "stb_image_write.h"

//template<unsigned int blockSize>
__device__ void wrapReduce(volatile uint32_t* sdata, int tid) {
    //if (blockSize >= 64);
    sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

//template<unsigned int blockSize>
__global__ void max_kernel(uint32_t* g_data, uint32_t* g_tmp, uint32_t size) {
    extern __shared__ uint32_t sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    uint32_t x1 = (i < size ? g_data[i] : 0);
    uint32_t x2 = (i + blockDim.x < size ? g_data[i + blockDim.x] : 0);
    sdata[tid] = max(x1, x2);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) wrapReduce(sdata, tid);

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

__global__ void save_kernel(float* g_data, uint8_t* image, uint32_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    image[i] = min(255.f, max(0.f, g_data[i] * 255.f));
}

/*float norm_mul = 1.f / max_value;
for (size_t i = 0; i < num_pixels; ++i) {
    float normalized = compute_buffer[i] * norm_mul;
    float corrected = correction_a * std::pow(normalized, correction_gamma);
    FloatRGB& output = image_buffer[i];
    output.r += corrected * r;
    output.g += corrected * g;
    output.b += corrected * b;
}*/

/*
uint8_t* image = new uint8_t[num_pixels * 3];
for (size_t i = 0; i < num_pixels; ++i) {
    image[i * 3] = std::min(255.f, std::max(0.f, image_buffer[i].r * 255.f));
    image[i * 3 + 1] = std::min(255.f, std::max(0.f, image_buffer[i].g * 255.f));
    image[i * 3 + 2] = std::min(255.f, std::max(0.f, image_buffer[i].b * 255.f));
}
*/






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
    const BuddhabrotViewport& viewport, uint32_t* buffer) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint32_t* dev_buffer;
    cudaError_t cudaStatus;
    curandState* dev_curand_states;

    cudaStream_t computeStream;
    cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking); //cudaStreamNonBlocking cudaStreamDefault

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
    std::cout << "\tsetup_kernel completed in " << milliseconds << "ms\n";
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "setup_kernel failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setup_kernel!\n", cudaStatus);
        goto Error;
    }

    // Result buffer allocation
	cudaStatus = cudaMalloc((void**) &dev_buffer,
        sizeof(uint32_t) * viewport.width * viewport.height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
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
            dev_buffer, dev_curand_states, needed_repeats, parameters, viewport);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\tbuddhabrot_kernel completed in " << milliseconds << "ms (launched "<< num_launches<<" times)\n";
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "buddhabrot_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching buddhabrot_kernel!\n", cudaStatus);
        goto Error;
    }

    //vvv
    //uint32_t max_value;
    //cudaMemcpyFromSymbol(&max_value, "d_max_value", sizeof(max_value), 0, cudaMemcpyDeviceToHost);

    cudaFree(dev_curand_states);

    uint32_t max_value;
    uint32_t size = viewport.width * viewport.height;
    uint32_t tmp_size = size / 2;
    uint32_t TPB = 1024;
    uint32_t TPBx2 = 2 * TPB;

    uint32_t* dev_tmp;
    cudaStatus = cudaMalloc((void**) &dev_tmp, size * sizeof(uint32_t));
    cudaStatus = cudaMemset(dev_tmp, 0, size * sizeof(uint32_t));

    dim3 gridDim((size + TPBx2 - 1) / TPBx2);
    dim3 blockDim(TPB);
    uint32_t sharedmem = TPB * sizeof(uint32_t);
    //std::cout << "" << gridDim.x << std::endl;
    //std::cout << "" << blockDim.x << std::endl;
    //std::cout << "" << sharedmem << std::endl;

    max_kernel << <gridDim, blockDim, sharedmem >> > (dev_buffer, dev_tmp, size);
    for (uint32_t i = TPB; i < size; i *= TPB) {
        max_kernel << <gridDim, blockDim, sharedmem >> > (dev_tmp, dev_tmp, size);
    }

    cudaStatus = cudaMemcpy(&max_value, dev_tmp, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Max value: " << max_value << std::endl;

    cudaFree(dev_tmp);

    uint8_t* dev_image;
    uint32_t image_size = viewport.width * viewport.height * 3;
    cudaStatus = cudaMalloc((void**) &dev_image, image_size * sizeof(uint8_t));
    cudaStatus = cudaMemset(dev_image, 0, image_size * sizeof(uint8_t));

    float* dev_fimage;
    cudaStatus = cudaMalloc((void**) &dev_fimage, image_size * sizeof(float));

    float norm_mul = 1.f / max_value;
    std::cout << "norm_mul: " << norm_mul << std::endl;
    dim3 gridDim2((image_size + TPB - 1) / TPB);
    dim3 blockDim2(TPB);
    layer_kernel<<<gridDim2, blockDim2>>>(dev_buffer, dev_fimage, image_size, norm_mul, 1, 0, 0, 2, 1);
    save_kernel<< <gridDim2, blockDim2 >> > (dev_fimage, dev_image, image_size);

    uint8_t* image = new uint8_t[image_size];
    cudaStatus = cudaMemcpy(image, dev_image, image_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_fimage);

    int status = stbi_write_png("test.png", viewport.width, viewport.height, 3, image, viewport.width * 3);
    if (status == 0) {
        std::cerr << "Error while saving image to file\n";
    }
    delete[] image;

    //^^^

    // Copying the results back to host
    cudaStatus = cudaMemcpy(buffer, dev_buffer,
        sizeof(uint32_t) * viewport.width * viewport.height, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\tComputing a layer (CPU time): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

Error:
    cudaFree(dev_buffer);
    return cudaStatus;
}

