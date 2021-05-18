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

BuddhabrotViewport::BuddhabrotViewport(uint32_t width, uint32_t height,
    float center_re, float center_im, float scale, float rotation)
	: width(width), height(height) {
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
    uint64_t max_repeats_per_thread = 256;
    uint64_t blocks_per_multiprocessor = 8;

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
	cudaStatus = cudaMalloc((void**)&dev_buffer,
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
    cudaFree(dev_curand_states);
    return cudaStatus;
}
