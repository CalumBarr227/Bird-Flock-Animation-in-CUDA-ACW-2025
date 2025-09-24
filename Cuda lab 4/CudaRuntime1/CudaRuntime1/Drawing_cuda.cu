#ifndef _DRAWING_CUDA_CU_
#define _DRAWING_CUDA_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

struct Bird 
{
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
    float maxSpeed;
};

#define NUM_BIRDS 10
#define MAX_SPEED 3.0f
#define PERCEPTION_RADIUS 0.25f
#define SEPARATION_WEIGHT 1.5f
#define ALIGNMENT_WEIGHT 1.0f
#define COHESION_WEIGHT 1.0f
#define BOUNDARY_SIZE 0.95f
#define BIRD_MASS 1.0f

Bird* global_birds = NULL;
float3* global_positions = NULL;
float3* global_colours = NULL;
float global_deltaTime = 0.01f;
bool global_firstRun = true;

__global__ void updateFlock(Bird* birds, int num_birds, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_birds) return;

    Bird& bird = birds[idx];

    bird.acceleration = make_float3(0.0f, 0.0f, 0.0f);

    float3 separationForce = make_float3(0.0f, 0.0f, 0.0f);
    int separationCount = 0;

    float3 alignmentForce = make_float3(0.0f, 0.0f, 0.0f);
    int alignmentCount = 0;

    float3 cohesionForce = make_float3(0.0f, 0.0f, 0.0f);
    int cohesionCount = 0;

    for (int i = 0; i < num_birds; i++) {
        if (i == idx) continue;

        Bird other = birds[i];
        float3 diff = bird.position - other.position;
        float d = length(diff);

        if (d > 0.0f && d < PERCEPTION_RADIUS) {
            separationForce -= (bird.position - other.position);
            separationCount++;

            alignmentForce += (other.velocity - bird.velocity);
            alignmentCount++;

            cohesionForce += (other.position - bird.position);
            cohesionCount++;
        }
    }

    if (alignmentCount > 0) {
        alignmentForce /= (float)alignmentCount;
    }

    if (cohesionCount > 0) {
        cohesionForce /= (float)cohesionCount;
    }

    float3 totalForce = SEPARATION_WEIGHT * separationForce +
        ALIGNMENT_WEIGHT * alignmentForce +
        COHESION_WEIGHT * cohesionForce;

    bird.acceleration = totalForce / bird.mass;

    bird.velocity += bird.acceleration * deltaTime;

    float speed = length(bird.velocity);
    if (speed > bird.maxSpeed) {
        bird.velocity = (bird.velocity / speed) * bird.maxSpeed;
    }

    bird.position += bird.velocity * deltaTime;

    if (bird.position.x < -BOUNDARY_SIZE) bird.position.x = -BOUNDARY_SIZE;
    if (bird.position.x > BOUNDARY_SIZE) bird.position.x = BOUNDARY_SIZE;
    if (bird.position.y < -BOUNDARY_SIZE) bird.position.y = -BOUNDARY_SIZE;
    if (bird.position.y > BOUNDARY_SIZE) bird.position.y = BOUNDARY_SIZE;
    if (bird.position.z < -BOUNDARY_SIZE) bird.position.z = -BOUNDARY_SIZE;
    if (bird.position.z > BOUNDARY_SIZE) bird.position.z = BOUNDARY_SIZE;
}

__global__ void initialiseBirds(Bird* birds, int num_birds, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_birds) return;

    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    birds[idx].position.x = (curand_uniform(&state) * 2.0f - 1.0f) * BOUNDARY_SIZE * 0.5f;
    birds[idx].position.y = (curand_uniform(&state) * 2.0f - 1.0f) * BOUNDARY_SIZE * 0.5f;
    birds[idx].position.z = (curand_uniform(&state) * 2.0f - 1.0f) * BOUNDARY_SIZE * 0.5f;

    birds[idx].velocity.x = (curand_uniform(&state) * 2.0f - 1.0f) * MAX_SPEED * 0.1f;
    birds[idx].velocity.y = (curand_uniform(&state) * 2.0f - 1.0f) * MAX_SPEED * 0.1f;
    birds[idx].velocity.z = (curand_uniform(&state) * 2.0f - 1.0f) * MAX_SPEED * 0.1f;

    birds[idx].acceleration = make_float3(0.0f, 0.0f, 0.0f);

    birds[idx].mass = BIRD_MASS;
    birds[idx].maxSpeed = MAX_SPEED;
}

__global__ void renderBirds(Bird* birds, int num_birds, float3* positions, float3* colours) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_birds) return;

    positions[idx] = birds[idx].position;

    float3 vel = normalize(birds[idx].velocity);
    colours[idx].x = fabsf(vel.x) * 0.5f + 0.5f;
    colours[idx].y = fabsf(vel.y) * 0.5f + 0.5f;
    colours[idx].z = fabsf(vel.z) * 0.5f + 0.5f;
}

__global__ void renderToTexture(int width, int height, Bird* birds, int num_birds, uchar4* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float4 colour = make_float4(0.0f, 0.0f, 0.1f, 1.0f);

    float2 screenPos = make_float2(
        (float)x / width * 2.0f - 1.0f,
        (float)y / height * 2.0f - 1.0f
    );

    for (int i = 0; i < num_birds; i++) {
        Bird bird = birds[i];

        float2 birdPos = make_float2(bird.position.x, bird.position.y);

        float dist = length(birdPos - screenPos);

        float birdRadius = 0.02f;
        if (dist < birdRadius) {
            float3 vel = normalize(bird.velocity);
            float3 birdColour;
            birdColour.x = fabsf(vel.x) * 0.5f + 0.5f;
            birdColour.y = fabsf(vel.y) * 0.5f + 0.5f;
            birdColour.z = fabsf(vel.z) * 0.5f + 0.5f;

            colour = make_float4(birdColour.x, birdColour.y, birdColour.z, 1.0f);
            break;
        }
    }

    output[idx] = make_uchar4(
        (unsigned char)(colour.x * 255),
        (unsigned char)(colour.y * 255),
        (unsigned char)(colour.z * 255),
        255
    );
}

extern "C" void setupBirdFlock(Bird** d_birds, float3** d_positions, float3** d_colours) {
    checkCudaErrors(cudaMalloc((void**)d_birds, NUM_BIRDS * sizeof(Bird)));
    checkCudaErrors(cudaMalloc((void**)d_positions, NUM_BIRDS * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)d_colours, NUM_BIRDS * sizeof(float3)));

    int blockSize = 256;
    int numBlocks = (NUM_BIRDS + blockSize - 1) / blockSize;

    unsigned int seed = (unsigned int)time(NULL);
    initialiseBirds << <numBlocks, blockSize >> > (*d_birds, NUM_BIRDS, seed);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void updateAndRenderBirds(Bird* d_birds, float3* d_positions, float3* d_colours, float deltaTime) {
    int blockSize = 256;
    int numBlocks = (NUM_BIRDS + blockSize - 1) / blockSize;

    updateFlock << <numBlocks, blockSize >> > (d_birds, NUM_BIRDS, deltaTime);
    checkCudaErrors(cudaGetLastError());

    renderBirds << <numBlocks, blockSize >> > (d_birds, NUM_BIRDS, d_positions, d_colours);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void cleanupBirdFlock(Bird* d_birds, float3* d_positions, float3* d_colours) {
    if (d_birds) checkCudaErrors(cudaFree(d_birds));
    if (d_positions) checkCudaErrors(cudaFree(d_positions));
    if (d_colours) checkCudaErrors(cudaFree(d_colours));
}

extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4* output) {

    if (global_firstRun)
    {
        setupBirdFlock(&global_birds, &global_positions, &global_colours);
        global_firstRun = false;
    }

    updateAndRenderBirds(global_birds, global_positions, global_colours, global_deltaTime);

    renderToTexture << <gridSize, blockSize >> > (width, height, global_birds, NUM_BIRDS, output);
    checkCudaErrors(cudaGetLastError());
}

#endif