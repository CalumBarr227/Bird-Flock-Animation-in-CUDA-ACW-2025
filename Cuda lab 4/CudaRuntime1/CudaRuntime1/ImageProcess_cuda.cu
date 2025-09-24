/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;


cudaArray* d_imageArray = 0;

cudaTextureObject_t rgbaTexdImage;


__global__ void d_render(uchar4* d_output, uint width, uint height, float tx,
    float ty, float scale, float cx, float cy,
    cudaTextureObject_t texObj) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;

    float2 T = { 20,10 };
    x += T.x;
    y += T.y;
    
    //float2 S = { 1.2, 0.5 };
    //x *= S.x;
    //y *= S.y;

    float angle = 0.5;
    float rx = x * cos(angle) - y * sin(angle);
    float ry = x * sin(angle) + y * cos(angle);



    //float u = (x - cx) * scale + cx + tx;
    //float v = (y - cy) * scale + cy + ty;

    //float u = (x - cx) * scale + cx;
    //float v = (y - cy) * scale + cy;

    float u = (x - cx) * scale + cx + tx;
    float v = (y - cy) * scale + cy + ty;

    if ((x < width) && (y < height)) {
        // write output color
        float c = tex2D<float>(texObj, u, v);




        //d_output[i] = make_uchar4(0xff, 0, 0, 0);
        //d_output[i] = make_uchar4(0, 0xff, 0, 0);
        //d_output[i] = make_uchar4(0, 0, 0xff, 0);
        d_output[i] = make_uchar4(0, 0, c * 0xff, 0);


    }
}


extern "C" void initTexture(int imageWidth, int imageHeight, uchar * h_data) {
    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(
        cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight));
    checkCudaErrors(cudaMemcpy2DToArray(
        d_imageArray, 0, 0, h_data, imageWidth * sizeof(uchar),
        imageWidth * sizeof(uchar), imageHeight, cudaMemcpyHostToDevice));
    free(h_data);

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_imageArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(
        cudaCreateTextureObject(&rgbaTexdImage, &texRes, &texDescr, NULL));

}

extern "C" void freeTexture() {

    checkCudaErrors(cudaFreeArray(d_imageArray));
}

// render image using CUDA
extern "C" void render(int width, int height,  dim3 blockSize, dim3 gridSize,
     uchar4 * output) {

    float tx = 240, ty = 120, scale = 2.5f, cx = 50, cy = 25;

        d_render << <gridSize, blockSize >> > (output, width, height, 240, 120, 2.5f,
            50, 25, rgbaTexdImage);


    getLastCudaError("kernel failed");
}

#endif