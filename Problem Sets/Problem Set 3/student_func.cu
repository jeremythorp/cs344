/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <cuda_runtime.h>
#include "utils.h"

__global__
void reduce_max(float* d_out, const float* d_in)
{
    extern __shared__ float shared_data[];
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadId  = threadIdx.x;

    shared_data[threadId] = d_in[myIndex];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>=1)
    {
        if (threadId < stride)
        {
            if (shared_data[threadId] < shared_data[threadId + stride])
            {
                shared_data[threadId] = shared_data[threadId + stride];
            }
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        d_out[blockIdx.x] = shared_data[0]; // copy result
    }
}


__global__
void reduce_min(float* d_out, const float* d_in)
{
    extern __shared__ float shared_data[];
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadId  = threadIdx.x;

    shared_data[threadId] = d_in[myIndex];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>=1)
    {
        if (threadId < stride)
        {
            if (shared_data[threadId] > shared_data[threadId + stride])
            {
                shared_data[threadId] = shared_data[threadId + stride];
            }
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        d_out[blockIdx.x] = shared_data[0]; // copy result
    }
}


__global__
void histogram(unsigned int* d_bins, const float* d_in, float lumMin, float lumRange, const int numBins)
{
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;
    float value = d_in[myIndex];
    int myBin = (value - lumMin) / lumRange * numBins;
    atomicAdd(&(d_bins[myBin]), 1);
}

__global__
void cumulativeDistribution(unsigned int* const d_cdf, const unsigned int* d_bins)
{
    extern __shared__ unsigned int shared_data2[];
    const int myIndex = threadIdx.x;

    shared_data2[myIndex] = d_bins[myIndex];
    __syncthreads();

    //
    // Use a Blelloch scan algorithm
    //

    // Reduce step

    const unsigned numBins = blockDim.x;
    const unsigned int maxIndex = numBins - 1;
    unsigned int offset = 1;

    for (unsigned int operations = numBins / 2; operations > 0; operations >>=1)
    {
        if (myIndex < operations)
        {
            unsigned int firstIndex  = offset * (2 * myIndex + 1) - 1;
            unsigned int secondIndex = offset * (2 * myIndex + 2) - 1;
            shared_data2[secondIndex] += shared_data2[firstIndex];
        }

        offset *= 2;
        __syncthreads();
    }


    //shared_data2[8] = offset;

    // Downsweep step

    if (myIndex == 0)
    {
        shared_data2[maxIndex] = 0;
        printf("blockDim.x = %d\n", blockDim.x);
    }
    
    __syncthreads();

    for (unsigned int operations = 1; operations <= (numBins / 2); operations <<=1)
    {
        offset >>= 1;

        if (myIndex < operations)
        {
            unsigned int firstIndex  = offset * (2 * myIndex + 1) - 1;
            unsigned int secondIndex = offset * (2 * myIndex + 2) - 1;

            unsigned int temp = shared_data2[firstIndex];
            shared_data2[firstIndex]  = shared_data2[secondIndex];
            shared_data2[secondIndex] += temp;
        }

        __syncthreads();
    }

    d_cdf[myIndex] = shared_data2[myIndex];
}



void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */


    // Step 1: find the min and max values.
    const size_t numPixels = numRows * numCols;
    const int numThreads = 1024;

    const dim3 blockSize(numThreads);
    const dim3 gridSize((numPixels + numThreads + 1)/numThreads);

    float* d_min_logLum = NULL;
    cudaMalloc((void**) &d_min_logLum, sizeof(float));
    float* d_max_logLum = NULL;
    cudaMalloc((void**) &d_max_logLum, sizeof(float));
    //cudaMemcpy(d_max_logLum, &max_logLum, sizeof(float), cudaMemcpyHostToDevice);

    float* d_intermediate = NULL;
    cudaMalloc((void**) &d_intermediate, numThreads * sizeof(float));
    
    //
    // Calculate the Minimum
    //

    reduce_min<<<gridSize, blockSize, numThreads * sizeof(float)>>>(d_intermediate, d_logLuminance);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Final reduce into a single results
    const dim3 final_blocks(1);
    const dim3 final_threads(gridSize);
    reduce_max<<<final_blocks, final_threads, numThreads * sizeof(float)>>>(d_min_logLum, d_intermediate);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Copy results back to host
    cudaMemcpy(&min_logLum, d_min_logLum, sizeof(float), cudaMemcpyDeviceToHost);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //
    // Calculate the Maximum
    //
    
    reduce_max<<<gridSize, blockSize, numThreads * sizeof(float)>>>(d_intermediate, d_logLuminance);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Final reduce into a single results
    reduce_max<<<final_blocks, final_threads, numThreads * sizeof(float)>>>(d_max_logLum, d_intermediate);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Copy results back to host
    cudaMemcpy(&max_logLum, d_max_logLum, sizeof(float), cudaMemcpyDeviceToHost);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //
    // Calculate the luminance range
    //

    const float intensityRange = max_logLum - min_logLum;

    // 
    // Calculate the histogram
    //

    unsigned int binSizeBytes = numBins * sizeof(int);
    unsigned int* d_bins = NULL;
    cudaMalloc((void**) &d_bins, binSizeBytes);
    cudaMemset(d_bins, 0, binSizeBytes);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    histogram<<<gridSize, blockSize>>>(d_bins, d_logLuminance, min_logLum, intensityRange, numBins);
    
    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // take a look at the histogram
    unsigned int h_histo[1024];
 
    cudaMemcpy(&h_histo, d_bins, binSizeBytes, cudaMemcpyDeviceToHost);

    //
    // Calculate the cumulative distribution
    //

    cudaMemset(d_cdf, 0, binSizeBytes);
    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    cumulativeDistribution<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_cdf, d_bins); 
    
    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // take a look at the cfd
    unsigned int h_cdf[1024];
 
    cudaMemcpy(&h_cdf, d_cdf, binSizeBytes, cudaMemcpyDeviceToHost);


    // Free up the CUDA device memory
    cudaFree(d_bins);
    d_bins = NULL;
    cudaFree(d_intermediate);
    d_intermediate = NULL;
    cudaFree(d_max_logLum);
    d_max_logLum = NULL;
}
