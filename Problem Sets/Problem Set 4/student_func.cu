//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */



__global__
void nibbleHistogram(unsigned int* d_bins, unsigned int* const d_inputVals, const unsigned int nibbleShift)
{
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int value = d_inputVals[myIndex];
    unsigned int myBin = (value >>  nibbleShift) & 0xf;
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



void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    //TODO
    //PUT YOUR SORT HERE

    //
    // Setup
    //

    const int numThreads = 1024;
    const dim3 blockSize(numThreads);
    const dim3 gridSize(numElems / numThreads);

    const unsigned int nibbleSize = 4;
    const unsigned int numBins = 2 << nibbleSize;

    unsigned int binSizeBytes = numBins * sizeof(int);
    unsigned int* d_bins = NULL;
    cudaMalloc((void**) &d_bins, binSizeBytes);
    cudaMemset(d_bins, 0, binSizeBytes);

    unsigned int cdfSizeBytes = numBins * sizeof(int);
    unsigned int* d_cdf = NULL;
    cudaMalloc((void**) &d_cdf, cdfSizeBytes);
    cudaMemset(d_bins, 0, cdfSizeBytes);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    const unsigned int numPasses = 8 * sizeof(unsigned int) / 4;

    unsigned int nibbleShift = 0;

    for (int pass = 0; pass < numPasses; pass++)
    {
        //
        // (1) Calculate the histogram
        //

        nibbleHistogram<<<gridSize, blockSize>>>(d_bins, d_inputVals, nibbleShift);
        
        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // take a look at the histogram
        unsigned int h_histo[numBins];
 
        cudaMemcpy(&h_histo, d_bins, binSizeBytes, cudaMemcpyDeviceToHost);

        //
        // (2) Exclusive Prefix Sum of Histogram
        //

        cudaMemset(d_cdf, 0, cdfSizeBytes);
        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        cumulativeDistribution<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_cdf, d_bins); 

        // take a look at the cfd
        unsigned int h_cdf[numBins];
 
        cudaMemcpy(&h_cdf, d_cdf, cdfSizeBytes, cudaMemcpyDeviceToHost);

        //
        // (3) Determine relative offset of each digit
        //
    
        //
        // (4) Combine the results of steps 2 & 3 to determine the final  output 
        //     location for each element and move it there)
        //

        nibbleShift += nibbleSize;
    }

    // Free up the CUDA device memory
    cudaFree(d_bins);
    d_bins = NULL;
    cudaFree(d_cdf);
    d_cdf = NULL;
}
