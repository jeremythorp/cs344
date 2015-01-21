//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <vector>

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
void histogram
(
    unsigned int* d_bins, 
    unsigned int* const d_inputVals, 
    const unsigned int shift, 
    const unsigned int mask, 
    const unsigned int numElems
)
{
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;

    if (myIndex < numElems)
    {
        unsigned int value = d_inputVals[myIndex];
        unsigned int myBin = (value >> shift) & mask;
        atomicAdd(&(d_bins[myBin]), 1);
    }
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


class sort_element
{
public:
    sort_element(const unsigned int value, const unsigned int position);
    unsigned int GetValue() const    { return m_value;    } ;
    unsigned int GetPosition() const { return m_position; } ;
private:
    unsigned int m_value;
    unsigned int m_position;
};


sort_element::sort_element(const unsigned int value, const unsigned int position)
  : m_value(value),
    m_position(position)
{
}

bool operator<(sort_element const& lhs, sort_element const& rhs)
{
    return lhs.GetValue() < rhs.GetValue();
}


void your_sort_cpu(unsigned int* const d_inputVals,
                   unsigned int* const d_inputPos,
                   unsigned int* const d_outputVals,
                   unsigned int* const d_outputPos,
                   const size_t numElems)
{
    // This is a simple, CPU-based, implementation for reference purposes.

    unsigned int* h_vals = new unsigned int[numElems];
    unsigned int* h_pos  = new unsigned int[numElems];

    unsigned int numBytes = numElems * sizeof(unsigned int);
    memset(h_vals, 0, numBytes);
    memset(h_pos, 0, numBytes);

    cudaMemcpy(h_vals, d_inputVals, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos,  d_inputPos,  numBytes, cudaMemcpyDeviceToHost);

    std::vector<sort_element> data;
    data.reserve(numElems);

    for (unsigned int i = 0; i < numElems; i++)
    {
        data.push_back(sort_element(h_vals[i], h_pos[i]));
    }

    // We need to use a stable sort. 
    // Otherwise output positions for the same input value may not match the reference code output positions giving a difference.
    stable_sort(data.begin(), data.end());

    for (unsigned int i = 0; i < numElems; i++)
    {
        h_vals[i] = data[i].GetValue();
        h_pos[i]  = data[i].GetPosition();
    }

    cudaMemcpy(d_outputVals, h_vals, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputPos,  h_pos,  numBytes, cudaMemcpyHostToDevice);

    delete[] h_vals;
    delete[] h_pos;
}


/*
__global__
void fillOffsetArray
(
    unsigned int* const d_inputVals, 
    unsigned int* d_offsetArray, 
    const size_t numElems,
    const unsigned int numBins,
    const unsigned int shift, 
    const unsigned int mask
)
{
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;

    if (myIndex < numElems)
    {
        unsigned int value = d_inputVals[myIndex];
        unsigned int myBin = (value >> shift) & mask;

        d_offsetArray[numElems * myBin + myIndex] = 1;
    }
}
*/


__global__
void calcOffsets
(
    unsigned int* const d_inputVals, 
    unsigned int* d_offsetArray, 
    unsigned int* d_binCount,
    const size_t numElems,
    const unsigned int numBins,
    const unsigned int shift, 
    const unsigned int mask
)
{
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;

    if (myIndex < numBins)
        d_binCount[myIndex] = 0;
        
    __syncthreads();

    if (myIndex == 0)
    {
        for (int i = 0; i < numElems; i++)
        {
            unsigned int value = d_inputVals[i];            
            int bin = (value >> shift) & mask;
            d_offsetArray[i] = d_binCount[bin];
            atomicAdd(&(d_binCount[bin]), 1);
        }
    }
}
/*
{
    extern __shared__ unsigned int binCount[];
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;

    if (myIndex < numBins)
        binCount[myIndex] = 0;
        
    __syncthreads();
    
    if (myIndex < numElems)
    {
        unsigned int value = d_inputVals[myIndex];            
        int bin = (value >> shift) & mask;
        d_offsetArray[myIndex] = binCount[bin];

        atomicAdd(&(binCount[bin]), 1);
    }
}
*/

__global__
void scatter
(
    unsigned int* const d_inputVals, 
    unsigned int* const d_inputPos,
    unsigned int* const d_outputVals,
    unsigned int* const d_outputPos,
    unsigned int* const d_cdf,
    unsigned int* d_offsetArray, 
    const size_t numElems,
    const unsigned int numBins,
    const unsigned int shift, 
    const unsigned int mask
)
{
    const int myIndex   = blockIdx.x * blockDim.x + threadIdx.x;

    if (myIndex < numElems)
    {
        unsigned int value = d_inputVals[myIndex];            
        int bin = (value >> shift) & mask;
        int destIndex = d_offsetArray[myIndex] + d_cdf[bin];

        d_outputVals[destIndex] = d_inputVals[myIndex];
        d_outputPos[destIndex]  = d_inputPos[myIndex];
    }
}


void your_sort_gpu(unsigned int* const d_inputVals,
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
    const int numBlocks  = (numElems + numThreads - 1) / numThreads;
    const dim3 blockSize(numThreads);
    const dim3 gridSize(numBlocks);

    const unsigned int shiftSize = 8;
    const unsigned int numBins = 1 << shiftSize;
    const unsigned int mask = (1 << shiftSize) - 1;

    unsigned int binSizeBytes = numBins * sizeof(unsigned int);
    unsigned int* d_bins = NULL;
    cudaMalloc((void**) &d_bins, binSizeBytes);
    cudaMemset(d_bins, 0, binSizeBytes);

    unsigned int* d_binCount = NULL;
    cudaMalloc((void**) &d_binCount, binSizeBytes);
    cudaMemset(d_binCount, 0, binSizeBytes);

    unsigned int cdfSizeBytes = numBins * sizeof(unsigned int);
    unsigned int* d_cdf = NULL;
    cudaMalloc((void**) &d_cdf, cdfSizeBytes);
    cudaMemset(d_bins, 0, cdfSizeBytes);

    unsigned int offsetArraySize = numElems;
    unsigned int offsetArraySizeBytes = offsetArraySize * sizeof(unsigned int);
    unsigned int* d_offsetArray = NULL;
    cudaMalloc((void**) &d_offsetArray, offsetArraySizeBytes);
    cudaMemset(d_offsetArray, 0, offsetArraySizeBytes);

    // Check all is ok
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    const unsigned int numPasses = 8 * sizeof(unsigned int) / shiftSize;

    unsigned int shift = 0;

    for (int pass = 0; pass < numPasses; pass++)
    {
        //
        // (1) Calculate the histogram
        //

        cudaMemset(d_bins, 0, binSizeBytes);

        histogram<<<gridSize, blockSize>>>(d_bins, d_inputVals, shift, mask, numElems);
        
        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // DEBUG: take a look at the histogram
        unsigned int h_histo[numBins];
        cudaMemcpy(&h_histo, d_bins, binSizeBytes, cudaMemcpyDeviceToHost);

        //
        // (2) Exclusive Prefix Sum of Histogram
        //

        cudaMemset(d_cdf, 0, cdfSizeBytes);

        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        cumulativeDistribution<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_cdf, d_bins); 

        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // DEBUG: take a look at the cfd
        unsigned int h_cdf[numBins];
        cudaMemcpy(&h_cdf, d_cdf, cdfSizeBytes, cudaMemcpyDeviceToHost);

        //
        // (3) Determine relative offset of each digit
        //

        cudaMemset(d_offsetArray, 0, offsetArraySizeBytes);
        //fillOffsetArray<<<gridSize, blockSize>>>(d_inputVals, d_offsetArray, numElems, numBins, shift, mask);

        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        
        calcOffsets<<<gridSize, blockSize>>>(d_inputVals, d_offsetArray, d_binCount, numElems, numBins, shift, mask);
         
        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
        //
        // (4) Combine the results of steps 2 & 3 to determine the final  output 
        //     location for each element and move it there)
        //

        scatter<<<gridSize, blockSize>>>(
            d_inputVals, 
            d_inputPos,
            d_outputVals,
            d_outputPos,
            d_cdf,
            d_offsetArray, 
            numElems,
            numBins,
            shift, 
            mask
        );

        // Check all is ok
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // copy ready for next loop.
        unsigned int intputSizeBytes = numElems * sizeof(unsigned int);
        cudaMemcpy(d_inputVals, d_outputVals, intputSizeBytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_inputPos,  d_outputPos,  intputSizeBytes, cudaMemcpyDeviceToDevice);

        shift += shiftSize;
    }

    // Free up the CUDA device memory
    cudaFree(d_bins);
    d_bins = NULL;
    cudaFree(d_cdf);
    d_cdf = NULL;
    cudaFree(d_offsetArray);
    d_offsetArray = NULL;
    cudaFree(d_binCount);
    d_binCount = NULL;
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    //your_sort_cpu(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems); // for debugging
    your_sort_gpu(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
}             
