/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"


__global__
void atomicHisto(const unsigned int* const vals, //INPUT
                 unsigned int* const histo,      //OUPUT
                 const unsigned int numBins,
                 int numVals)
{
    const int myIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (myIndex < numVals)
    {
        unsigned int value = vals[myIndex];
        unsigned int myBin = value;
        atomicAdd(&(histo[myBin]), 1);
    }
}


__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
    //TODO Launch the yourHisto kernel

    //if you want to use/launch more than one kernel,
    //feel free

    const int numThreads = 1024;
    const int numBlocks = (numElems + numThreads - 1) / numThreads;
    const dim3 blockSize(numThreads);
    const dim3 gridSize(numBlocks);

    atomicHisto <<<gridSize, blockSize >>>(d_vals, d_histo, numBins, numElems);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 
}
