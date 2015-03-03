//Udacity HW 6
//Poisson Blending

#include <vector>
#include <algorithm>

using namespace std;

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

void reference_calc(const uchar4* const h_sourceImg,
    const size_t numRowsSource, const size_t numColsSource,
    const uchar4* const h_destImg,
    uchar4* const h_blendedImg);

size_t calcIndex(size_t x, size_t y, size_t /*numRows*/, size_t numCols)
{
    const unsigned int index = y * numCols + x;

    return index;
}


bool isImageEdge(size_t x, size_t y, size_t numRows, size_t numCols)
{
    bool isEdge = (x == 0) || (y == 0) || (x == (numCols - 1)) || (y == (numRows - 1));

    return isEdge;
}


void newChannelValue
(
    vector<float> const& sourceChannel,
    vector<float> const& destChannel,
    vector<float> const& sourceSums,
    vector<float> const& blendedChannel,
    vector<float>& newBlendedChannel,
    const size_t numRowsSource,
    const size_t numColsSource,
    vector<bool> const& mask,
    vector<bool> const& interior,
    vector<bool> const& border
)
{
    for (unsigned int row = 0; row < numRowsSource; row++)
    {
        for (unsigned int col = 0; col < numColsSource; col++)
        {
            if (!isImageEdge(col, row, numRowsSource, numColsSource))
            {
                const unsigned int index = calcIndex(col, row, numRowsSource, numColsSource);
                const unsigned int index1 = calcIndex(col + 1, row, numRowsSource, numColsSource);
                const unsigned int index2 = calcIndex(col - 1, row, numRowsSource, numColsSource);
                const unsigned int index3 = calcIndex(col, row + 1, numRowsSource, numColsSource);
                const unsigned int index4 = calcIndex(col, row - 1, numRowsSource, numColsSource);

                float blendedSum = 0.0f;
                float borderSum = 0.0f;

                if (interior[index1])
                    blendedSum += blendedChannel[index1];
                else
                    borderSum += destChannel[index1];

                if (interior[index2])
                    blendedSum += blendedChannel[index2];
                else
                    borderSum += destChannel[index2];

                if (interior[index3])
                    blendedSum += blendedChannel[index3];
                else
                    borderSum += destChannel[index3];

                if (interior[index4])
                    blendedSum += blendedChannel[index4];
                else
                    borderSum += destChannel[index4];
                
                //float sourceSum = 0.0f;
                //sourceSum += sourceChannel[index] - sourceChannel[index1];
                //sourceSum += sourceChannel[index] - sourceChannel[index2];
                //sourceSum += sourceChannel[index] - sourceChannel[index3];
                //sourceSum += sourceChannel[index] - sourceChannel[index4];

                float newVal = (blendedSum + borderSum + sourceSums[index]) / 4.0f;
                newVal = std::min(255.0f, std::max(0.0f, newVal));

                newBlendedChannel[index] = newVal;
            }
        }
    }
}


void calcSourceSums
(
    vector<float> const& sourceChannel, 
    vector<float>& sourceSums,
    const size_t numRowsSource,
    const size_t numColsSource
)
{
    for (unsigned int row = 0; row < numRowsSource; row++)
    {
        for (unsigned int col = 0; col < numColsSource; col++)
        {
            if (!isImageEdge(col, row, numRowsSource, numColsSource))
            {
                const unsigned int index = calcIndex(col, row, numRowsSource, numColsSource);
                const unsigned int index1 = calcIndex(col + 1, row, numRowsSource, numColsSource);
                const unsigned int index2 = calcIndex(col - 1, row, numRowsSource, numColsSource);
                const unsigned int index3 = calcIndex(col, row + 1, numRowsSource, numColsSource);
                const unsigned int index4 = calcIndex(col, row - 1, numRowsSource, numColsSource);

                float sourceSum = 0.0f;
                sourceSum += sourceChannel[index] - sourceChannel[index1];
                sourceSum += sourceChannel[index] - sourceChannel[index2];
                sourceSum += sourceChannel[index] - sourceChannel[index3];
                sourceSum += sourceChannel[index] - sourceChannel[index4];

                sourceSums[index] = sourceSum;
            }
        }
    }
}


void your_blend_cpu(const uchar4* const h_sourceImg,  //IN
    const size_t numRowsSource, const size_t numColsSource,
    const uchar4* const h_destImg, //IN
    uchar4* const h_blendedImg) //OUT
{
    const size_t numPixels = numRowsSource * numColsSource;

    // 1. Create the mask

    vector<bool> mask; // true for 'non-white' pixels
    mask.resize(numPixels);

    for (unsigned int row = 0; row < numRowsSource; row++)
    {
        for (unsigned int col = 0; col < numColsSource; col++)
        {
            const unsigned int index = calcIndex(col, row, numRowsSource, numColsSource);
            const uchar4 pixel = h_sourceImg[index];

            const unsigned char red   = pixel.x;
            const unsigned char green = pixel.y;
            const unsigned char blue = pixel.z;

            const bool white = (red == 255) && (green == 255) && (blue == 255);
            mask[index] = !white;
        }
    }

    // 2. Compute interior and border regions of the mask.

    vector<bool> interior;
    interior.resize(numPixels);
    vector<bool> border;
    border.resize(numPixels);

    for (unsigned int row = 0; row < numRowsSource; row++)
    {
        for (unsigned int col = 0; col < numColsSource; col++)
        {
            const unsigned int index = calcIndex(col, row, numRowsSource, numColsSource);

            bool isInterior = false;
            bool isBorder = false;

            if (!isImageEdge(col, row, numRowsSource, numColsSource))
            {
                isInterior = mask[calcIndex(col, row, numRowsSource, numColsSource)];
                isInterior = isInterior && mask[calcIndex(col - 1, row, numRowsSource, numColsSource)];
                isInterior = isInterior && mask[calcIndex(col + 1, row    , numRowsSource, numColsSource)];
                isInterior = isInterior && mask[calcIndex(col,     row - 1, numRowsSource, numColsSource)];
                isInterior = isInterior && mask[calcIndex(col,     row + 1, numRowsSource, numColsSource)];
            }

            interior[index] = isInterior;

            if (!isInterior)
                isBorder = mask[index];

            border[index] = isBorder;
        }
    }

    // 3 & 4: Split into colour channels.

    vector<float> sourceRed;
    sourceRed.resize(numPixels);
    vector<float> sourceGreen;
    sourceGreen.resize(numPixels);
    vector<float> sourceBlue;
    sourceBlue.resize(numPixels);

    vector<float> destRed;
    destRed.resize(numPixels);
    vector<float> destGreen;
    destGreen.resize(numPixels);
    vector<float> destBlue;
    destBlue.resize(numPixels);

    vector<float> blendedRed;
    blendedRed.resize(numPixels);
    vector<float> blendedGreen;
    blendedGreen.resize(numPixels);
    vector<float> blendedBlue;
    blendedBlue.resize(numPixels);

    for (unsigned int row = 0; row < numRowsSource; row++)
    {
        for (unsigned int col = 0; col < numColsSource; col++)
        {
            const unsigned int index = calcIndex(col, row, numRowsSource, numColsSource);
            const uchar4 pixel = h_sourceImg[index];

            const unsigned char red = pixel.x;
            const unsigned char green = pixel.y;
            const unsigned char blue = pixel.z;

            blendedRed[index] = red;
            blendedGreen[index] = green;
            blendedBlue[index] = blue;
            sourceRed[index] = red;
            sourceGreen[index] = green;
            sourceBlue[index] = blue;

            destRed[index] = h_destImg[index].x;
            destGreen[index] = h_destImg[index].y;
            destBlue[index] = h_destImg[index].z;
        }
    }

    vector<float> blendedRed2(blendedRed);
    vector<float> blendedGreen2(blendedGreen);
    vector<float> blendedBlue2(blendedBlue);

    // 5. Jacobi calculations

    vector<float> sourceSumsRed;
    sourceSumsRed.resize(numPixels);
    vector<float> sourceSumsGreen;
    sourceSumsGreen.resize(numPixels);
    vector<float> sourceSumsBlue;
    sourceSumsBlue.resize(numPixels);

    calcSourceSums(sourceRed,   sourceSumsRed,   numRowsSource, numColsSource);
    calcSourceSums(sourceGreen, sourceSumsGreen, numRowsSource, numColsSource);
    calcSourceSums(sourceBlue,  sourceSumsBlue,  numRowsSource, numColsSource);
    
    const int numIterations = 800;

    // Red channel
    for (unsigned int iteration = 0; iteration < numIterations; iteration++)
    {
        newChannelValue(sourceRed, destRed, sourceSumsRed, blendedRed, blendedRed2, numRowsSource, numColsSource, mask, interior, border);
        blendedRed = blendedRed2;
    }

    // Green channel
    for (unsigned int iteration = 0; iteration < numIterations; iteration++)
    {
        newChannelValue(sourceGreen, destGreen, sourceSumsGreen, blendedGreen, blendedGreen2, numRowsSource, numColsSource, mask, interior, border);
        blendedGreen = blendedGreen2;
    }

    // Blue channel
    for (unsigned int iteration = 0; iteration < numIterations; iteration++)
    {
        newChannelValue(sourceBlue, destBlue, sourceSumsBlue, blendedBlue, blendedBlue2, numRowsSource, numColsSource, mask, interior, border);
        blendedBlue = blendedBlue2;
    }

    // 6. Create the output image

    memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * numPixels);

    for (unsigned int row = 0; row < numRowsSource; row++)
    {
        for (unsigned int col = 0; col < numColsSource; col++)
        {
            const unsigned int index = calcIndex(col, row, numRowsSource, numColsSource);

            if (interior[index])
            {
                h_blendedImg[index].x = blendedRed[index];
                h_blendedImg[index].y = blendedGreen[index];
                h_blendedImg[index].z = blendedBlue[index];
            }
        }
    }

}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

    your_blend_cpu(h_sourceImg, numRowsSource, numColsSource, h_destImg, h_blendedImg);
    //reference_calc(h_sourceImg, numRowsSource, numColsSource, h_destImg, h_blendedImg);
}
