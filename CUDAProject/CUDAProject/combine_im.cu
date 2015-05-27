#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "histogram_common.h"

__global__ void
combine_im_kernel(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	/*
	combines images for a joint histogram computation with the formula:
	comb_im = B1*(im1 + im2*(B2-1))/(B1*B2 - 1)

	for a joint histogram of 256: B1*B2 must equal 256
	--> choose B1=B2=16
	*/

	float B1 = 16.0;
	float B2 = 16.0;

	if (i < numElements)
	{
		C[i] = B1*(A[i] + B[i] * (B2 - 1)) / (B1*B2 - 1);
	}
}

extern "C" void combine(const float *d_A, const float *d_B, float *d_C, int numElements)
{
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 512;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	combine_im_kernel << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);
	//err = cudaGetLastError();
}

