/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

// CUDA Runtime
#include <cuda_runtime.h>
#include <math.h>       /* for log function */

// Utility and system includes
#include "helper_cuda.h"
#include "helper_functions.h"  // helper for shared that are common to CUDA Samples

// project include
#include "histogram_common.h"

//Autre
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cmath>
#include "opencv\cv.h"




#include "opencv2/highgui/highgui.hpp"
using namespace cv;


const int numRuns = 16;
const static char *sSDKsample = "[histogram]\0";
void combine_images(float *im1, float *im2, float *&comb_im, uint im_size);
void mutual_information_CPU(float *histogram256GPUImage1, float* histogram256GPUImage2, float *histogram256ImageJoint, int row, int col, float &MI, int numElem, float *data1, float *data2);


int main(int argc, char **argv)
{

	std::filebuf fb;
	fb.open("test2.txt", std::ios::out);
	std::ostream ost(&fb);

	float  *h_HistogramCPU, *h_HistogramGPU;

	float *histogramGPUImage1, *histogramGPUImage2;
	float *histogram256GPUImage1, *histogram256GPUImage2;
	uchar *d_Data;
	float  *d_Histogram;
	StopWatchInterface *hTimer = NULL;
	StopWatchInterface *hCPUTimer = NULL;
	int PassFailFlag = 1;
	
	uint uiSizeMult = 1;
	Mat image = imread("Image_Test.jpg",CV_BGR2GRAY);
	//Mat image2 = imread("Image_Sobel8.png", CV_BGR2GRAY);
	Mat image2 = imread("Image_Test.jpg", CV_BGR2GRAY);

	int col = image.cols;
	int lin = image.rows;
	int col2 = image2.cols;
	int lin2 = image2.rows;

	int byteCount = col*lin*4;
	int byteCount2 = col2*lin2 * 4;

	float *data;
	float *data2;
	data = (float *)malloc(byteCount);
	data2 = (float *)malloc(byteCount2);

	
	int nombre = 0;
	int index;
	for (int i = 0; i <lin; i++)
	{
		index = 0;
		for (int j = 0; j <col*3; j+=3)
		{
			Scalar passage = image.at<uchar>(i, j);
			data[index + i*col] = passage.val[0];
			index++;
			
		}
		nombre += index;
	}

	int nombre2 = 0;
	int index2;
	for (int i = 0; i <lin2; i++)
	{
		index2 = 0;
		for (int j = 0; j <col2 * 3; j += 3)
		{
			Scalar passage = image2.at<uchar>(i, j);
			data2[index2 + i*col2] = passage.val[0];
			index2++;

		}
		nombre2 += index2;
	}

	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	// set logfile name and start logs
	printf("[%s] - Starting...\n", sSDKsample);

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	sdkCreateTimer(&hTimer);
	sdkCreateTimer(&hCPUTimer);

	// Optional Command-line multiplier to increase size of array to histogram
	if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
	{
		uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
		uiSizeMult = MAX(1, MIN(uiSizeMult, 10));
		byteCount *= uiSizeMult;
	}

	printf("Initializing data...\n");
	printf("...allocating CPU memory.\n");
	h_HistogramCPU = (float *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	h_HistogramGPU = (float*)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	histogramGPUImage2 = (float *)malloc(64*4);
	histogramGPUImage1 = (float *)malloc(64*4);
	histogram256GPUImage2 = (float *)malloc(256 * 4);
	histogram256GPUImage1 = (float *)malloc(256 * 4);
	
	// Starting histogram 64 bins
	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_Data, data, byteCount, cudaMemcpyHostToDevice));
	{
		printf("Starting up 64-bin histogram...\n\n");
		initHistogram64();
		printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
		for (int iter = -1; iter < numRuns; iter++)
		{
			//iter == -1 -- warmup iteration
			if (iter == 0)
			{
				cudaDeviceSynchronize();
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);
			}
			histogram64(d_Histogram, d_Data, byteCount);
		}

		cudaDeviceSynchronize();
		sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
		printf("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
		printf("histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
			(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM64_THREADBLOCK_SIZE);

		printf("\nValidating GPU results...\n");
		printf(" ...reading back GPU results\n");
		checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

		printf(" ...histogram64CPU()\n");
		sdkResetTimer(&hCPUTimer);
		sdkStartTimer(&hCPUTimer);
		histogram64CPU(h_HistogramCPU,data,byteCount);
		sdkStopTimer(&hCPUTimer);

		double dAvgSecs1 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer);
		printf("histogram64CPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs1, ((double)byteCount * 1.0e-6) / dAvgSecs1);
		printf(" ...comparing the results...\n");

		for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
		{
			histogramGPUImage1[i] = h_HistogramGPU[i];
			if (histogramGPUImage1[i] != h_HistogramCPU[i])
			{
				PassFailFlag = 0;
			}
			histogramGPUImage1[i]  /= (byteCount );
			//ost << histogramGPUImage1[i] << " " << h_HistogramCPU[i] << " " << std::endl;
		}
		printf(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n");

		std::cout << PassFailFlag << " " << "le Flag error de cet histo" << std::endl;
		printf("Shutting down 64-bin histogram...\n\n\n");
		closeHistogram64();
	}
	// fin de histrogram 64 bins 1 ere Image

	//Debut histogram 256 bins
	{
		printf("Initializing 256-bin histogram...\n");
		initHistogram256();
		printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

		for (int iter = -1; iter < numRuns; iter++)
		{
			//iter == -1 -- warmup iteration
			if (iter == 0)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);
			}

			histogram256(d_Histogram, d_Data, byteCount);
		}

		cudaDeviceSynchronize();
		sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
		printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
		printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
			(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);

		printf("\nValidating GPU results...\n");
		printf(" ...reading back GPU results\n");
		checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

		printf(" ...histogram256CPU()\n");
		sdkResetTimer(&hCPUTimer);
		sdkStartTimer(&hCPUTimer);
		histogram256CPU(h_HistogramCPU, data, byteCount);
		sdkStopTimer(&hCPUTimer);
		double dAvgSecs2 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer);
		printf("histogram256CPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs2, ((double)byteCount * 1.0e-6) / dAvgSecs2);
		printf(" ...comparing the results\n");

		for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
		{
			histogram256GPUImage1[i] = h_HistogramGPU[i];
			if (histogram256GPUImage1[i] != h_HistogramCPU[i])
			{
				PassFailFlag = 0;
			}
			histogram256GPUImage1[i] /= (byteCount);

			//ost << histogram256GPUImage1[i] << " " << h_HistogramCPU[i] << " " << std::endl;
		}
		std::cout << PassFailFlag << " " << "le Flag error de cet histo" << std::endl;
		printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");

		printf("Shutting down 256-bin histogram...\n\n\n");
		closeHistogram256();
	}
	// Fin du premier histogram 256 bins

	//Debut histrogram 64 bins 2 ieme Image
	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount2));
	checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_Data, data2, byteCount2, cudaMemcpyHostToDevice));
	{
		printf("Starting up 64-bin histogram...\n\n");
		initHistogram64();
		printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount2, numRuns);
		for (int iter = -1; iter < numRuns; iter++)
		{
			//iter == -1 -- warmup iteration
			if (iter == 0)
			{
				cudaDeviceSynchronize();
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);
			}

			histogram64(d_Histogram, d_Data, byteCount2);
		}
		cudaDeviceSynchronize();
		sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
		printf("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount2 * 1.0e-6) / dAvgSecs);
		printf("histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
			(1.0e-6 * (double)byteCount2 / dAvgSecs), dAvgSecs, byteCount2, 1, HISTOGRAM64_THREADBLOCK_SIZE);

		printf("\nValidating GPU results...\n");
		printf(" ...reading back GPU results\n");
		checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
		printf(" ...histogram64CPU()\n");
		sdkResetTimer(&hCPUTimer);
		sdkStartTimer(&hCPUTimer);
		histogram64CPU(h_HistogramCPU, data2, byteCount2);
		sdkStopTimer(&hCPUTimer);

		double dAvgSecs1 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer);
		printf("histogram64CPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs1, ((double)byteCount * 1.0e-6) / dAvgSecs1);
		printf(" ...comparing the results...\n");
		for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
		{
			histogramGPUImage2[i] = h_HistogramGPU[i];
			if (histogramGPUImage2[i] != h_HistogramCPU[i])
			{
				PassFailFlag = 0;
			}
			histogramGPUImage2[i] /= (byteCount );

			//ost << histogramGPUImage2[i] << " " << h_HistogramCPU[i] << " " << std::endl;
		}
		printf(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n");

		printf("Shutting down 64-bin histogram...\n\n\n");
		closeHistogram64();

		std::cout << PassFailFlag << " " << "le Flag error de cet histo" << std::endl;
	}
	// Fin histogram 64 bins 2 ieme image

	// Debut du deuxieme histogram 256 bins
	{
		printf("Initializing 256-bin histogram...\n");
		initHistogram256();

		printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount2, numRuns);

		for (int iter = -1; iter < numRuns; iter++)
		{
			//iter == -1 -- warmup iteration
			if (iter == 0)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);
			}
			histogram256(d_Histogram, d_Data, byteCount2);
		}

		cudaDeviceSynchronize();
		sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
		printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
		printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
			(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);

		printf("\nValidating GPU results...\n");
		printf(" ...reading back GPU results\n");
		checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

		printf(" ...histogram256CPU()\n");
		sdkResetTimer(&hCPUTimer);
		sdkStartTimer(&hCPUTimer);
		histogram256CPU(h_HistogramCPU, data2, byteCount2);
		sdkStopTimer(&hCPUTimer);
		double dAvgSecs2 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer);
		printf("histogram256CPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs2, ((double)byteCount * 1.0e-6) / dAvgSecs2);
		printf(" ...comparing the results\n");

		for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
		{
			histogram256GPUImage2[i] = h_HistogramGPU[i];
			if (histogram256GPUImage2[i] != h_HistogramCPU[i])
			{
				PassFailFlag = 0;
			}
			histogram256GPUImage1[i] /= (byteCount );

			//ost << histogram256GPUImage2[i] << " " << h_HistogramCPU[i] << " " << std::endl;
		}
		printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");

		std::cout << PassFailFlag << " " << "le Flag error de cet histo" << std::endl;
		printf("Shutting down 256-bin histogram...\n\n\n");
		closeHistogram256();
	}
	// fin deuxieme histogram 256 bins
	






	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = byteCount/4;
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	// Allocate the host input vector A
	float *h_A = (float *)malloc(size);

	// Allocate the host input vector B
	float *h_B = (float *)malloc(size);

	// Allocate the host output vector C
	float *h_C = (float *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = data[i];
		h_B[i] = data2[i];
	}

	// Allocate the device input vector A
	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float *d_C = NULL;
	err = cudaMalloc((void **)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	sdkResetTimer(&hCPUTimer);
	sdkStartTimer(&hCPUTimer);

	
	sdkStopTimer(&hCPUTimer);
	for (int iter = -1; iter < numRuns; iter++)
	{
		//iter == -1 -- warmup iteration
		if (iter == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}
		combine(d_A, d_B, d_C, numElements);
	}

	cudaDeviceSynchronize();
	sdkStopTimer(&hTimer);

	double dAvgSecs2 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer) / (double) numRuns;
	printf("GPU Image Join time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs2, ((double)byteCount * 1.0e-6) / dAvgSecs2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float B1 = 16.0;
	float B2 = 16.0;
	// B1*(A[i] + B[i] * (B2 - 1)) / (B1*B2 - 1);

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(B1*(h_A[i] + h_B[i] * (B2 - 1)) / (B1*B2 - 1) - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			system("pause");
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");
	printf("Done\n");

	float *histogram256ImageJoint;
	histogram256ImageJoint = (float *)malloc(256 * 4);
	float *histogram64ImageJoint;
	histogram64ImageJoint = (float*)malloc(64 * 4);
	// Starting histogram 64 bins
	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_Data, h_C, byteCount, cudaMemcpyHostToDevice));
	{
		printf("Starting up 64-bin histogram...\n\n");
		initHistogram64();
		printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
		for (int iter = -1; iter < numRuns; iter++)
		{
			//iter == -1 -- warmup iteration
			if (iter == 0)
			{
				cudaDeviceSynchronize();
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);
			}
			histogram64(d_Histogram, d_Data, byteCount);
		}

		cudaDeviceSynchronize();
		sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
		printf("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
		printf("histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
			(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM64_THREADBLOCK_SIZE);

		printf("\nValidating GPU results...\n");
		printf(" ...reading back GPU results\n");
		checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

		printf(" ...histogram64CPU()\n");
		sdkResetTimer(&hCPUTimer);
		sdkStartTimer(&hCPUTimer);
		histogram64CPU(h_HistogramCPU, h_C, byteCount);
		sdkStopTimer(&hCPUTimer);

		double dAvgSecs1 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer);
		printf("histogram64CPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs1, ((double)byteCount * 1.0e-6) / dAvgSecs1);
		printf(" ...comparing the results...\n");

		for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
		{
			histogram64ImageJoint[i] = h_HistogramGPU[i];
			if (histogram64ImageJoint[i] != h_HistogramCPU[i])
			{
				PassFailFlag = 0;
			}
			histogram64ImageJoint[i] /= (byteCount );
			//ost << histogramGPUImage1[i] << " " << h_HistogramCPU[i] << " " << std::endl;
		}
		printf(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n");

		std::cout << PassFailFlag << " " << "le Flag error de cet histo" << std::endl;
		printf("Shutting down 64-bin histogram...\n\n\n");
		closeHistogram64();
	}
	// fin de histrogram 64 bins 1 ere Image

	//Debut histogram 256 bins
	{
		printf("Initializing 256-bin histogram...\n");
		initHistogram256();
		printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

		for (int iter = -1; iter < numRuns; iter++)
		{
			//iter == -1 -- warmup iteration
			if (iter == 0)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);
			}

			histogram256(d_Histogram, d_Data, byteCount);
		}

		cudaDeviceSynchronize();
		sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
		printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
		printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
			(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);

		printf("\nValidating GPU results...\n");
		printf(" ...reading back GPU results\n");
		checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

		printf(" ...histogram256CPU()\n");
		sdkResetTimer(&hCPUTimer);
		sdkStartTimer(&hCPUTimer);
		histogram256CPU(h_HistogramCPU, h_C, byteCount);
		sdkStopTimer(&hCPUTimer);
		double dAvgSecs2 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer);
		printf("histogram256CPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs2, ((double)byteCount * 1.0e-6) / dAvgSecs2);
		printf(" ...comparing the results\n");
		float somme = 0;
		for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
		{
			histogram256ImageJoint[i] = h_HistogramGPU[i];
			if (histogram256ImageJoint[i] != h_HistogramCPU[i])
			{
				PassFailFlag = 0;
			}
			histogram256ImageJoint[i] /= (byteCount );
			somme += histogram256ImageJoint[i];
			//ost << histogram256ImageJoint[i] << " " << h_HistogramCPU[i] << " " << std::endl;
			
		}
		//ost << somme << std::endl;
		std::cout << PassFailFlag << " " << "le Flag error de cet histo" << std::endl;
		printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");

		printf("Shutting down 256-bin histogram...\n\n\n");
		closeHistogram256();
	}
	// Fin du premier histogram 256 bins
	
	sdkResetTimer(&hCPUTimer);
	sdkStartTimer(&hCPUTimer);


	float *jointCPU;
	jointCPU = (float*)malloc(byteCount);
	combine_images(data, data2, jointCPU,byteCount/4);
	
	sdkStopTimer(&hCPUTimer);
	dAvgSecs2 = 1.0e-3 * (double)sdkGetTimerValue(&hCPUTimer);
	printf("CPU Image Join time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs2, ((double)byteCount * 1.0e-6) / dAvgSecs2);
	
	
	float MI= 0;
	
	mutual_information_CPU(histogram256GPUImage1, histogram256GPUImage2, histogram256ImageJoint, lin, col, MI, byteCount/4, data, data2);
	std::cout << "Information mutuelle" << MI << std::endl;


	free(h_A);
	free(h_B);
	free(jointCPU);
	printf("Shutting down...\n");



	sdkDeleteTimer(&hCPUTimer);
	sdkDeleteTimer(&hTimer);
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
	free(h_C);
	free(data);
	free(histogramGPUImage1);
	free(histogram256GPUImage1);
	free(histogramGPUImage2);
	free(histogram256GPUImage2);
	free(data2);
	free(
		histogram256ImageJoint);
	free(
		histogram64ImageJoint);

	checkCudaErrors(cudaFree(d_Histogram));
	checkCudaErrors(cudaFree(d_Data));
	free(h_HistogramGPU);
	free(h_HistogramCPU);
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
	std::cin.get();
	printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

	printf("%s - Test Summary\n", sSDKsample);

	// pass or fail (for both 64 bit and 256 bit histograms)
	if (!PassFailFlag)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

	printf("Test passed\n");
	fb.close();
	exit(EXIT_SUCCESS);
}



// tout est en float car sinon il faut faire des cast partout pour les +-*/ (sinon il utilise les +-*/ pour int)

void mutual_information_CPU(float *p1, float *p2, float *joint, int row, int col, float &MI, int numElem,float *data1,float *data2)
{
	for (int i = 0; i < 50; i++)
	{
		for (int j = 0; j < 50; j++)
		{
			if (joint[(int)data1[i]] >0.0000000000000000000 && (p1[(int)data1[i]] * p2[(int)data2[j]]) >0.00000000000000001 && joint[(int)data1[i]] <1 && (p1[(int)data1[i]] * p2[(int)data2[j]]) < 1) MI += joint[(int)data1[i]] * log((joint[(int)data1[i]]) / (p1[(int)data1[i]] * p2[(int)data2[j]]));
		}
	}
}

//il y a aussi une version GPU
//voir combine_im.cu
//j'ai juste addapté le vector add
void combine_images(float *im1, float *im2, float *&comb_im, uint im_size)
{
	/*
	combines images for a joint histogram computation with the formula:
	comb_im = B1*(im1 + im2*(B2-1))/(B1*B2 - 1)

	for a joint histogram of 256: B1*B2 must equal 256
	--> choose B1=B2=16
	*/

	uint B1=16, B2 = 16;

	for (int i = 0; i < im_size; i++)
	{
		comb_im[i] = B1*(im1[i] + im2[i] * (B2 - 1)) / (B1*B2 - 1);
	}

}

/*
Ca n'a pas trop de sens de faire le calcul sur de reduction de somme sur GPU car il faut d'abord creer sur CPU un array qui représente la somme a reduire...
voici neamoins le code pour avoir l'array...
reste seulement appeler le CUDA sample "reduction" et lui donner MI
*/

/*void mutual_information_GPU(float *p1, float *p2, float *joint, uint size, float *MI)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			MI[i*size + j] = joint[i] * log(joint[i]) / (p1[i] * p2[j]));
		}
	}

	//appel de la fonction reduction des cuda samples...
}*/

