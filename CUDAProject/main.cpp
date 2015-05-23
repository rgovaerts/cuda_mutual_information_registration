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

int main(int argc, char **argv)
{

	//std::filebuf fb;
	//fb.open("test2.txt", std::ios::out);
	//std::ostream ost(&fb);

	uint  *h_HistogramCPU, *h_HistogramGPU;

	int *histogramGPUImage1, *histogramGPUImage2;
	int *histogram256GPUImage1, *histogram256GPUImage2;
	uchar *d_Data;
	uint  *d_Histogram;
	StopWatchInterface *hTimer = NULL;
	StopWatchInterface *hCPUTimer = NULL;
	int PassFailFlag = 1;
	
	uint uiSizeMult = 1;
	Mat image = imread("Image_Test.jpg",CV_BGR2GRAY);
	Mat image2 = imread("Image_Sobel8.png", CV_BGR2GRAY);

	int col = image.cols;
	int lin = image.rows;
	int col2 = image2.cols;
	int lin2 = image2.rows;

	int byteCount = col*lin*4;
	int byteCount2 = col2*lin2 * 4;

	uint *data;
	uint *data2;
	data = (uint *)malloc(byteCount);
	data2 = (uint *)malloc(byteCount2);

	
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
	h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	histogramGPUImage2 = (int *)malloc(64*4);
	histogramGPUImage1 = (int *)malloc(64*4);
	histogram256GPUImage2 = (int *)malloc(256 * 4);
	histogram256GPUImage1 = (int *)malloc(256 * 4);
	
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
			//ost << histogram256GPUImage2[i] << " " << h_HistogramCPU[i] << " " << std::endl;
		}
		printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");

		std::cout << PassFailFlag << " " << "le Flag error de cet histo" << std::endl;
		printf("Shutting down 256-bin histogram...\n\n\n");
		closeHistogram256();
	}
	// fin deuxieme histogram 256 bins
	
	
	printf("Shutting down...\n");
	sdkDeleteTimer(&hTimer);
	checkCudaErrors(cudaFree(d_Histogram));
	checkCudaErrors(cudaFree(d_Data));

	free(h_HistogramGPU);
	free(h_HistogramCPU);
	free(data);
	free(histogramGPUImage1);
	free(histogramGPUImage2);
	free(data2);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

	printf("%s - Test Summary\n", sSDKsample);

	// pass or fail (for both 64 bit and 256 bit histograms)
	if (!PassFailFlag)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

	printf("Test passed\n");
	//fb.close();

	std::cin.get();
	exit(EXIT_SUCCESS);
}
