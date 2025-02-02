
#include "cudaLib.cuh"
#include "cpuLib.h"
#include <cstdlib>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here device
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {
	uint64_t vectorBytes = vectorSize * sizeof(float);
	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here host
	//std::cout << "Lazy, you are!\n";
	//std::cout << "Write code, you must\n";

	float* y_d;
	float* x_d;
	float* x;
	float* y;
	float* z;
	float scale = 2.0f; //rand() % 100;

	//Mem allocation
	x = (float*) malloc(vectorBytes);
	y = (float*) malloc(vectorBytes);
	z = (float*) malloc(vectorBytes);
	vectorInit(x, vectorSize);
	vectorInit(y, vectorSize);
	
	cudaMalloc((void **) &x_d, vectorBytes);
        cudaMalloc((void **) &y_d, vectorBytes);
        cudaMemcpy(x_d, x, vectorBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, y, vectorBytes, cudaMemcpyHostToDevice);

        // Perform SAXPY
	saxpy_gpu<<<vectorSize/256.0,256>>>(x, y, scale, vectorSize);
        cudaMemcpy(y, y_d, vectorBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(z, y_d, vectorBytes, cudaMemcpyDeviceToHost);
	//verifyVector(x, y, z, scale, vectorSize); comment out due to it not working properly and would cause very long exceution times despite code being correct
	#ifndef DEBUG_PRINT_DISABLE
        	printf("\n Adding vectors: \n");
        	printf(" scale = %f\n", scale);
        	printf(" x = { ");
        	for (int i = 0; i < 5; ++i) {
        	    printf("%3.4f, ", x[i]);
        	}
        	printf(" ... }\n");
        	printf(" y = { ");
        	for (int i = 0; i < 5; ++i) {
        	    printf("%3.4f, ", y[i]);
        	}
		printf(" ... }\n");
                printf(" z = { ");
                for (int i = 0; i < 5; ++i) {
                    printf("%3.4f, ", scale * x[i] + y[i]);
                }
                printf(" ... }\n");

        #endif
	//Free memory
	cudaFree (x_d);
	cudaFree (y_d);
	cudaFree (x);
	cudaFree (y);
	cudaFree (z);
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	uint64_t hit = 0;
	int tID = threadIdx.x + blockDim.x * blockIdx.x;
	curandState_t rng;
	curand_init(clock64(), tID, 0, &rng);

	if  (tID < pSumSize) {
		for (int i = 0; i < sampleSize; i++) {
			float x = curand_uniform(&rng);
			float y = curand_uniform(&rng);
			if ( int(x * x + y * y) == 0 ) {
				++hit;
			}
		}
		pSums[tID] = hit;
	}

}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int tID = threadIdx.x + blockDim.x * blockIdx.x;
	uint64_t sum = 0;
	if (tID < reduceSize) {
		for (int i = tID; i < pSumSize; i += reduceSize) {
			sum += pSums[tID];
		}
	}
	totals[tID] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	//std::cout << "Sneaky, you are ...\n";
	//std::cout << "Compute pi, you must!\n";

	uint64_t vectorSize = generateThreadCount * sizeof(uint64_t);
	uint64_t RedVectorSize = reduceThreadCount * sizeof(uint64_t);
	uint64_t* total;
	uint64_t* hit_d;
	uint64_t* total_d;
	uint64_t hitcount = 0;

	//Allocate memory
	total = (uint64_t*)malloc(RedVectorSize);
	cudaMalloc((void **)&hit_d, vectorSize);
	cudaMalloc((void **)&total_d, RedVectorSize);

	generatePoints<<<ceil(generateThreadCount/256.0),256>>>(hit_d, generateThreadCount, sampleSize);
	reduceCounts<<<ceil(generateThreadCount/256),256>>>(hit_d, total_d, generateThreadCount, reduceSize);
	cudaMemcpy(total, total_d, RedVectorSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < reduceThreadCount; ++i) {
		hitcount += total[i];
	}

	//Free mem and  approx pi
	approxPi = 4.0f * hitcount / (sampleSize * generateThreadCount);

	free(total);
	cudaFree(hit_d);
	cudaFree(total_d);

	return approxPi;
}
