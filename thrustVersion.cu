#include <cuda.h>
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <iostream>
#include "cudaUtils.cu"


template <typename T>
class KernelVector {
	T* _array;
	int size;
	KernelVector(thrust::device_vector<T> &deviceVector) {
		_array = thrust::raw_pointer_cast( &deviceVector[0] );
		size = deviceVector.size();
	}
};

/*__device__ void printErr(const char *msg) {
	 cuPrintf("[%d,%d,%d].(%d,%d,%d):  %s\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, msg);
}
*/
__global__ void fooKernel() {
	char *msg = "hello";
    printErr(msg);
    return;
}
__global__ void barKernel() {
	printErr("good bye");
    return;
}

__global__ void bazKernel() {

}

int main(void) {
	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;
	std::cout << "Thrust v" << major << "." << minor << std::endl;


	cudaPrintfInit(); // call once

	// generate random data on the host
	thrust::host_vector<int> h_vec(100);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// transfer to device and compute sum
	thrust::device_vector<int> d_vec = h_vec;
	int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

	dim3 numBlocks(2,2,1);
	dim3 blockSize(3, 2);

	fooKernel<<<numBlocks,blockSize>>>();
	cudaPrintfDisplay(stdout, true);

	barKernel<<<numBlocks,blockSize>>>();
	cudaPrintfDisplay(stdout, true);

	cudaPrintfEnd(); // call once

	/*
	thrust::device_vector< Foo > fooVector;
	// Do something thrust-y with fooVector

	// Pass raw array and its size to kernel
	someKernelCall<<< x, y >>>( thrust::raw_pointer_cast(&fooVector[0]), fooVector.size() );
	*/

  return 0;
}