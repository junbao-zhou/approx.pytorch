#include <cuda.h>
#include <stdio.h>
#include <iostream>

template <typename data_type, int N>
__global__ void show(data_type th_id_x[N][N], data_type th_id_y[N][N], data_type blc_id_x[N][N], data_type blc_id_y[N][N])
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	th_id_x[i][j] = threadIdx.x;
	th_id_y[i][j] = threadIdx.y;
	blc_id_x[i][j] = blockIdx.x;
	blc_id_y[i][j] = blockIdx.y;
}

template <typename data_type, int N>
void print_list_list(data_type list[N][N])
{
	for (int x = 0; x < N; ++x)
	{
		for (int y = 0; y < N; ++y)
		{
			std::cout << list[x][y] << "    ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void show_cuda_info()
{
	int dev = 0;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);
	std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
	std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}

int main(void)
{
	show_cuda_info();
	constexpr int N = 16;
	typedef uint16_t type_t;
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

#define MALLOC_2D(name, n) \
	type_t *name;          \
	cudaMallocManaged(&name, n *n * sizeof(type_t));

	MALLOC_2D(thread_x, N)
	MALLOC_2D(thread_y, N)
	MALLOC_2D(block_x, N)
	MALLOC_2D(block_y, N)

#define TO_2DARRAY(x, n) reinterpret_cast<type_t(*)[n]>(x)

	show<type_t, N><<<numBlocks, threadsPerBlock>>>(
		TO_2DARRAY(thread_x, N),
		TO_2DARRAY(thread_y, N),
		TO_2DARRAY(block_x, N),
		TO_2DARRAY(block_y, N));

	cudaDeviceSynchronize();

	print_list_list<type_t, N>(TO_2DARRAY(thread_x, N));
	print_list_list<type_t, N>(TO_2DARRAY(thread_y, N));
	print_list_list<type_t, N>(TO_2DARRAY(block_x, N));
	print_list_list<type_t, N>(TO_2DARRAY(block_y, N));
}