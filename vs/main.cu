#include "cuda_fp16.h"
#include "cudaUtil.cuh"
#include "cuda_runtime.h"
#include "cuda_uint128.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>

#define my_swap(x,y) x ^= y, y ^= x, x ^= y

constexpr uint64_t MOD = 0xFFFFFFFF00000001;
constexpr uint64_t ROOT = 17492915097719143606;
int L;
uint64_t* rev;

//#ifdef __CUDA_ARCH__
__constant__ uint64_t d_MOD = 0xFFFFFFFF00000001;
__constant__ uint64_t d_ROOT = 17492915097719143606;
__device__ uint64_t d_r, d_mid, d_wn;
//#endif // __CUDA_ARCH__

template<typename Scalar>
_CUDA_GENERAL_CALL_ Scalar nextPow2(Scalar x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

/**
* 大数乘法
*/
_CUDA_GENERAL_CALL_ uint64_t modularMultiplication(uint64_t a, uint64_t b)
{
	//#ifdef __CUDA_ARCH__  // CUDA设备端实现
	//	uint64_t resultLow, resultHigh;
	//	asm volatile("mul.lo.u64 %0, %1, %2;\n" : "=l"(resultLow) : "l"(a), "l"(b));
	//	asm volatile("mul.hi.u64 %0, %1, %2;\n" : "=l"(resultHigh) : "l"(a), "l"(b));
	//
	//	uint64_t quotient = (resultHigh / d_MOD) << 64;
	//	uint64_t remainder = resultHigh - quotient * d_MOD;
	//
	//	quotient += resultLow / d_MOD;
	//	remainder = (remainder << 64) + (resultLow - quotient * d_MOD);
	//
	//	uint64_t temp = (remainder << 64) + resultLow;
	//	if (temp >= d_MOD)
	//	{
	//		temp -= d_MOD;
	//	}
	//	temp += quotient * d_MOD;
	//	if (temp >= d_MOD)
	//	{
	//		temp -= d_MOD;
	//	}
	//	return temp % d_MOD;
	//#else  // 主机端实现
	uint64_t a_lo = (uint32_t)a;
	uint64_t a_hi = a >> 32;
	uint64_t b_lo = (uint32_t)b;
	uint64_t b_hi = b >> 32;

	uint64_t a_x_b_hi = a_hi * b_hi;
	uint64_t a_x_b_mid = a_hi * b_lo;
	uint64_t b_x_a_mid = b_hi * a_lo;
	uint64_t a_x_b_lo = a_lo * b_lo;

	/*
		This is implementing schoolbook multiplication:

				x1 x0
		X       y1 y0
		-------------
				   00  LOW PART
		-------------
				00
			 10 10     MIDDLE PART
		+       01
		-------------
			 01
		+ 11 11        HIGH PART
		-------------
	*/

	// 64-bit product + two 32-bit values
	uint64_t middle = a_x_b_mid + (a_x_b_lo >> 32) + uint32_t(b_x_a_mid);

	// 64-bit product + two 32-bit values
	uint64_t carry = a_x_b_hi + (middle >> 32) + (b_x_a_mid >> 32);

	// Add LOW PART and lower half of MIDDLE PART
	uint64_t result = ((middle << 32) | uint32_t(a_x_b_lo));

	/*uint64_t result = 0;
	while (a > 0)
	{
		if (a & 1)
		{
			result = (result + b) % MOD;
		}
		a >>= 1;
		b = (b << 1) % MOD;
	}*/
#ifdef __CUDA_ARCH__  // CUDA设备端实现
	return result % d_MOD;
#else
	return result % MOD;
#endif
	//#endif // __CUDA_ARCH__
}

//inline __device__ uint64_t modularMultiplication(uint64_t a, uint64_t b)
//{
//    uint64_t result = 0;
//    a %= MOD; b %= MOD;
//
//    while (b)
//    {
//        if (b & 1) result = (result + a) % MOD;
//
//        a = (a << 1) % MOD;
//        b >>= 1;
//    }
//
//    return result;
//}

/**
* 快速幂
*/
inline _CUDA_GENERAL_CALL_ uint64_t modularExponentiation(uint128_t base, uint64_t exponent)
{
#ifdef __CUDA_ARCH__  // CUDA设备端实现
	uint128_t result(1);
	while (exponent > 0)
	{
		if (exponent & 1) result = (result * uint128_t::u128tou64(base)) % d_MOD;

		//base = modularMultiplication(base, base);
		base = base * uint128_t::u128tou64(base) % d_MOD;
		exponent >>= 1;
	}

	return result % d_MOD;
#else
	uint128_t result(1);
	while (exponent > 0)
	{
		if (exponent & 1) result = (result * uint128_t::u128tou64(base)) % MOD;

		//base = modularMultiplication(base, base);
		base = base * uint128_t::u128tou64(base) % MOD;
		exponent >>= 1;
	}
	return result % MOD;
#endif
}

__global__ void nttKernel(const uint64_t numDivGroups, uint64_t* d_data)
{
	unsigned int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_idx = threadIdx.y + blockIdx.y * blockDim.y;

	if (x_idx < numDivGroups && y_idx < d_mid)
	{
		const uint64_t omega = modularExponentiation(d_wn, y_idx);
		//printf("d_wn = %lu, omega = %llu\n", (unsigned long long)d_wn, (unsigned long long)omega);

		uint128_t u = d_data[x_idx * d_r + y_idx];
		uint128_t v = (uint128_t)d_data[x_idx * d_r + y_idx + d_mid] * omega;

		d_data[x_idx * d_r + y_idx] = (u + v) % d_MOD;
		d_data[x_idx * d_r + y_idx + d_mid] = (u - v + d_MOD) % d_MOD;
	}
}

//namespace {
//	// Estimate best block and grid size using CUDA Occupancy Calculator
//	int blockSize;   // The launch configurator returned block size 
//	int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
//	int gridSize;    // The actual grid size needed, based on input size 
//}

void launchNTT(const bool& isInverse, const uint64_t& paddedN, uint64_t* data)
{
	uint64_t* d_data;
	CUDA_CHECK(cudaMalloc((void**)&d_data, paddedN * sizeof(uint64_t)));
	CUDA_CHECK(cudaMemcpy(d_data, data, paddedN * sizeof(uint64_t), cudaMemcpyHostToDevice));

	/*uint64_t* d_wn;
	CUDA_CHECK(cudaMalloc((void**)&d_wn, sizeof(uint64_t)));

	uint64_t* d_mid;
	CUDA_CHECK(cudaMalloc((void**)&d_mid, sizeof(uint64_t)));*/
	dim3 blockSize, gridSize;
	blockSize.x = 8, blockSize.y = 128;
	for (int k = 1; k <= L; k++)
	{
		uint64_t mid = (1ULL) << (k - 1);
		//CUDA_CHECK(cudaMemcpy(d_mid, &mid, sizeof(uint64_t), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol(d_mid, &mid, sizeof(uint64_t)));
		uint64_t wn = modularExponentiation((uint128_t)ROOT, ((MOD - 1) >> k));
		std::cout << "wn = " << wn << std::endl;
		system("pause");
		//CUDA_CHECK(cudaMemcpy(d_wn, &wn, sizeof(uint64_t), cudaMemcpyHostToDevice));
		if (isInverse) wn = modularExponentiation(wn, MOD - 2);
		CUDA_CHECK(cudaMemcpyToSymbol(d_wn, &wn, sizeof(uint64_t)));
		//uint64_t numGroups = (1ULL) << (k - 1);
		//uint64_t r = (1ULL) << (k);
		uint64_t r = mid << 1;
		uint64_t numDivGroups = (paddedN + r - 1) / r;
		CUDA_CHECK(cudaMemcpyToSymbol(d_r, &r, sizeof(uint64_t)));

		gridSize.x = (numDivGroups + blockSize.x - 1) / blockSize.x;
		gridSize.y = (mid + blockSize.y - 1) / blockSize.y;

		nttKernel << <gridSize, blockSize >> > (numDivGroups, d_data);
		getLastCudaError("Kernel 'nttKernel' launch failed!\n");

		cudaDeviceSynchronize();
	}

	CUDA_CHECK(cudaMemcpy(data, d_data, paddedN * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_data));
}

//void inverseNtt(const uint64_t& N, uint64_t* data)
//{
//	uint64_t paddedN = nextPow2(N);
//	uint64_t L = log2(paddedN);
//
//	uint64_t* dev_data;
//	CUDA_CHECK(cudaMalloc((void**)&dev_data, paddedN * sizeof(uint64_t)));
//	CUDA_CHECK(cudaMemcpy(dev_data, data, paddedN * sizeof(uint64_t), cudaMemcpyHostToDevice));
//
//	for (uint64_t k = L; k >= 1; k--)
//	{
//		uint64_t m = (1ULL) << k;
//
//		/*getOccupancyMaxPotentialBlockSize(paddedN, minGridSize, blockSize, gridSize, nttKernel, 0, 0);
//		nttKernel << <gridSize, blockSize >> > (paddedN, L, k, dev_data);
//		getLastCudaError("Kernel 'nttKernel' launch failed!\n");*/
//
//		cudaDeviceSynchronize();
//	}
//
//	uint64_t inversedN = modularExponentiation(paddedN, MOD - 2); // Compute modular inverse of paddedN
//	uint64_t inversedNMod = modularExponentiation(inversedN, N);
//
//	for (uint64_t i = 0; i < paddedN; i++)
//	{
//		data[i] = modularMultiplication(data[i], inversedNMod);
//	}
//
//	CUDA_CHECK(cudaMemcpy(data, dev_data, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//	CUDA_CHECK(cudaFree(dev_data));
//}

void polynomialMultiply(const uint64_t* coeffA, const uint64_t& degreeA, const uint64_t* coeffB, const uint64_t& degreeB, uint64_t* result)
{
	uint64_t degreeLimit = degreeA + degreeB;
	uint64_t paddedDegreeSize = 1;
	while (paddedDegreeSize <= degreeLimit) paddedDegreeSize <<= 1, ++L;
	/*uint64_t paddedDegreeSize = (degreeLimit == 0 ? 1 : nextPow2(degreeLimit));
	L = log2(paddedDegreeSize);*/

	uint64_t* tempA = new uint64_t[paddedDegreeSize];
	uint64_t* tempB = new uint64_t[paddedDegreeSize];
	rev = new uint64_t[paddedDegreeSize];

	std::fill(tempA, tempA + paddedDegreeSize, 0);
	std::fill(tempB, tempB + paddedDegreeSize, 0);
	std::copy(coeffA, coeffA + degreeA + 1, tempA);
	std::copy(coeffB, coeffB + degreeB + 1, tempB);

	std::fill(rev, rev + paddedDegreeSize, 0);
	for (int i = 0; i < paddedDegreeSize; i++)
	{
		rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
		if (i < rev[i])
		{
			my_swap(tempA[i], tempA[rev[i]]);
			my_swap(tempB[i], tempB[rev[i]]);
		}
	}

	launchNTT(false, paddedDegreeSize, tempA);
	launchNTT(false, paddedDegreeSize, tempB);

	for (uint64_t i = 0; i < paddedDegreeSize; ++i)
	{
		tempA[i] = modularMultiplication(tempA[i], tempB[i]);
	}

	launchNTT(true, paddedDegreeSize, tempA);
	//inverseNtt(true, paddedSize, tempA);

	std::copy(tempA, tempA + degreeLimit + 1, result);

	delete[] tempA;
	delete[] tempB;
	delete[] rev;
}

int main(int argc, char** argv)
{
	// 最高次数
	uint64_t degreeA, degreeB;
	std::cin >> degreeA >> degreeB;

	uint64_t* coeffA = new uint64_t[degreeA + 1];
	uint64_t* coeffB = new uint64_t[degreeB + 1];

	// 从低到高的系数
	for (int i = 0; i <= degreeA; ++i) std::cin >> coeffA[i];
	for (int i = 0; i <= degreeB; ++i) std::cin >> coeffB[i];

	// 从低到高的系数
	//uint64_t coeffA[] = { 1, 2, 1 }; // A(x) = x^2 + 2x + 1
	//uint64_t coeffB[] = { 3, 4, 5, 6 }; // B(x) = 6x^3 + 5x^2 + 4x + 3

	uint64_t degreeLimit = degreeA + degreeB; // 卷积后的最高次数
	uint64_t* result = new uint64_t[degreeLimit + 1];

	polynomialMultiply(coeffA, degreeA, coeffB, degreeB, result);

	std::cout << "Result of polynomial multiplication:" << std::endl;
	for (uint64_t i = 0; i <= degreeLimit; i++)
	{
		std::cout << result[i];
		if (i == degreeLimit) std::cout << std::endl;
		else std::cout << " ";
	}

	delete[] result;

	return 0;
}