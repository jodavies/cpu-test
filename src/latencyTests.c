#include <immintrin.h>
#include "latencyTests.h"

// Here we have the function definitions for the latency tests.
//
// We perform a loop-dependent operation on a vector, and
// test SSE and AVX in single and double precision.
//
// Intel vector intrinsics provide access to the instructions, but still
// need to check the produced binary with objdump to ensure the expected
// code is being produced by various compilers.
//
// Instructions implemented: (v)addp(s,d)
//                           (v)mulp(s,d)
//                           (v)divp(s,d)


//// Addition
// sse
int TestLatAddSSESP(float * RESTRICT array, CONST float scaleFac)
{
	__m128 v_scaleFac = _mm_set1_ps(scaleFac);
	__m128 v_array = _mm_load_ps(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm_add_ps(v_array, v_scaleFac);
	}
	_mm_store_ps(array, v_array);
	return 1;
}
int TestLatAddSSEDP(double * RESTRICT array, CONST double scaleFac)
{
	__m128d v_scaleFac = _mm_set1_pd(scaleFac);
	__m128d v_array = _mm_load_pd(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm_add_pd(v_array, v_scaleFac);
	}
	_mm_store_pd(array, v_array);
	return 1;
}
// avx
int TestLatAddAVXSP(float * RESTRICT array, CONST float scaleFac)
{
	__m256 v_scaleFac = _mm256_set1_ps(scaleFac);
	__m256 v_array = _mm256_load_ps(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm256_add_ps(v_array, v_scaleFac);
	}
	_mm256_store_ps(array, v_array);
	return 1;
}
int TestLatAddAVXDP(double * RESTRICT array, CONST double scaleFac)
{
	__m256d v_scaleFac = _mm256_set1_pd(scaleFac);
	__m256d v_array = _mm256_load_pd(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm256_add_pd(v_array, v_scaleFac);
	}
	_mm256_store_pd(array, v_array);
	return 1;
}


//// Multiplication
// sse
int TestLatMulSSESP(float * RESTRICT array, CONST float scaleFac)
{
	__m128 v_scaleFac = _mm_set1_ps(scaleFac);
	__m128 v_array = _mm_load_ps(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm_mul_ps(v_array, v_scaleFac);
	}
	_mm_store_ps(array, v_array);
	return 1;
}
int TestLatMulSSEDP(double * RESTRICT array, CONST double scaleFac)
{
	__m128d v_scaleFac = _mm_set1_pd(scaleFac);
	__m128d v_array = _mm_load_pd(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm_mul_pd(v_array, v_scaleFac);
	}
	_mm_store_pd(array, v_array);
	return 1;
}
// avx
int TestLatMulAVXSP(float * RESTRICT array, CONST float scaleFac)
{
	__m256 v_scaleFac = _mm256_set1_ps(scaleFac);
	__m256 v_array = _mm256_load_ps(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm256_mul_ps(v_array, v_scaleFac);
	}
	_mm256_store_ps(array, v_array);
	return 1;
}
int TestLatMulAVXDP(double * RESTRICT array, CONST double scaleFac)
{
	__m256d v_scaleFac = _mm256_set1_pd(scaleFac);
	__m256d v_array = _mm256_load_pd(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm256_mul_pd(v_array, v_scaleFac);
	}
	_mm256_store_pd(array, v_array);
	return 1;
}


//// Division
// sse
int TestLatDivSSESP(float * RESTRICT array, CONST float scaleFac)
{
	__m128 v_scaleFac = _mm_set1_ps(scaleFac);
	__m128 v_array = _mm_load_ps(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm_div_ps(v_array, v_scaleFac);
	}
	_mm_store_ps(array, v_array);
	return 1;
}
int TestLatDivSSEDP(double * RESTRICT array, CONST double scaleFac)
{
	__m128d v_scaleFac = _mm_set1_pd(scaleFac);
	__m128d v_array = _mm_load_pd(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm_div_pd(v_array, v_scaleFac);
	}
	_mm_store_pd(array, v_array);
	return 1;
}
// avx
int TestLatDivAVXSP(float * RESTRICT array, CONST float scaleFac)
{
	__m256 v_scaleFac = _mm256_set1_ps(scaleFac);
	__m256 v_array = _mm256_load_ps(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm256_div_ps(v_array, v_scaleFac);
	}
	_mm256_store_ps(array, v_array);
	return 1;
}
int TestLatDivAVXDP(double * RESTRICT array, CONST double scaleFac)
{
	__m256d v_scaleFac = _mm256_set1_pd(scaleFac);
	__m256d v_array = _mm256_load_pd(array);
	for (int i = 0; i < NTIMES; i++) {
		v_array = _mm256_div_pd(v_array, v_scaleFac);
	}
	_mm256_store_pd(array, v_array);
	return 1;
}
