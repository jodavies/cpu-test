#include <immintrin.h>
#include "throughputTests.h"

// Here we have the function definitions for the throughput tests
//
// Loop over sufficient non-dependent operations to completely hide latency
// without exceeding the number of vector registers, testing SSE and AVX in
// single and double precision
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
int TestThrMulSSESP(float * RESTRICT array, CONST float scaleFac)
{
	__m128 v_scaleFac = _mm_set1_ps(scaleFac);
	__m128 v_array0 = _mm_load_ps(&(array[0]));
	__m128 v_array1 = _mm_load_ps(&(array[4]));
	__m128 v_array2 = _mm_load_ps(&(array[8]));
	__m128 v_array3 = _mm_load_ps(&(array[12]));
	__m128 v_array4 = _mm_load_ps(&(array[16]));
	__m128 v_array5 = _mm_load_ps(&(array[20]));
	__m128 v_array6 = _mm_load_ps(&(array[24]));
	__m128 v_array7 = _mm_load_ps(&(array[28]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_add_ps(v_array0, v_scaleFac);
		v_array1 = _mm_add_ps(v_array1, v_scaleFac);
		v_array2 = _mm_add_ps(v_array2, v_scaleFac);
		v_array3 = _mm_add_ps(v_array3, v_scaleFac);
		v_array4 = _mm_add_ps(v_array4, v_scaleFac);
		v_array5 = _mm_add_ps(v_array5, v_scaleFac);
		v_array6 = _mm_add_ps(v_array6, v_scaleFac);
		v_array7 = _mm_add_ps(v_array7, v_scaleFac);
	}

	_mm_store_ps(&(array[0]), v_array0);
	_mm_store_ps(&(array[4]), v_array1);
	_mm_store_ps(&(array[8]), v_array2);
	_mm_store_ps(&(array[12]), v_array3);
	_mm_store_ps(&(array[16]), v_array4);
	_mm_store_ps(&(array[20]), v_array5);
	_mm_store_ps(&(array[24]), v_array6);
	_mm_store_ps(&(array[28]), v_array7);

	return 8;
}
int TestThrMulSSEDP(double * RESTRICT array, CONST double scaleFac)
{
	__m128d v_scaleFac = _mm_set1_pd(scaleFac);
	__m128d v_array0 = _mm_load_pd(&(array[0]));
	__m128d v_array1 = _mm_load_pd(&(array[2]));
	__m128d v_array2 = _mm_load_pd(&(array[4]));
	__m128d v_array3 = _mm_load_pd(&(array[6]));
	__m128d v_array4 = _mm_load_pd(&(array[8]));
	__m128d v_array5 = _mm_load_pd(&(array[10]));
	__m128d v_array6 = _mm_load_pd(&(array[12]));
	__m128d v_array7 = _mm_load_pd(&(array[14]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_add_pd(v_array0, v_scaleFac);
		v_array1 = _mm_add_pd(v_array1, v_scaleFac);
		v_array2 = _mm_add_pd(v_array2, v_scaleFac);
		v_array3 = _mm_add_pd(v_array3, v_scaleFac);
		v_array4 = _mm_add_pd(v_array4, v_scaleFac);
		v_array5 = _mm_add_pd(v_array5, v_scaleFac);
		v_array6 = _mm_add_pd(v_array6, v_scaleFac);
		v_array7 = _mm_add_pd(v_array7, v_scaleFac);
	}

	_mm_store_pd(&(array[0]), v_array0);
	_mm_store_pd(&(array[2]), v_array1);
	_mm_store_pd(&(array[4]), v_array2);
	_mm_store_pd(&(array[6]), v_array3);
	_mm_store_pd(&(array[8]), v_array4);
	_mm_store_pd(&(array[10]), v_array5);
	_mm_store_pd(&(array[12]), v_array6);
	_mm_store_pd(&(array[14]), v_array7);

	return 8;
}
// avx
int TestThrMulAVXSP(float * RESTRICT array, CONST float scaleFac)
{
	__m256 v_scaleFac = _mm256_set1_ps(scaleFac);
	__m256 v_array0 = _mm256_load_ps(&(array[0]));
	__m256 v_array1 = _mm256_load_ps(&(array[8]));
	__m256 v_array2 = _mm256_load_ps(&(array[16]));
	__m256 v_array3 = _mm256_load_ps(&(array[24]));
	__m256 v_array4 = _mm256_load_ps(&(array[32]));
	__m256 v_array5 = _mm256_load_ps(&(array[40]));
	__m256 v_array6 = _mm256_load_ps(&(array[48]));
	__m256 v_array7 = _mm256_load_ps(&(array[56]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_add_ps(v_array0, v_scaleFac);
		v_array1 = _mm256_add_ps(v_array1, v_scaleFac);
		v_array2 = _mm256_add_ps(v_array2, v_scaleFac);
		v_array3 = _mm256_add_ps(v_array3, v_scaleFac);
		v_array4 = _mm256_add_ps(v_array4, v_scaleFac);
		v_array5 = _mm256_add_ps(v_array5, v_scaleFac);
		v_array6 = _mm256_add_ps(v_array6, v_scaleFac);
		v_array7 = _mm256_add_ps(v_array7, v_scaleFac);
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);

	return 8;
}
int TestThrMulAVXDP(double * RESTRICT array, CONST double scaleFac)
{
	__m256d v_scaleFac = _mm256_set1_pd(scaleFac);
	__m256d v_array0 = _mm256_load_pd(&(array[0]));
	__m256d v_array1 = _mm256_load_pd(&(array[4]));
	__m256d v_array2 = _mm256_load_pd(&(array[8]));
	__m256d v_array3 = _mm256_load_pd(&(array[12]));
	__m256d v_array4 = _mm256_load_pd(&(array[16]));
	__m256d v_array5 = _mm256_load_pd(&(array[20]));
	__m256d v_array6 = _mm256_load_pd(&(array[24]));
	__m256d v_array7 = _mm256_load_pd(&(array[28]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_add_pd(v_array0, v_scaleFac);
		v_array1 = _mm256_add_pd(v_array1, v_scaleFac);
		v_array2 = _mm256_add_pd(v_array2, v_scaleFac);
		v_array3 = _mm256_add_pd(v_array3, v_scaleFac);
		v_array4 = _mm256_add_pd(v_array4, v_scaleFac);
		v_array5 = _mm256_add_pd(v_array5, v_scaleFac);
		v_array6 = _mm256_add_pd(v_array6, v_scaleFac);
		v_array7 = _mm256_add_pd(v_array7, v_scaleFac);
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);

	return 8;
}


//// Multiplication
// sse
int TestThrAddSSESP(float * RESTRICT array, CONST float scaleFac)
{
	__m128 v_scaleFac = _mm_set1_ps(scaleFac);
	__m128 v_array0 = _mm_load_ps(&(array[0]));
	__m128 v_array1 = _mm_load_ps(&(array[4]));
	__m128 v_array2 = _mm_load_ps(&(array[8]));
	__m128 v_array3 = _mm_load_ps(&(array[12]));
	__m128 v_array4 = _mm_load_ps(&(array[16]));
	__m128 v_array5 = _mm_load_ps(&(array[20]));
	__m128 v_array6 = _mm_load_ps(&(array[24]));
	__m128 v_array7 = _mm_load_ps(&(array[28]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_mul_ps(v_array0, v_scaleFac);
		v_array1 = _mm_mul_ps(v_array1, v_scaleFac);
		v_array2 = _mm_mul_ps(v_array2, v_scaleFac);
		v_array3 = _mm_mul_ps(v_array3, v_scaleFac);
		v_array4 = _mm_mul_ps(v_array4, v_scaleFac);
		v_array5 = _mm_mul_ps(v_array5, v_scaleFac);
		v_array6 = _mm_mul_ps(v_array6, v_scaleFac);
		v_array7 = _mm_mul_ps(v_array7, v_scaleFac);
	}

	_mm_store_ps(&(array[0]), v_array0);
	_mm_store_ps(&(array[4]), v_array1);
	_mm_store_ps(&(array[8]), v_array2);
	_mm_store_ps(&(array[12]), v_array3);
	_mm_store_ps(&(array[16]), v_array4);
	_mm_store_ps(&(array[20]), v_array5);
	_mm_store_ps(&(array[24]), v_array6);
	_mm_store_ps(&(array[28]), v_array7);

	return 8;
}
int TestThrAddSSEDP(double * RESTRICT array, CONST double scaleFac)
{
	__m128d v_scaleFac = _mm_set1_pd(scaleFac);
	__m128d v_array0 = _mm_load_pd(&(array[0]));
	__m128d v_array1 = _mm_load_pd(&(array[2]));
	__m128d v_array2 = _mm_load_pd(&(array[4]));
	__m128d v_array3 = _mm_load_pd(&(array[6]));
	__m128d v_array4 = _mm_load_pd(&(array[8]));
	__m128d v_array5 = _mm_load_pd(&(array[10]));
	__m128d v_array6 = _mm_load_pd(&(array[12]));
	__m128d v_array7 = _mm_load_pd(&(array[14]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_mul_pd(v_array0, v_scaleFac);
		v_array1 = _mm_mul_pd(v_array1, v_scaleFac);
		v_array2 = _mm_mul_pd(v_array2, v_scaleFac);
		v_array3 = _mm_mul_pd(v_array3, v_scaleFac);
		v_array4 = _mm_mul_pd(v_array4, v_scaleFac);
		v_array5 = _mm_mul_pd(v_array5, v_scaleFac);
		v_array6 = _mm_mul_pd(v_array6, v_scaleFac);
		v_array7 = _mm_mul_pd(v_array7, v_scaleFac);
	}

	_mm_store_pd(&(array[0]), v_array0);
	_mm_store_pd(&(array[2]), v_array1);
	_mm_store_pd(&(array[4]), v_array2);
	_mm_store_pd(&(array[6]), v_array3);
	_mm_store_pd(&(array[8]), v_array4);
	_mm_store_pd(&(array[10]), v_array5);
	_mm_store_pd(&(array[12]), v_array6);
	_mm_store_pd(&(array[14]), v_array7);

	return 8;
}
// avx
int TestThrAddAVXSP(float * RESTRICT array, CONST float scaleFac)
{
	__m256 v_scaleFac = _mm256_set1_ps(scaleFac);
	__m256 v_array0 = _mm256_load_ps(&(array[0]));
	__m256 v_array1 = _mm256_load_ps(&(array[8]));
	__m256 v_array2 = _mm256_load_ps(&(array[16]));
	__m256 v_array3 = _mm256_load_ps(&(array[24]));
	__m256 v_array4 = _mm256_load_ps(&(array[32]));
	__m256 v_array5 = _mm256_load_ps(&(array[40]));
	__m256 v_array6 = _mm256_load_ps(&(array[48]));
	__m256 v_array7 = _mm256_load_ps(&(array[56]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_mul_ps(v_array0, v_scaleFac);
		v_array1 = _mm256_mul_ps(v_array1, v_scaleFac);
		v_array2 = _mm256_mul_ps(v_array2, v_scaleFac);
		v_array3 = _mm256_mul_ps(v_array3, v_scaleFac);
		v_array4 = _mm256_mul_ps(v_array4, v_scaleFac);
		v_array5 = _mm256_mul_ps(v_array5, v_scaleFac);
		v_array6 = _mm256_mul_ps(v_array6, v_scaleFac);
		v_array7 = _mm256_mul_ps(v_array7, v_scaleFac);
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);

	return 8;
}
int TestThrAddAVXDP(double * RESTRICT array, CONST double scaleFac)
{
	__m256d v_scaleFac = _mm256_set1_pd(scaleFac);
	__m256d v_array0 = _mm256_load_pd(&(array[0]));
	__m256d v_array1 = _mm256_load_pd(&(array[4]));
	__m256d v_array2 = _mm256_load_pd(&(array[8]));
	__m256d v_array3 = _mm256_load_pd(&(array[12]));
	__m256d v_array4 = _mm256_load_pd(&(array[16]));
	__m256d v_array5 = _mm256_load_pd(&(array[20]));
	__m256d v_array6 = _mm256_load_pd(&(array[24]));
	__m256d v_array7 = _mm256_load_pd(&(array[28]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_mul_pd(v_array0, v_scaleFac);
		v_array1 = _mm256_mul_pd(v_array1, v_scaleFac);
		v_array2 = _mm256_mul_pd(v_array2, v_scaleFac);
		v_array3 = _mm256_mul_pd(v_array3, v_scaleFac);
		v_array4 = _mm256_mul_pd(v_array4, v_scaleFac);
		v_array5 = _mm256_mul_pd(v_array5, v_scaleFac);
		v_array6 = _mm256_mul_pd(v_array6, v_scaleFac);
		v_array7 = _mm256_mul_pd(v_array7, v_scaleFac);
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);

	return 8;
}


//// Division
// sse
int TestThrDivSSESP(float * RESTRICT array, CONST float scaleFac)
{
	__m128 v_scaleFac = _mm_set1_ps(scaleFac);
	__m128 v_array0 = _mm_load_ps(&(array[0]));
	__m128 v_array1 = _mm_load_ps(&(array[4]));
	__m128 v_array2 = _mm_load_ps(&(array[8]));
	__m128 v_array3 = _mm_load_ps(&(array[12]));
	__m128 v_array4 = _mm_load_ps(&(array[16]));
	__m128 v_array5 = _mm_load_ps(&(array[20]));
	__m128 v_array6 = _mm_load_ps(&(array[24]));
	__m128 v_array7 = _mm_load_ps(&(array[28]));
	__m128 v_array8 = _mm_load_ps(&(array[32]));
	__m128 v_array9 = _mm_load_ps(&(array[36]));
	__m128 v_array10 = _mm_load_ps(&(array[40]));
	__m128 v_array11 = _mm_load_ps(&(array[44]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_div_ps(v_array0, v_scaleFac);
		v_array1 = _mm_div_ps(v_array1, v_scaleFac);
		v_array2 = _mm_div_ps(v_array2, v_scaleFac);
		v_array3 = _mm_div_ps(v_array3, v_scaleFac);
		v_array4 = _mm_div_ps(v_array4, v_scaleFac);
		v_array5 = _mm_div_ps(v_array5, v_scaleFac);
		v_array6 = _mm_div_ps(v_array6, v_scaleFac);
		v_array7 = _mm_div_ps(v_array7, v_scaleFac);
		v_array8 = _mm_div_ps(v_array8, v_scaleFac);
		v_array9 = _mm_div_ps(v_array9, v_scaleFac);
		v_array10 = _mm_div_ps(v_array10, v_scaleFac);
		v_array11 = _mm_div_ps(v_array11, v_scaleFac);
	}

	_mm_store_ps(&(array[0]), v_array0);
	_mm_store_ps(&(array[4]), v_array1);
	_mm_store_ps(&(array[8]), v_array2);
	_mm_store_ps(&(array[12]), v_array3);
	_mm_store_ps(&(array[16]), v_array4);
	_mm_store_ps(&(array[20]), v_array5);
	_mm_store_ps(&(array[24]), v_array6);
	_mm_store_ps(&(array[28]), v_array7);
	_mm_store_ps(&(array[32]), v_array8);
	_mm_store_ps(&(array[36]), v_array9);
	_mm_store_ps(&(array[40]), v_array10);
	_mm_store_ps(&(array[44]), v_array11);

	return 12;
}
int TestThrDivSSEDP(double * RESTRICT array, CONST double scaleFac)
{
	__m128d v_scaleFac = _mm_set1_pd(scaleFac);
	__m128d v_array0 = _mm_load_pd(&(array[0]));
	__m128d v_array1 = _mm_load_pd(&(array[2]));
	__m128d v_array2 = _mm_load_pd(&(array[4]));
	__m128d v_array3 = _mm_load_pd(&(array[6]));
	__m128d v_array4 = _mm_load_pd(&(array[8]));
	__m128d v_array5 = _mm_load_pd(&(array[10]));
	__m128d v_array6 = _mm_load_pd(&(array[12]));
	__m128d v_array7 = _mm_load_pd(&(array[14]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_div_pd(v_array0, v_scaleFac);
		v_array1 = _mm_div_pd(v_array1, v_scaleFac);
		v_array2 = _mm_div_pd(v_array2, v_scaleFac);
		v_array3 = _mm_div_pd(v_array3, v_scaleFac);
		v_array4 = _mm_div_pd(v_array4, v_scaleFac);
		v_array5 = _mm_div_pd(v_array5, v_scaleFac);
		v_array6 = _mm_div_pd(v_array6, v_scaleFac);
		v_array7 = _mm_div_pd(v_array7, v_scaleFac);
	}

	_mm_store_pd(&(array[0]), v_array0);
	_mm_store_pd(&(array[2]), v_array1);
	_mm_store_pd(&(array[4]), v_array2);
	_mm_store_pd(&(array[6]), v_array3);
	_mm_store_pd(&(array[8]), v_array4);
	_mm_store_pd(&(array[10]), v_array5);
	_mm_store_pd(&(array[12]), v_array6);
	_mm_store_pd(&(array[14]), v_array7);

	return 8;
}
// avx
int TestThrDivAVXSP(float * RESTRICT array, CONST float scaleFac)
{
	__m256 v_scaleFac = _mm256_set1_ps(scaleFac);
	__m256 v_array0 = _mm256_load_ps(&(array[0]));
	__m256 v_array1 = _mm256_load_ps(&(array[8]));
	__m256 v_array2 = _mm256_load_ps(&(array[16]));
	__m256 v_array3 = _mm256_load_ps(&(array[24]));
	__m256 v_array4 = _mm256_load_ps(&(array[32]));
	__m256 v_array5 = _mm256_load_ps(&(array[40]));
	__m256 v_array6 = _mm256_load_ps(&(array[48]));
	__m256 v_array7 = _mm256_load_ps(&(array[56]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_div_ps(v_array0, v_scaleFac);
		v_array1 = _mm256_div_ps(v_array1, v_scaleFac);
		v_array2 = _mm256_div_ps(v_array2, v_scaleFac);
		v_array3 = _mm256_div_ps(v_array3, v_scaleFac);
		v_array4 = _mm256_div_ps(v_array4, v_scaleFac);
		v_array5 = _mm256_div_ps(v_array5, v_scaleFac);
		v_array6 = _mm256_div_ps(v_array6, v_scaleFac);
		v_array7 = _mm256_div_ps(v_array7, v_scaleFac);
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);

	return 8;
}
int TestThrDivAVXDP(double * RESTRICT array, CONST double scaleFac)
{
	__m256d v_scaleFac = _mm256_set1_pd(scaleFac);
	__m256d v_array0 = _mm256_load_pd(&(array[0]));
	__m256d v_array1 = _mm256_load_pd(&(array[4]));
	__m256d v_array2 = _mm256_load_pd(&(array[8]));
	__m256d v_array3 = _mm256_load_pd(&(array[12]));
	__m256d v_array4 = _mm256_load_pd(&(array[16]));
	__m256d v_array5 = _mm256_load_pd(&(array[20]));
	__m256d v_array6 = _mm256_load_pd(&(array[24]));
	__m256d v_array7 = _mm256_load_pd(&(array[28]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_div_pd(v_array0, v_scaleFac);
		v_array1 = _mm256_div_pd(v_array1, v_scaleFac);
		v_array2 = _mm256_div_pd(v_array2, v_scaleFac);
		v_array3 = _mm256_div_pd(v_array3, v_scaleFac);
		v_array4 = _mm256_div_pd(v_array4, v_scaleFac);
		v_array5 = _mm256_div_pd(v_array5, v_scaleFac);
		v_array6 = _mm256_div_pd(v_array6, v_scaleFac);
		v_array7 = _mm256_div_pd(v_array7, v_scaleFac);
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);

	return 8;
}
