#include <immintrin.h>
#include <x86intrin.h>
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
	__m128 v_array8 = _mm_load_ps(&(array[32]));
	__m128 v_array9 = _mm_load_ps(&(array[36]));
	__m128 v_array10 = _mm_load_ps(&(array[40]));
	__m128 v_array11 = _mm_load_ps(&(array[44]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_add_ps(v_array0, v_scaleFac);
		v_array1 = _mm_add_ps(v_array1, v_scaleFac);
		v_array2 = _mm_add_ps(v_array2, v_scaleFac);
		v_array3 = _mm_add_ps(v_array3, v_scaleFac);
		v_array4 = _mm_add_ps(v_array4, v_scaleFac);
		v_array5 = _mm_add_ps(v_array5, v_scaleFac);
		v_array6 = _mm_add_ps(v_array6, v_scaleFac);
		v_array7 = _mm_add_ps(v_array7, v_scaleFac);
		v_array8 = _mm_add_ps(v_array8, v_scaleFac);
		v_array9 = _mm_add_ps(v_array9, v_scaleFac);
		v_array10 = _mm_add_ps(v_array10, v_scaleFac);
		v_array11 = _mm_add_ps(v_array11, v_scaleFac);
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
	__m128d v_array8 = _mm_load_pd(&(array[16]));
	__m128d v_array9 = _mm_load_pd(&(array[18]));
	__m128d v_array10 = _mm_load_pd(&(array[20]));
	__m128d v_array11 = _mm_load_pd(&(array[22]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_add_pd(v_array0, v_scaleFac);
		v_array1 = _mm_add_pd(v_array1, v_scaleFac);
		v_array2 = _mm_add_pd(v_array2, v_scaleFac);
		v_array3 = _mm_add_pd(v_array3, v_scaleFac);
		v_array4 = _mm_add_pd(v_array4, v_scaleFac);
		v_array5 = _mm_add_pd(v_array5, v_scaleFac);
		v_array6 = _mm_add_pd(v_array6, v_scaleFac);
		v_array7 = _mm_add_pd(v_array7, v_scaleFac);
		v_array8 = _mm_add_pd(v_array8, v_scaleFac);
		v_array9 = _mm_add_pd(v_array9, v_scaleFac);
		v_array10 = _mm_add_pd(v_array10, v_scaleFac);
		v_array11 = _mm_add_pd(v_array11, v_scaleFac);
	}

	_mm_store_pd(&(array[0]), v_array0);
	_mm_store_pd(&(array[2]), v_array1);
	_mm_store_pd(&(array[4]), v_array2);
	_mm_store_pd(&(array[6]), v_array3);
	_mm_store_pd(&(array[8]), v_array4);
	_mm_store_pd(&(array[10]), v_array5);
	_mm_store_pd(&(array[12]), v_array6);
	_mm_store_pd(&(array[14]), v_array7);
	_mm_store_pd(&(array[16]), v_array8);
	_mm_store_pd(&(array[18]), v_array9);
	_mm_store_pd(&(array[20]), v_array10);
	_mm_store_pd(&(array[22]), v_array11);

	return 12;
}
// avx
#ifdef WITHAVX
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
	__m256 v_array8 = _mm256_load_ps(&(array[64]));
	__m256 v_array9 = _mm256_load_ps(&(array[72]));
	__m256 v_array10 = _mm256_load_ps(&(array[80]));
	__m256 v_array11 = _mm256_load_ps(&(array[88]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_add_ps(v_array0, v_scaleFac);
		v_array1 = _mm256_add_ps(v_array1, v_scaleFac);
		v_array2 = _mm256_add_ps(v_array2, v_scaleFac);
		v_array3 = _mm256_add_ps(v_array3, v_scaleFac);
		v_array4 = _mm256_add_ps(v_array4, v_scaleFac);
		v_array5 = _mm256_add_ps(v_array5, v_scaleFac);
		v_array6 = _mm256_add_ps(v_array6, v_scaleFac);
		v_array7 = _mm256_add_ps(v_array7, v_scaleFac);
		v_array8 = _mm256_add_ps(v_array8, v_scaleFac);
		v_array9 = _mm256_add_ps(v_array9, v_scaleFac);
		v_array10 = _mm256_add_ps(v_array10, v_scaleFac);
		v_array11 = _mm256_add_ps(v_array11, v_scaleFac);
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);
	_mm256_store_ps(&(array[64]), v_array8);
	_mm256_store_ps(&(array[72]), v_array9);
	_mm256_store_ps(&(array[80]), v_array10);
	_mm256_store_ps(&(array[88]), v_array11);

	return 12;
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
	__m256d v_array8 = _mm256_load_pd(&(array[32]));
	__m256d v_array9 = _mm256_load_pd(&(array[36]));
	__m256d v_array10 = _mm256_load_pd(&(array[40]));
	__m256d v_array11 = _mm256_load_pd(&(array[44]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_add_pd(v_array0, v_scaleFac);
		v_array1 = _mm256_add_pd(v_array1, v_scaleFac);
		v_array2 = _mm256_add_pd(v_array2, v_scaleFac);
		v_array3 = _mm256_add_pd(v_array3, v_scaleFac);
		v_array4 = _mm256_add_pd(v_array4, v_scaleFac);
		v_array5 = _mm256_add_pd(v_array5, v_scaleFac);
		v_array6 = _mm256_add_pd(v_array6, v_scaleFac);
		v_array7 = _mm256_add_pd(v_array7, v_scaleFac);
		v_array8 = _mm256_add_pd(v_array8, v_scaleFac);
		v_array9 = _mm256_add_pd(v_array9, v_scaleFac);
		v_array10 = _mm256_add_pd(v_array10, v_scaleFac);
		v_array11 = _mm256_add_pd(v_array11, v_scaleFac);
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);
	_mm256_store_pd(&(array[32]), v_array8);
	_mm256_store_pd(&(array[36]), v_array9);
	_mm256_store_pd(&(array[40]), v_array10);
	_mm256_store_pd(&(array[44]), v_array11);

	return 12;
}
#endif


//// Multiplication
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
	__m128 v_array8 = _mm_load_ps(&(array[32]));
	__m128 v_array9 = _mm_load_ps(&(array[36]));
	__m128 v_array10 = _mm_load_ps(&(array[40]));
	__m128 v_array11 = _mm_load_ps(&(array[44]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_mul_ps(v_array0, v_scaleFac);
		v_array1 = _mm_mul_ps(v_array1, v_scaleFac);
		v_array2 = _mm_mul_ps(v_array2, v_scaleFac);
		v_array3 = _mm_mul_ps(v_array3, v_scaleFac);
		v_array4 = _mm_mul_ps(v_array4, v_scaleFac);
		v_array5 = _mm_mul_ps(v_array5, v_scaleFac);
		v_array6 = _mm_mul_ps(v_array6, v_scaleFac);
		v_array7 = _mm_mul_ps(v_array7, v_scaleFac);
		v_array8 = _mm_mul_ps(v_array8, v_scaleFac);
		v_array9 = _mm_mul_ps(v_array9, v_scaleFac);
		v_array10 = _mm_mul_ps(v_array10, v_scaleFac);
		v_array11 = _mm_mul_ps(v_array11, v_scaleFac);
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
	__m128d v_array8 = _mm_load_pd(&(array[16]));
	__m128d v_array9 = _mm_load_pd(&(array[18]));
	__m128d v_array10 = _mm_load_pd(&(array[20]));
	__m128d v_array11 = _mm_load_pd(&(array[22]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_mul_pd(v_array0, v_scaleFac);
		v_array1 = _mm_mul_pd(v_array1, v_scaleFac);
		v_array2 = _mm_mul_pd(v_array2, v_scaleFac);
		v_array3 = _mm_mul_pd(v_array3, v_scaleFac);
		v_array4 = _mm_mul_pd(v_array4, v_scaleFac);
		v_array5 = _mm_mul_pd(v_array5, v_scaleFac);
		v_array6 = _mm_mul_pd(v_array6, v_scaleFac);
		v_array7 = _mm_mul_pd(v_array7, v_scaleFac);
		v_array8 = _mm_mul_pd(v_array8, v_scaleFac);
		v_array9 = _mm_mul_pd(v_array9, v_scaleFac);
		v_array10 = _mm_mul_pd(v_array10, v_scaleFac);
		v_array11 = _mm_mul_pd(v_array11, v_scaleFac);
	}

	_mm_store_pd(&(array[0]), v_array0);
	_mm_store_pd(&(array[2]), v_array1);
	_mm_store_pd(&(array[4]), v_array2);
	_mm_store_pd(&(array[6]), v_array3);
	_mm_store_pd(&(array[8]), v_array4);
	_mm_store_pd(&(array[10]), v_array5);
	_mm_store_pd(&(array[12]), v_array6);
	_mm_store_pd(&(array[14]), v_array7);
	_mm_store_pd(&(array[16]), v_array8);
	_mm_store_pd(&(array[18]), v_array9);
	_mm_store_pd(&(array[20]), v_array10);
	_mm_store_pd(&(array[22]), v_array11);

	return 12;
}
// avx
#ifdef WITHAVX
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
	__m256 v_array8 = _mm256_load_ps(&(array[64]));
	__m256 v_array9 = _mm256_load_ps(&(array[72]));
	__m256 v_array10 = _mm256_load_ps(&(array[80]));
	__m256 v_array11 = _mm256_load_ps(&(array[88]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_mul_ps(v_array0, v_scaleFac);
		v_array1 = _mm256_mul_ps(v_array1, v_scaleFac);
		v_array2 = _mm256_mul_ps(v_array2, v_scaleFac);
		v_array3 = _mm256_mul_ps(v_array3, v_scaleFac);
		v_array4 = _mm256_mul_ps(v_array4, v_scaleFac);
		v_array5 = _mm256_mul_ps(v_array5, v_scaleFac);
		v_array6 = _mm256_mul_ps(v_array6, v_scaleFac);
		v_array7 = _mm256_mul_ps(v_array7, v_scaleFac);
		v_array8 = _mm256_mul_ps(v_array8, v_scaleFac);
		v_array9 = _mm256_mul_ps(v_array9, v_scaleFac);
		v_array10 = _mm256_mul_ps(v_array10, v_scaleFac);
		v_array11 = _mm256_mul_ps(v_array11, v_scaleFac);
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);
	_mm256_store_ps(&(array[64]), v_array8);
	_mm256_store_ps(&(array[72]), v_array9);
	_mm256_store_ps(&(array[80]), v_array10);
	_mm256_store_ps(&(array[88]), v_array11);

	return 12;
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
	__m256d v_array8 = _mm256_load_pd(&(array[32]));
	__m256d v_array9 = _mm256_load_pd(&(array[36]));
	__m256d v_array10 = _mm256_load_pd(&(array[40]));
	__m256d v_array11 = _mm256_load_pd(&(array[44]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_mul_pd(v_array0, v_scaleFac);
		v_array1 = _mm256_mul_pd(v_array1, v_scaleFac);
		v_array2 = _mm256_mul_pd(v_array2, v_scaleFac);
		v_array3 = _mm256_mul_pd(v_array3, v_scaleFac);
		v_array4 = _mm256_mul_pd(v_array4, v_scaleFac);
		v_array5 = _mm256_mul_pd(v_array5, v_scaleFac);
		v_array6 = _mm256_mul_pd(v_array6, v_scaleFac);
		v_array7 = _mm256_mul_pd(v_array7, v_scaleFac);
		v_array8 = _mm256_mul_pd(v_array8, v_scaleFac);
		v_array9 = _mm256_mul_pd(v_array9, v_scaleFac);
		v_array10 = _mm256_mul_pd(v_array10, v_scaleFac);
		v_array11 = _mm256_mul_pd(v_array11, v_scaleFac);
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);
	_mm256_store_pd(&(array[32]), v_array8);
	_mm256_store_pd(&(array[36]), v_array9);
	_mm256_store_pd(&(array[40]), v_array10);
	_mm256_store_pd(&(array[44]), v_array11);

	return 12;
}
#endif


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
	__m128 v_array12 = _mm_load_ps(&(array[48]));
	__m128 v_array13 = _mm_load_ps(&(array[52]));
	__m128 v_array14 = _mm_load_ps(&(array[56]));
	
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
		v_array12 = _mm_div_ps(v_array12, v_scaleFac);
		v_array13 = _mm_div_ps(v_array13, v_scaleFac);
		v_array14 = _mm_div_ps(v_array14, v_scaleFac);
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
	_mm_store_ps(&(array[48]), v_array12);
	_mm_store_ps(&(array[52]), v_array13);
	_mm_store_ps(&(array[56]), v_array14);

	return 15;
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
	__m128d v_array8 = _mm_load_pd(&(array[16]));
	__m128d v_array9 = _mm_load_pd(&(array[18]));
	__m128d v_array10 = _mm_load_pd(&(array[20]));
	__m128d v_array11 = _mm_load_pd(&(array[22]));
	__m128d v_array12 = _mm_load_pd(&(array[24]));
	__m128d v_array13 = _mm_load_pd(&(array[26]));
	__m128d v_array14 = _mm_load_pd(&(array[28]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm_div_pd(v_array0, v_scaleFac);
		v_array1 = _mm_div_pd(v_array1, v_scaleFac);
		v_array2 = _mm_div_pd(v_array2, v_scaleFac);
		v_array3 = _mm_div_pd(v_array3, v_scaleFac);
		v_array4 = _mm_div_pd(v_array4, v_scaleFac);
		v_array5 = _mm_div_pd(v_array5, v_scaleFac);
		v_array6 = _mm_div_pd(v_array6, v_scaleFac);
		v_array7 = _mm_div_pd(v_array7, v_scaleFac);
		v_array8 = _mm_div_pd(v_array8, v_scaleFac);
		v_array9 = _mm_div_pd(v_array9, v_scaleFac);
		v_array10 = _mm_div_pd(v_array10, v_scaleFac);
		v_array11 = _mm_div_pd(v_array11, v_scaleFac);
		v_array12 = _mm_div_pd(v_array12, v_scaleFac);
		v_array13 = _mm_div_pd(v_array13, v_scaleFac);
		v_array14 = _mm_div_pd(v_array14, v_scaleFac);
	}

	_mm_store_pd(&(array[0]), v_array0);
	_mm_store_pd(&(array[2]), v_array1);
	_mm_store_pd(&(array[4]), v_array2);
	_mm_store_pd(&(array[6]), v_array3);
	_mm_store_pd(&(array[8]), v_array4);
	_mm_store_pd(&(array[10]), v_array5);
	_mm_store_pd(&(array[12]), v_array6);
	_mm_store_pd(&(array[14]), v_array7);
	_mm_store_pd(&(array[16]), v_array8);
	_mm_store_pd(&(array[18]), v_array9);
	_mm_store_pd(&(array[20]), v_array10);
	_mm_store_pd(&(array[22]), v_array11);
	_mm_store_pd(&(array[24]), v_array12);
	_mm_store_pd(&(array[26]), v_array13);
	_mm_store_pd(&(array[28]), v_array14);

	return 15;
}
// avx
#ifdef WITHAVX
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
	__m256 v_array8 = _mm256_load_ps(&(array[64]));
	__m256 v_array9 = _mm256_load_ps(&(array[72]));
	__m256 v_array10 = _mm256_load_ps(&(array[80]));
	__m256 v_array11 = _mm256_load_ps(&(array[88]));
	__m256 v_array12 = _mm256_load_ps(&(array[96]));
	__m256 v_array13 = _mm256_load_ps(&(array[104]));
	__m256 v_array14 = _mm256_load_ps(&(array[112]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_div_ps(v_array0, v_scaleFac);
		v_array1 = _mm256_div_ps(v_array1, v_scaleFac);
		v_array2 = _mm256_div_ps(v_array2, v_scaleFac);
		v_array3 = _mm256_div_ps(v_array3, v_scaleFac);
		v_array4 = _mm256_div_ps(v_array4, v_scaleFac);
		v_array5 = _mm256_div_ps(v_array5, v_scaleFac);
		v_array6 = _mm256_div_ps(v_array6, v_scaleFac);
		v_array7 = _mm256_div_ps(v_array7, v_scaleFac);
		v_array8 = _mm256_div_ps(v_array8, v_scaleFac);
		v_array9 = _mm256_div_ps(v_array9, v_scaleFac);
		v_array10 = _mm256_div_ps(v_array10, v_scaleFac);
		v_array11 = _mm256_div_ps(v_array11, v_scaleFac);
		v_array12 = _mm256_div_ps(v_array12, v_scaleFac);
		v_array13 = _mm256_div_ps(v_array13, v_scaleFac);
		v_array14 = _mm256_div_ps(v_array14, v_scaleFac);
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);
	_mm256_store_ps(&(array[64]), v_array8);
	_mm256_store_ps(&(array[72]), v_array9);
	_mm256_store_ps(&(array[80]), v_array10);
	_mm256_store_ps(&(array[88]), v_array11);
	_mm256_store_ps(&(array[96]), v_array12);
	_mm256_store_ps(&(array[104]), v_array13);
	_mm256_store_ps(&(array[112]), v_array14);

	return 15;
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
	__m256d v_array8 = _mm256_load_pd(&(array[32]));
	__m256d v_array9 = _mm256_load_pd(&(array[36]));
	__m256d v_array10 = _mm256_load_pd(&(array[40]));
	__m256d v_array11 = _mm256_load_pd(&(array[44]));
	__m256d v_array12 = _mm256_load_pd(&(array[48]));
	__m256d v_array13 = _mm256_load_pd(&(array[52]));
	__m256d v_array14 = _mm256_load_pd(&(array[56]));
	
	for (int i = 0; i < NTIMES; i++) {
		v_array0 = _mm256_div_pd(v_array0, v_scaleFac);
		v_array1 = _mm256_div_pd(v_array1, v_scaleFac);
		v_array2 = _mm256_div_pd(v_array2, v_scaleFac);
		v_array3 = _mm256_div_pd(v_array3, v_scaleFac);
		v_array4 = _mm256_div_pd(v_array4, v_scaleFac);
		v_array5 = _mm256_div_pd(v_array5, v_scaleFac);
		v_array6 = _mm256_div_pd(v_array6, v_scaleFac);
		v_array7 = _mm256_div_pd(v_array7, v_scaleFac);
		v_array8 = _mm256_div_pd(v_array8, v_scaleFac);
		v_array9 = _mm256_div_pd(v_array9, v_scaleFac);
		v_array10 = _mm256_div_pd(v_array10, v_scaleFac);
		v_array11 = _mm256_div_pd(v_array11, v_scaleFac);
		v_array12 = _mm256_div_pd(v_array12, v_scaleFac);
		v_array13 = _mm256_div_pd(v_array13, v_scaleFac);
		v_array14 = _mm256_div_pd(v_array14, v_scaleFac);
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);
	_mm256_store_pd(&(array[32]), v_array8);
	_mm256_store_pd(&(array[36]), v_array9);
	_mm256_store_pd(&(array[40]), v_array10);
	_mm256_store_pd(&(array[44]), v_array11);
	_mm256_store_pd(&(array[48]), v_array12);
	_mm256_store_pd(&(array[52]), v_array13);
	_mm256_store_pd(&(array[56]), v_array14);

	return 15;
}
#endif


// FMA
#ifdef WITHFMA
int TestThrFMAAVXSP(float * RESTRICT array, CONST float scaleFac)
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
	__m256 v_array8 = _mm256_load_ps(&(array[64]));
	__m256 v_array9 = _mm256_load_ps(&(array[72]));
	__m256 v_array10 = _mm256_load_ps(&(array[80]));
	__m256 v_array11 = _mm256_load_ps(&(array[88]));


	for (int i = 0; i < NTIMES; i++) {
		v_array0 = FMAINTRINAVXSP(v_array0, v_scaleFac, v_array0);
		v_array1 = FMAINTRINAVXSP(v_array1, v_scaleFac, v_array1);
		v_array2 = FMAINTRINAVXSP(v_array2, v_scaleFac, v_array2);
		v_array3 = FMAINTRINAVXSP(v_array3, v_scaleFac, v_array3);
		v_array4 = FMAINTRINAVXSP(v_array4, v_scaleFac, v_array4);
		v_array5 = FMAINTRINAVXSP(v_array5, v_scaleFac, v_array5);
		v_array6 = FMAINTRINAVXSP(v_array6, v_scaleFac, v_array6);
		v_array7 = FMAINTRINAVXSP(v_array7, v_scaleFac, v_array7);
		v_array8 = FMAINTRINAVXSP(v_array8, v_scaleFac, v_array8);
		v_array9 = FMAINTRINAVXSP(v_array9, v_scaleFac, v_array9);
		v_array10 = FMAINTRINAVXSP(v_array10, v_scaleFac, v_array10);
		v_array11 = FMAINTRINAVXSP(v_array11, v_scaleFac, v_array11);
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);
	_mm256_store_ps(&(array[64]), v_array8);
	_mm256_store_ps(&(array[72]), v_array9);
	_mm256_store_ps(&(array[80]), v_array10);
	_mm256_store_ps(&(array[88]), v_array11);

	return 12;
}
int TestThrFMAAVXDP(double * RESTRICT array, CONST double scaleFac)
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
	__m256d v_array8 = _mm256_load_pd(&(array[32]));
	__m256d v_array9 = _mm256_load_pd(&(array[36]));
	__m256d v_array10 = _mm256_load_pd(&(array[40]));
	__m256d v_array11 = _mm256_load_pd(&(array[44]));

	for (int i = 0; i < NTIMES; i++) {
		v_array0 = FMAINTRINAVXDP(v_array0, v_scaleFac, v_array0);
		v_array1 = FMAINTRINAVXDP(v_array1, v_scaleFac, v_array1);
		v_array2 = FMAINTRINAVXDP(v_array2, v_scaleFac, v_array2);
		v_array3 = FMAINTRINAVXDP(v_array3, v_scaleFac, v_array3);
		v_array4 = FMAINTRINAVXDP(v_array4, v_scaleFac, v_array4);
		v_array5 = FMAINTRINAVXDP(v_array5, v_scaleFac, v_array5);
		v_array6 = FMAINTRINAVXDP(v_array6, v_scaleFac, v_array6);
		v_array7 = FMAINTRINAVXDP(v_array7, v_scaleFac, v_array7);
		v_array8 = FMAINTRINAVXDP(v_array8, v_scaleFac, v_array8);
		v_array9 = FMAINTRINAVXDP(v_array9, v_scaleFac, v_array9);
		v_array10 = FMAINTRINAVXDP(v_array10, v_scaleFac, v_array10);
		v_array11 = FMAINTRINAVXDP(v_array11, v_scaleFac, v_array11);
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);
	_mm256_store_pd(&(array[32]), v_array8);
	_mm256_store_pd(&(array[36]), v_array9);
	_mm256_store_pd(&(array[40]), v_array10);
	_mm256_store_pd(&(array[44]), v_array11);

	return 12;
}
#endif
