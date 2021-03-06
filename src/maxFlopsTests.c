#include <immintrin.h>
#include <x86intrin.h>
#include "maxFlopsTests.h"

// Here we try to achieve the theoretical maximum floating point
// throughput.
//
// Return the number of ops per iteration
//
// For sandy-bridge, ivy-bridge this is achieved by submitting an add
// and a mul each cycle.
//
// For FMA supporting CPUs, (Intel Haswell+, AMD Bulldozer+), this is
// achieved by submitting one or two FMAs per cycle. Correct FMA
// instruction for the architecture set in runParameters.h.
// For Haswell we need 10 independent operations in flight, due to the
// latency of 5 and throughput of 0.5.


// sandy bridge, ivy bridge
// sse
int TestMaxFlopsSSESP(float * RESTRICT array, CONST float scaleFac)
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

	for (int i = 0; i < NTIMES*100; i++) {
		v_array0 = _mm_add_ps(v_array0, _mm_mul_ps(v_array0, v_scaleFac));
		v_array1 = _mm_add_ps(v_array1, _mm_mul_ps(v_array1, v_scaleFac));
		v_array2 = _mm_add_ps(v_array2, _mm_mul_ps(v_array2, v_scaleFac));
		v_array3 = _mm_add_ps(v_array3, _mm_mul_ps(v_array3, v_scaleFac));
		v_array4 = _mm_add_ps(v_array4, _mm_mul_ps(v_array4, v_scaleFac));
		v_array5 = _mm_add_ps(v_array5, _mm_mul_ps(v_array5, v_scaleFac));
		v_array6 = _mm_add_ps(v_array6, _mm_mul_ps(v_array6, v_scaleFac));
		v_array7 = _mm_add_ps(v_array7, _mm_mul_ps(v_array7, v_scaleFac));
	}

	_mm_store_ps(&(array[0]), v_array0);
	_mm_store_ps(&(array[4]), v_array1);
	_mm_store_ps(&(array[8]), v_array2);
	_mm_store_ps(&(array[12]), v_array3);
	_mm_store_ps(&(array[16]), v_array4);
	_mm_store_ps(&(array[20]), v_array5);
	_mm_store_ps(&(array[24]), v_array6);
	_mm_store_ps(&(array[28]), v_array7);

	return 16;
}

int TestMaxFlopsSSEDP(double * RESTRICT array, CONST double scaleFac)
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

	for (int i = 0; i < NTIMES*100; i++) {
		v_array0 = _mm_add_pd(v_array0, _mm_mul_pd(v_array0, v_scaleFac));
		v_array1 = _mm_add_pd(v_array1, _mm_mul_pd(v_array1, v_scaleFac));
		v_array2 = _mm_add_pd(v_array2, _mm_mul_pd(v_array2, v_scaleFac));
		v_array3 = _mm_add_pd(v_array3, _mm_mul_pd(v_array3, v_scaleFac));
		v_array4 = _mm_add_pd(v_array4, _mm_mul_pd(v_array4, v_scaleFac));
		v_array5 = _mm_add_pd(v_array5, _mm_mul_pd(v_array5, v_scaleFac));
		v_array6 = _mm_add_pd(v_array6, _mm_mul_pd(v_array6, v_scaleFac));
		v_array7 = _mm_add_pd(v_array7, _mm_mul_pd(v_array7, v_scaleFac));
	}

	_mm_store_pd(&(array[0]), v_array0);
	_mm_store_pd(&(array[2]), v_array1);
	_mm_store_pd(&(array[4]), v_array2);
	_mm_store_pd(&(array[6]), v_array3);
	_mm_store_pd(&(array[8]), v_array4);
	_mm_store_pd(&(array[10]), v_array5);
	_mm_store_pd(&(array[12]), v_array6);
	_mm_store_pd(&(array[14]), v_array7);

	return 16;
}
// avx
#ifdef WITHAVX
int TestMaxFlopsAVXSP(float * RESTRICT array, CONST float scaleFac)
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

	for (int i = 0; i < NTIMES*100; i++) {
		v_array0 = _mm256_add_ps(v_array0, _mm256_mul_ps(v_array0, v_scaleFac));
		v_array1 = _mm256_add_ps(v_array1, _mm256_mul_ps(v_array1, v_scaleFac));
		v_array2 = _mm256_add_ps(v_array2, _mm256_mul_ps(v_array2, v_scaleFac));
		v_array3 = _mm256_add_ps(v_array3, _mm256_mul_ps(v_array3, v_scaleFac));
		v_array4 = _mm256_add_ps(v_array4, _mm256_mul_ps(v_array4, v_scaleFac));
		v_array5 = _mm256_add_ps(v_array5, _mm256_mul_ps(v_array5, v_scaleFac));
		v_array6 = _mm256_add_ps(v_array6, _mm256_mul_ps(v_array6, v_scaleFac));
		v_array7 = _mm256_add_ps(v_array7, _mm256_mul_ps(v_array7, v_scaleFac));
	}

	_mm256_store_ps(&(array[0]), v_array0);
	_mm256_store_ps(&(array[8]), v_array1);
	_mm256_store_ps(&(array[16]), v_array2);
	_mm256_store_ps(&(array[24]), v_array3);
	_mm256_store_ps(&(array[32]), v_array4);
	_mm256_store_ps(&(array[40]), v_array5);
	_mm256_store_ps(&(array[48]), v_array6);
	_mm256_store_ps(&(array[56]), v_array7);

	return 16;
}

int TestMaxFlopsAVXDP(double * RESTRICT array, CONST double scaleFac)
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

	for (int i = 0; i < NTIMES*100; i++) {
		v_array0 = _mm256_add_pd(v_array0, _mm256_mul_pd(v_array0, v_scaleFac));
		v_array1 = _mm256_add_pd(v_array1, _mm256_mul_pd(v_array1, v_scaleFac));
		v_array2 = _mm256_add_pd(v_array2, _mm256_mul_pd(v_array2, v_scaleFac));
		v_array3 = _mm256_add_pd(v_array3, _mm256_mul_pd(v_array3, v_scaleFac));
		v_array4 = _mm256_add_pd(v_array4, _mm256_mul_pd(v_array4, v_scaleFac));
		v_array5 = _mm256_add_pd(v_array5, _mm256_mul_pd(v_array5, v_scaleFac));
		v_array6 = _mm256_add_pd(v_array6, _mm256_mul_pd(v_array6, v_scaleFac));
		v_array7 = _mm256_add_pd(v_array7, _mm256_mul_pd(v_array7, v_scaleFac));
	}

	_mm256_store_pd(&(array[0]), v_array0);
	_mm256_store_pd(&(array[4]), v_array1);
	_mm256_store_pd(&(array[8]), v_array2);
	_mm256_store_pd(&(array[12]), v_array3);
	_mm256_store_pd(&(array[16]), v_array4);
	_mm256_store_pd(&(array[20]), v_array5);
	_mm256_store_pd(&(array[24]), v_array6);
	_mm256_store_pd(&(array[28]), v_array7);

	return 16;
}

#ifdef WITHFMA
int TestMaxFlopsFMASP(float * RESTRICT array, CONST float scaleFac)
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

	for (int i = 0; i < NTIMES*100; i++) {
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

	return 20;
}

int TestMaxFlopsFMADP(double * RESTRICT array, CONST double scaleFac)
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

	for (int i = 0; i < NTIMES*100; i++) {
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

	return 20;
}

#endif
#endif
