//TODO: fix freq measurement

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <papi.h>
#include <omp.h>
#include <immintrin.h>

#include "latencyTests.h"
#include "throughputTests.h"
#include "maxFlopsTests.h"


// print per-thread latencies? Maybe something interesting happens with HT?
//#define PERTHREADPRINT 1



void RunLatencyTestsParallel();
void RunThroughputTestsParallel();
void RunMaxFlopsTests(double peak);
void TestFreqs(double * freqs);
void TestMemoryBandwidth();

void CheckPAPIError(int err, int line);

// Compute mean cycles
double ComputeMean(long long * array, int arrayLen);
double ComputeMeanDouble(double * array, int arrayLen);
// Compute stdev of cycles
double ComputeStdev(long long * array, int arrayLen);

double GetWallTime();


int main(void)
{
	double * freqs;
	freqs = malloc(omp_get_max_threads() * sizeof *freqs);
	TestFreqs(freqs);

	RunMaxFlopsTests(1.0);

	TestMemoryBandwidth();

	RunLatencyTestsParallel();
	RunThroughputTestsParallel();

	return 0;
}


// Run latency tests on varying numbers of threads. 
void RunLatencyTestsParallel()
{
	printf("\n\nRunning instruction latency tests...\n");

	const int nThreads = omp_get_max_threads();

	// Make arrays to hold results. Print everything at the end.
	// array: [threads-1][instruction][sub-instruction][thread id*NRUNS+run id]
	long long resultsCycles[8][3][4][8*NRUNS];
	// To index result array
	const int add = 0; const int mul = 1; const int div = 2;
	const int ps = 0; const int pd = 1; const int vps = 2; const int vpd = 3;

	for (int threads = 1; threads <= nThreads; threads++) {

		// Start parallel region
		omp_set_num_threads(threads);
		#pragma omp parallel default(none) shared(resultsCycles, threads)
		{

			// Allocate and initialize memory for this thread
			// (32byte aligned) allocation of arrays for tests.
			// Single vector-width arrays for SP,DP, (AVX and SSE)
			float *arraySSESP, *arrayAVXSP, arraySumSP = 0.0f;
			double *arraySSEDP, *arrayAVXDP, arraySumDP = 0.0;
			arraySSESP = _mm_malloc(VECWIDTHSSESP * sizeof *arraySSESP, 32);
			arraySSEDP = _mm_malloc(VECWIDTHSSEDP * sizeof *arraySSEDP, 32);
			arrayAVXSP = _mm_malloc(VECWIDTHAVXSP * sizeof *arrayAVXSP, 32);
			arrayAVXDP = _mm_malloc(VECWIDTHAVXDP * sizeof *arrayAVXDP, 32);
			// initialize
			for (int j = 0; j < VECWIDTHSSESP; j++) arraySSESP[j] = j;
			for (int j = 0; j < VECWIDTHSSEDP; j++) arraySSEDP[j] = j;
			for (int j = 0; j < VECWIDTHAVXSP; j++) arrayAVXSP[j] = j;
			for (int j = 0; j < VECWIDTHAVXDP; j++) arrayAVXDP[j] = j;

			// Set up PAPI counters:
			int counters[2] = {PAPI_TOT_CYC};
			const int numCounters = 1;
			long long counterValues[1];

			// Warm-Up run, makes the first test accurate. Takes a few 100? cycles to ramp up 
			// clock speed and start up vector units
			TestLatMulSSESP(arraySSESP, 1.0000001);
			TestLatMulSSESP(arraySSESP, 1.0000001);

			// Start PAPI counters
			CheckPAPIError(PAPI_start_counters(counters, numCounters), __LINE__);

			// Ensure all threads are running the test at the same time, by putting barrier before starting each test.

			// // // Addition tests // // //
			// addps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatAddSSESP(arraySSESP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][ps][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSESP; j++) arraySumSP += arraySSESP[j];
			
			// addpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatAddSSEDP(arraySSEDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][pd][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSEDP; j++) arraySumDP += arraySSEDP[j];
			
			// vaddps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatAddAVXSP(arrayAVXSP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][vps][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXSP; j++) arraySumSP += arrayAVXSP[j];
			
			// vaddpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatAddAVXDP(arrayAVXDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][vpd][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXDP; j++) arraySumDP += arrayAVXDP[j];



			// // // Multiplication tests // // //
			// mulps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatMulSSESP(arraySSESP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][ps][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSESP; j++) arraySumSP += arraySSESP[j];
			
			// mulpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatMulSSEDP(arraySSEDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][pd][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSEDP; j++) arraySumDP += arraySSEDP[j];
			
			// vmulps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatMulAVXSP(arrayAVXSP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][vps][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXSP; j++) arraySumSP += arrayAVXSP[j];
			
			// vmulpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatMulAVXDP(arrayAVXDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][vpd][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXDP; j++) arraySumDP += arrayAVXDP[j];



			// // // Division tests // // //
			// divps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatDivSSESP(arraySSESP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][ps][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSESP; j++) arraySumSP += arraySSESP[j];
			
			// divpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatDivSSEDP(arraySSEDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][pd][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSEDP; j++) arraySumDP += arraySSEDP[j];
			
			// vdivps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatDivAVXSP(arrayAVXSP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][vps][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXSP; j++) arraySumSP += arrayAVXSP[j];
			
			// vdivpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				TestLatDivAVXDP(arrayAVXDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][vpd][omp_get_thread_num()*NRUNS+i] = counterValues[0];
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXDP; j++) arraySumDP += arrayAVXDP[j];


			CheckPAPIError(PAPI_stop_counters(counterValues, numCounters), __LINE__);

			_mm_free(arraySSESP);
			_mm_free(arraySSEDP);
			_mm_free(arrayAVXSP);
			_mm_free(arrayAVXDP);

		} // end omp parallel region

	} // end loop over number of threads


	// Now print results:
	printf("-----------------------------------------------------------------\n");
	printf("| Instruction | Tot Threads | Thread No |   Latency |     stdev |\n");
	printf("-----------------------------------------------------------------\n");
	for (int threads = 1; threads <= nThreads; threads++) {

#ifdef PERTHREADPRINT
		// Print a separate mean and stdev for each participating thread
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "addps", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][ps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][ps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "addpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][pd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][pd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vaddps", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][vps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vaddpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "mulps", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][ps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][ps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "mulpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][pd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][pd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vmulps", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][vps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vmulpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "divps", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][ps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][ps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "divpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][pd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][pd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vdivps", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][vps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vdivpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES);
#else
		// Compute mean and stdev between all participating threads
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "addps", threads, ComputeMean(&(resultsCycles[threads-1][add][ps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][ps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "addpd", threads, ComputeMean(&(resultsCycles[threads-1][add][pd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][pd][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vaddps", threads, ComputeMean(&(resultsCycles[threads-1][add][vps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vaddpd", threads, ComputeMean(&(resultsCycles[threads-1][add][vpd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vpd][0]), threads*NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "mulps", threads, ComputeMean(&(resultsCycles[threads-1][mul][ps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][ps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "mulpd", threads, ComputeMean(&(resultsCycles[threads-1][mul][pd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][pd][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vmulps", threads, ComputeMean(&(resultsCycles[threads-1][mul][vps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vmulpd", threads, ComputeMean(&(resultsCycles[threads-1][mul][vpd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vpd][0]), threads*NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "divps", threads, ComputeMean(&(resultsCycles[threads-1][div][ps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][ps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "divpd", threads, ComputeMean(&(resultsCycles[threads-1][div][pd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][pd][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vdivps", threads, ComputeMean(&(resultsCycles[threads-1][div][vps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vdivpd", threads, ComputeMean(&(resultsCycles[threads-1][div][vpd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vpd][0]), threads*NRUNS)/(double)NTIMES);
#endif
		printf("-----------------------------------------------------------------\n");
	}


}



// Run throughput tests on varying numbers of threads. 
void RunThroughputTestsParallel()
{
	printf("\n\nRunning instruction throughput tests...\n");

	const int nThreads = omp_get_max_threads();

	// Make arrays to hold results. Print everything at the end.
	// array: [threads-1][instruction][sub-instruction][thread id*NRUNS+run id]
	long long resultsCycles[8][3][4][8*NRUNS];
	// To index result array
	const int add = 0; const int mul = 1; const int div = 2;
	const int ps = 0; const int pd = 1; const int vps = 2; const int vpd = 3;

	for (int threads = 1; threads <= nThreads; threads++) {

		// Start parallel region
		omp_set_num_threads(threads);
		#pragma omp parallel default(none) shared(resultsCycles, threads)
		{

			// Allocate and initialize memory for this thread
			// 16 vector-width arrays for SP,DP, (AVX and SSE), since
			// this is (for now...) the number of vector registers and
			// so the most independent instructions that will be queued
			// up in the tests
			float *arraySSESP, *arrayAVXSP, arraySumSP = 0.0f;
			double *arraySSEDP, *arrayAVXDP, arraySumDP = 0.0;
			arraySSESP = _mm_malloc(16 * VECWIDTHSSESP * sizeof *arraySSESP, 32);
			arraySSEDP = _mm_malloc(16 * VECWIDTHSSEDP * sizeof *arraySSEDP, 32);
			arrayAVXSP = _mm_malloc(16 * VECWIDTHAVXSP * sizeof *arrayAVXSP, 32);
			arrayAVXDP = _mm_malloc(16 * VECWIDTHAVXDP * sizeof *arrayAVXDP, 32);
			// initialize
			for (int j = 0; j < 16 * VECWIDTHSSESP; j++) arraySSESP[j] = j;
			for (int j = 0; j < 16 * VECWIDTHSSEDP; j++) arraySSEDP[j] = j;
			for (int j = 0; j < 16 * VECWIDTHAVXSP; j++) arrayAVXSP[j] = j;
			for (int j = 0; j < 16 * VECWIDTHAVXDP; j++) arrayAVXDP[j] = j;

			// Set up PAPI counters:
			int counters[1] = {PAPI_TOT_CYC};
			const int numCounters = 1;
			long long counterValues[1];

			// Test functions return the number of instructions that they ran (per NTIMES test)
			int instrRun;

			// Warm-Up run, makes the first test accurate. Takes a few 100? cycles to ramp up 
			// clock speed and start up vector units
			TestThrMulSSESP(arraySSESP, 1.0000001);
			TestThrMulSSESP(arraySSESP, 1.0000001);

			// Start PAPI counters
			CheckPAPIError(PAPI_start_counters(counters, numCounters), __LINE__);

			// Ensure all threads are running the test at the same time, by putting barrier before starting each test.

			// // // Addition tests // // //
			// addps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrAddSSESP(arraySSESP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][ps][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSESP; j++) arraySumSP += arraySSESP[j];
			
			// addpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrAddSSEDP(arraySSEDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][pd][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSEDP; j++) arraySumDP += arraySSEDP[j];
			
			// vaddps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrAddAVXSP(arrayAVXSP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][vps][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXSP; j++) arraySumSP += arrayAVXSP[j];
			
			// vaddpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrAddAVXDP(arrayAVXDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][add][vpd][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXDP; j++) arraySumDP += arrayAVXDP[j];



			// // // Multiplication tests // // //
			// mulps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrMulSSESP(arraySSESP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][ps][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSESP; j++) arraySumSP += arraySSESP[j];
			
			// mulpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrMulSSEDP(arraySSEDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][pd][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSEDP; j++) arraySumDP += arraySSEDP[j];
			
			// vmulps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrMulAVXSP(arrayAVXSP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][vps][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXSP; j++) arraySumSP += arrayAVXSP[j];
			
			// vmulpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrMulAVXDP(arrayAVXDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][mul][vpd][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXDP; j++) arraySumDP += arrayAVXDP[j];



			// // // Division tests // // //
			// divps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrDivSSESP(arraySSESP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][ps][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSESP; j++) arraySumSP += arraySSESP[j];
			
			// divpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrDivSSEDP(arraySSEDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][pd][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHSSEDP; j++) arraySumDP += arraySSEDP[j];
			
			// vdivps
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrDivAVXSP(arrayAVXSP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][vps][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXSP; j++) arraySumSP += arrayAVXSP[j];
			
			// vdivpd
			#pragma omp barrier
			CheckPAPIError(PAPI_read_counters(counterValues, numCounters), __LINE__);
			for (int i = 0; i < NRUNS; i ++) {
				instrRun = TestThrDivAVXDP(arrayAVXDP, 1.0000001f);
				PAPI_read_counters(counterValues, numCounters);
				resultsCycles[threads-1][div][vpd][omp_get_thread_num()*NRUNS+i] = counterValues[0]/instrRun;
			}
			// use result to ensure compiler doesn't optimize out
			for (int j = 0; j < VECWIDTHAVXDP; j++) arraySumDP += arrayAVXDP[j];


			CheckPAPIError(PAPI_stop_counters(counterValues, numCounters), __LINE__);

			_mm_free(arraySSESP);
			_mm_free(arraySSEDP);
			_mm_free(arrayAVXSP);
			_mm_free(arrayAVXDP);

		} // end omp parallel region

	} // end loop over number of threads


	// Now print results:
	printf("-----------------------------------------------------------------\n");
	printf("| Instruction | Tot Threads | Thread No |Throughput |     stdev |\n");
	printf("-----------------------------------------------------------------\n");
	for (int threads = 1; threads <= nThreads; threads++) {

#ifdef PERTHREADPRINT
		// Print a separate mean and stdev for each participating thread
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "addps", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][ps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][ps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "addpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][pd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][pd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vaddps", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][vps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vaddpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][add][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "mulps", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][ps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][ps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "mulpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][pd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][pd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vmulps", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][vps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vmulpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][mul][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "divps", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][ps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][ps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "divpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][pd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][pd][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vdivps", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][vps][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vps][tid*NRUNS]), NRUNS)/(double)NTIMES);
		for (int tid = 0; tid < threads; tid++) printf("| %11s | %11d | %9d | %9.3lf | %9.3lf |\n", "vdivpd", threads, tid, ComputeMean(&(resultsCycles[threads-1][div][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vpd][tid*NRUNS]), NRUNS)/(double)NTIMES);
#else
		// Compute mean and stdev between all participating threads
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "addps", threads, ComputeMean(&(resultsCycles[threads-1][add][ps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][ps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "addpd", threads, ComputeMean(&(resultsCycles[threads-1][add][pd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][pd][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vaddps", threads, ComputeMean(&(resultsCycles[threads-1][add][vps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vaddpd", threads, ComputeMean(&(resultsCycles[threads-1][add][vpd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][add][vpd][0]), threads*NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "mulps", threads, ComputeMean(&(resultsCycles[threads-1][mul][ps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][ps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "mulpd", threads, ComputeMean(&(resultsCycles[threads-1][mul][pd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][pd][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vmulps", threads, ComputeMean(&(resultsCycles[threads-1][mul][vps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vmulpd", threads, ComputeMean(&(resultsCycles[threads-1][mul][vpd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][mul][vpd][0]), threads*NRUNS)/(double)NTIMES);
		printf("-----------------------------------------------------------------\n");
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "divps", threads, ComputeMean(&(resultsCycles[threads-1][div][ps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][ps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "divpd", threads, ComputeMean(&(resultsCycles[threads-1][div][pd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][pd][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vdivps", threads, ComputeMean(&(resultsCycles[threads-1][div][vps][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vps][0]), threads*NRUNS)/(double)NTIMES);
		printf("| %11s | %11d |  combined | %9.3lf | %9.3lf |\n", "vdivpd", threads, ComputeMean(&(resultsCycles[threads-1][div][vpd][0]), threads*NRUNS)/(double)NTIMES, ComputeStdev(&(resultsCycles[threads-1][div][vpd][0]), threads*NRUNS)/(double)NTIMES);
#endif
		printf("-----------------------------------------------------------------\n");
	}


}



void RunMaxFlopsTests(double peak)
{
	printf("\n\nRunning Peak FLOPS tests (single thread) ...\n");
	printf("!! These tests do not currently achieve peak FLOPS on CPUs which support FMA !!\n");

	// (32byte aligned) allocation of arrays for tests.
	// 16 vector-width arrays for SP,DP, (AVX and SSE), since
	// this is (for now...) the number of vector registers and
	// so the most independent instructions that will be queued
	// up in the tests
	float *arraySSESP, *arrayAVXSP, arraySumSP = 0.0f;
	double *arraySSEDP, *arrayAVXDP, arraySumDP = 0.0;
	arraySSESP = _mm_malloc(16 * VECWIDTHSSESP * sizeof *arraySSESP, 32);
	arraySSEDP = _mm_malloc(16 * VECWIDTHSSEDP * sizeof *arraySSEDP, 32);
	arrayAVXSP = _mm_malloc(16 * VECWIDTHAVXSP * sizeof *arrayAVXSP, 32);
	arrayAVXDP = _mm_malloc(16 * VECWIDTHAVXDP * sizeof *arrayAVXDP, 32);
	// initialize
	for (int j = 0; j < 16 * VECWIDTHSSESP; j++) arraySSESP[j] = j;
	for (int j = 0; j < 16 * VECWIDTHSSEDP; j++) arraySSEDP[j] = j;
	for (int j = 0; j < 16 * VECWIDTHAVXSP; j++) arrayAVXSP[j] = j;
	for (int j = 0; j < 16 * VECWIDTHAVXDP; j++) arrayAVXDP[j] = j;

	// To hold results:
	double runTimes[NRUNS];

	// Warm-Up run
	TestMaxFlopsAVXSP(arrayAVXSP, 1.0000001f);

	int instrRun;

	for (int i = 0; i < NRUNS; i++) {
		double timeStart = GetWallTime();
		instrRun = TestMaxFlopsAVXSP(arrayAVXSP, 1.0000001f);
		runTimes[i] = GetWallTime() - timeStart;
	}
	for (int j = 0; j < instrRun * VECWIDTHAVXSP; j++) arraySumSP += arrayAVXSP[j];
	printf("Single Precision Peak GFLOPS: %.3lf\n", (double)instrRun*VECWIDTHAVXSP*NTIMES*100.0/ComputeMeanDouble(runTimes, NRUNS)/1.0e9);


	for (int i = 0; i < NRUNS; i++) {
		double timeStart = GetWallTime();
		instrRun = TestMaxFlopsAVXDP(arrayAVXDP, 1.0000001f);
		runTimes[i] = GetWallTime() - timeStart;
	}
	for (int j = 0; j < instrRun * VECWIDTHAVXSP; j++) arraySumDP += arrayAVXDP[j];
	printf("Double Precision Peak GFLOPS: %.3lf\n", (double)instrRun*VECWIDTHAVXDP*NTIMES*100.0/ComputeMeanDouble(runTimes, NRUNS)/1.0e9);





}



// Test the load frequencies of the processor, for load on 1 to omp_get_max_threads().
void TestFreqs(double *freqs)
{
	printf("\n\nMeasuring loaded CPU frequencies...\n");

	// Check maximum number of threads
	int maxThreads = omp_get_max_threads();

	// For each number of threads, collect total elapsed cycles using PAPI,
	// and also the elapsed time, measured outside of the parallel region.
	double *totalCycles;
	totalCycles = malloc(maxThreads * sizeof *totalCycles);
	double *runTimes;
	runTimes = malloc(maxThreads * sizeof *runTimes);

	// For each possible thread count:
	for (int threads = 1; threads <= maxThreads; threads++) {

		// allocate an array for PAPI counter results
		long long *cycles;
		cycles = malloc(threads * sizeof *cycles);

		// allocate an array to use to provide load
		double *array;
		array = _mm_malloc(32 * threads * sizeof *array, 32);


		// Start openmp team. Disable dynamic threads so we are sure of the number.
		omp_set_dynamic(0);
		omp_set_num_threads(threads);

		// warm-up CPU
		#pragma omp parallel default(none) shared(array)
		{
			int tid = omp_get_thread_num();
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
		}

		// Start timing
		double time = GetWallTime();

		#pragma omp parallel default(none) shared(array, cycles, threads)
		{
			int tid = omp_get_thread_num();

			// Start PAPI counters
			int counters[1] = {PAPI_TOT_CYC};
			int numCounters = 1;
			CheckPAPIError(PAPI_start_counters(counters, numCounters), __LINE__);

			// Load CPU
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);
			TestMaxFlopsAVXDP(&(array[tid*32]), 3.0);

			// Stop PAPI counters
			CheckPAPIError(PAPI_stop_counters(&(cycles[omp_get_thread_num()]), numCounters), __LINE__);
		}

		// Stop timer
		time = GetWallTime() - time;

		totalCycles[threads-1] = 0;
		for (int i = 0; i < threads; i++) {
			totalCycles[threads-1] += cycles[i]/(double)threads;
		}

		printf("Threads: %d, freq: %.2lf GHz (%.0lf cyc, %.3lf s)\n", threads, totalCycles[threads-1]/time/1.0e9, totalCycles[threads-1], time);
		freqs[threads-1] = totalCycles[threads-1]/time/1.0e9;
	
		// Free allocated memory
		free(cycles);
		_mm_free(array);
	}

}


void TestMemoryBandwidth()
{
	int nRuns = 1000;

	printf("\n\nCache and Memory Bandwidth test:\n");
	// Run loops with different numbers of tests, depending on array size. Smaller arrays
	// produce highly variable results so need a larger number of tests.
	for (int i = pow(2,8); i <= pow(2,12); i *= 2) {
		// allocate array for testing
		double *array;
		array = _mm_malloc(i * sizeof *array, 32);
		for (int j = 0; j < i; j++) {
			array[j] = (double)j;
		}
		// start timer
		double time = GetWallTime();
		// increment the array values, NTIMES
		for (int k = 0; k < nRuns*10000; k++) {
			for (int j = 0; j < i; j++) {
				array[j]+=1.0;
			}
		}
		// Stop timer
		time = GetWallTime() - time;
		printf("Size: %6.0lf KB, time: %2.3lfs, bandwidth: %.2lfGB/s\n",
		       i*(sizeof *array)/1024.0, time, 2.0*nRuns*10000*i*(double)(sizeof *array)/time/1024.0/1024.0/1024.0);
		_mm_free(array);
	}
	for (int i = pow(2,13); i <= pow(2,15); i *= 2) {
		// allocate array for testing
		double *array;
		array = _mm_malloc(i * sizeof *array, 32);
		for (int j = 0; j < i; j++) {
			array[j] = (double)j;
		}
		// start timer
		double time = GetWallTime();
		// increment the array values, NTIMES
		for (int k = 0; k < nRuns*100; k++) {
			for (int j = 0; j < i; j++) {
				array[j]+=1.0;
			}
		}
		// Stop timer
		time = GetWallTime() - time;
		printf("Size: %6.0lf KB, time: %2.3lfs, bandwidth: %.2lfGB/s\n",
		       i*(sizeof *array)/1024.0, time, 2.0*nRuns*100*i*(double)(sizeof *array)/time/1024.0/1024.0/1024.0);
		_mm_free(array);
	}
	for (int i = pow(2,16); i <= pow(2,20); i *= 2) {
		// allocate array for testing
		double *array;
		array = _mm_malloc(i * sizeof *array, 32);
		for (int j = 0; j < i; j++) {
			array[j] = (double)j;
		}
		// start timer
		double time = GetWallTime();
		// increment the array values, NTIMES
		for (int k = 0; k < nRuns*10; k++) {
			for (int j = 0; j < i; j++) {
				array[j]+=1.0;
			}
		}
		// Stop timer
		time = GetWallTime() - time;
		printf("Size: %6.0lf KB, time: %2.3lfs, bandwidth: %.2lfGB/s\n",
		       i*(sizeof *array)/1024.0, time, 2.0*nRuns*10*i*(double)(sizeof *array)/time/1024.0/1024.0/1024.0);
		_mm_free(array);
	}
	for (int i = pow(2,21); i <= pow(2,24); i *= 2) {
		// allocate array for testing
		double *array;
		array = _mm_malloc(i * sizeof *array, 32);
		for (int j = 0; j < i; j++) {
			array[j] = (double)j;
		}
		// start timer
		double time = GetWallTime();
		// increment the array values, NTIMES
		for (int k = 0; k < nRuns*1; k++) {
			for (int j = 0; j < i; j++) {
				array[j]+=1.0;
			}
		}
		// Stop timer
		time = GetWallTime() - time;
		printf("Size: %6.0lf KB, time: %2.3lfs, bandwidth: %.2lfGB/s\n",
		       i*(sizeof *array)/1024.0, time, 2.0*nRuns*1*i*(double)(sizeof *array)/time/1024.0/1024.0/1024.0);
		_mm_free(array);
	}
	printf("----------------------------------------------------------\n");
}


void CheckPAPIError(int err, int line)
{
	if (err != PAPI_OK) {
		char *errorName = PAPI_strerror(err);
		fprintf(stderr, "PAPI Error on line %d: %s\n", line, errorName);
	}
}



double ComputeMean(long long * array, int arrayLen)
{
	double mean = 0.0;

	for (int i = 0; i < arrayLen; i++) {
		mean += array[i];
	}

	return mean/(double)arrayLen;
}
double ComputeMeanDouble( double* array, int arrayLen)
{
	double mean = 0.0;

	for (int i = 0; i < arrayLen; i++) {
		mean += array[i];
	}

	return mean/(double)arrayLen;
}



double ComputeStdev(long long * array, int arrayLen)
{
	double mean = ComputeMean(array, arrayLen);
	double stdev = 0.0;

	for (int i = 0; i < arrayLen; i++) {
		stdev += (array[i] - mean)*(array[i] - mean);
	}

	return sqrt(stdev/(double)(arrayLen - 1));
}



double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}
