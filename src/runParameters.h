// Parameters that control run behaviour
#define NTIMES (1000000)
#define NRUNS 20

// print per-thread latencies, throughputs when running parallel tests.
//#define PERTHREADPRINT 1


// Comment out to remove all AVX functions and instructions, for compiling on non-supporting CPUs
#define WITHAVX 1


// All FMA CPUs also support AVX, and we test with AVX-width vectors...
#ifdef WITHAVX
// Comment out to remove all FMA instructions. Set intrinsic here,
// since it differs for Intel's FMA3 and AMD's FMA4
//#define WITHFMA 1
// FMA3 (intel haswell+)
#define FMAINTRINAVXSP _mm256_fmadd_ps
#define FMAINTRINAVXDP _mm256_fmadd_pd
// FMA4 (amd bulldozer+)
//#define FMAINTRINAVXSP _mm256_macc_ps
//#define FMAINTRINAVXDP _mm256_macc_pd
#endif


// Define a single-core operating frequency in GHz, for a report % of max flops achieved
#define CPUFREQ 4.4


// Minimum thread count. Parallel tests will start with this number of threads, and run up to
// omp_get_max_threads().
#define MINTHREADS 1
// Maximum thread count, this just determines the size of the arrays. TODO: make dynamic?
#define MAXTHREADS 32
// Set how the number of threads increments. Incrementing by 1 each time means some tests take a
// long time, and don't produce any extra interesting data.
//#define THREADINCREMENT ++
#define THREADINCREMENT *=2


// Parameters to set vector lengths for sizes of allocated arrays.
// SSE and AVX, single and double precision:
#define VECWIDTHSSESP 4
#define VECWIDTHSSEDP 2
#define VECWIDTHAVXSP 8
#define VECWIDTHAVXDP 4

#define RESTRICT restrict
#define CONST const

