// Prototypes
#include "runParameters.h"

// Addition
int TestThrAddSSESP(float * RESTRICT array, CONST float scaleFac);
int TestThrAddSSEDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHAVX
int TestThrAddAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestThrAddAVXDP(double * RESTRICT array, CONST double scaleFac);
#endif

// Multiplication
int TestThrMulSSESP(float * RESTRICT array, CONST float scaleFac);
int TestThrMulSSEDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHAVX
int TestThrMulAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestThrMulAVXDP(double * RESTRICT array, CONST double scaleFac);
#endif

// Division
int TestThrDivSSESP(float * RESTRICT array, CONST float scaleFac);
int TestThrDivSSEDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHAVX
int TestThrDivAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestThrDivAVXDP(double * RESTRICT array, CONST double scaleFac);
#endif

// FMA
#ifdef WITHFMA
int TestThrFMAAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestThrFMAAVXDP(double * RESTRICT array, CONST double scaleFac);
#endif
