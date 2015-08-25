// Prototypes
#include "runParameters.h"

// Addition
int TestLatAddSSESP(float * RESTRICT array, CONST float scaleFac);
int TestLatAddSSEDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHAVX
int TestLatAddAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestLatAddAVXDP(double * RESTRICT array, CONST double scaleFac);
#endif

// Multiplication
int TestLatMulSSESP(float * RESTRICT array, CONST float scaleFac);
int TestLatMulSSEDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHAVX
int TestLatMulAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestLatMulAVXDP(double * RESTRICT array, CONST double scaleFac);

#endif

// Division
int TestLatDivSSESP(float * RESTRICT array, CONST float scaleFac);
int TestLatDivSSEDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHAVX
int TestLatDivAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestLatDivAVXDP(double * RESTRICT array, CONST double scaleFac);
#endif

// FMA
#ifdef WITHFMA
int TestLatFMAAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestLatFMAAVXDP(double * RESTRICT array, CONST double scaleFac);
#endif
