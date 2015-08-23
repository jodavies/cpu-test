// Prototypes
#include "runParameters.h"

// Addition
int TestLatAddSSESP(float * RESTRICT array, CONST float scaleFac);
int TestLatAddSSEDP(double * RESTRICT array, CONST double scaleFac);
int TestLatAddAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestLatAddAVXDP(double * RESTRICT array, CONST double scaleFac);

// Multiplication
int TestLatMulSSESP(float * RESTRICT array, CONST float scaleFac);
int TestLatMulSSEDP(double * RESTRICT array, CONST double scaleFac);
int TestLatMulAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestLatMulAVXDP(double * RESTRICT array, CONST double scaleFac);

// Division
int TestLatDivSSESP(float * RESTRICT array, CONST float scaleFac);
int TestLatDivSSEDP(double * RESTRICT array, CONST double scaleFac);
int TestLatDivAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestLatDivAVXDP(double * RESTRICT array, CONST double scaleFac);
