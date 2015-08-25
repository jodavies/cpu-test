// Prototypes
#include "runParameters.h"

int TestMaxFlopsSSESP(float * RESTRICT array, CONST float scaleFac);
int TestMaxFlopsSSEDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHAVX
int TestMaxFlopsAVXSP(float * RESTRICT array, CONST float scaleFac);
int TestMaxFlopsAVXDP(double * RESTRICT array, CONST double scaleFac);
#ifdef WITHFMA
int TestMaxFlopsFMASP(float * RESTRICT array, CONST float scaleFac);
int TestMaxFlopsFMADP(double * RESTRICT array, CONST double scaleFac);
#endif
#endif
