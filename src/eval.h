#include <vector> 
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"
#include "constants.h"
#include "cfr.h"
#include "util.h"

double evaluate(
    void* policy_net,
    int player
);