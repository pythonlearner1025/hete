#include <vector> 
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"
#include "constants.h"
#include "cfr.h"
#include "util.h"

double evaluate(
    DeepCFRModel policy_net,
    int player
);