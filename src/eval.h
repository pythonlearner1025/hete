#include <vector> 
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"
#include "constants.h"
#include "util.h"
#include "model/model.h"

double evaluate(
    DeepCFRModel policy_net,
    int player
);