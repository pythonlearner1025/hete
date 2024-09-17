#ifndef CFR_H
#define CFR_H

#include <vector>
#include <mutex>
#include <tuple>
#include <torch/torch.h>
#include "engine.h"
#include "model/model.h"
#include "constants.h"
#include "util.h"

// Define a structure for traverse advantages
struct TraverseAdvantage {
    Infoset infoset;
    int iteration;
    std::array<double, MAX_ACTIONS> advantages;
};

// The traverse function
double traverse(
    PokerEngine& engine, 
    int player,
    std::array<std::array<void*, NUM_TRAVERSALS>, NUM_PLAYERS>& nets,
    int t,
    std::vector<TraverseAdvantage>& all_traverse_advs,
    std::atomic<size_t>& all_traverse_advs_index,
    std::mutex& advs_mutex
);

// Function to run multiple traversals in parallel
void run_traversals(
    int player,
    std::array<std::array<void*, NUM_TRAVERSALS>, NUM_PLAYERS>& nets,
    int t,
    int max_bets_per_player,
    std::vector<TraverseAdvantage>& all_traverse_advs,
    std::atomic<size_t>& all_traverse_advs_index,
    std::mutex& advs_mutex
);

#endif // CFR_H

