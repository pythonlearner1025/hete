// cfr.h

#ifndef CFR_H
#define CFR_H

#include <vector>
#include <mutex>
#include <tuple>
#include <torch/torch.h>
#include "engine.h"
#include "model/model.h"

// Define a structure for Infoset
struct Infoset {
    std::array<torch::Tensor, 4> cards;
    torch::Tensor bet_fracs;
    torch::Tensor bet_status;
};

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
    std::vector<std::vector<void*>>& nets,
    int t,
    int max_bets_per_player,
    std::vector<TraverseAdvantage>& traverse_advs,
    std::mutex& advs_mutex
);

// Function to run multiple traversals in parallel
void run_traversals(
    int num_traversals,
    int player,
    std::vector<std::vector<void*>>& nets,
    int t,
    int max_bets_per_player,
    std::vector<TraverseAdvantage>& all_traverse_advs,
    std::mutex& advs_mutex
);

#endif // CFR_H
