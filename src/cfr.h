// cfr.h

#ifndef CFR_H
#define CFR_H

#include <vector>
#include <mutex>
#include <tuple>
#include <torch/torch.h>
#include "engine.h"
#include "model/model.h"
#include "constants.h"

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
    std::array<std::array<void*, NUM_TRAVERSALS>, NUM_PLAYERS> nets,
    int t,
    int max_bets_per_player,
    std::array<TraverseAdvantage, MAX_ADVS> all_traverse_advs,
    std::mutex& advs_mutex
);

// Function to run multiple traversals in parallel
void run_traversals(
    int player,
    std::array<std::array<void*, NUM_TRAVERSALS>, NUM_PLAYERS> nets,
    int t,
    int max_bets_per_player,
    std::array<TraverseAdvantage, MAX_ADVS> all_traverse_advs,
    std::mutex& advs_mutex
);

//std::array<double, MAX_ACTIONS> regret_match(const torch::Tensor& logits, int n_acts)

#endif // CFR_H
