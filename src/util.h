#ifndef UTIL_H
#define UTIL_H
#include <torch/torch.h>
#include "engine.h"
#include "debug.h"

// Define a structure for Infoset
struct Infoset {
    std::array<torch::Tensor, 4> cards;
    torch::Tensor bet_fracs;
    torch::Tensor bet_status;
};

Infoset prepare_infoset(
    PokerEngine& game,
    int player,
    int max_bets_per_player
);

std::array<double, MAX_ACTIONS> regret_match(const torch::Tensor& logits, int n_acts);

#endif