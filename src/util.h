#ifndef UTIL_H
#define UTIL_H
#include <torch/torch.h>
#include "engine.h"
#include "debug.h"
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

// Define a structure for Infoset
struct Infoset {
    std::array<torch::Tensor, 4> cards;
    torch::Tensor bet_fracs;
    torch::Tensor bet_status;
};

Infoset prepare_infoset(
    PokerEngine& game,
    int player
);

std::array<double, NUM_ACTIONS> regret_match(const torch::Tensor& logits);

template <typename T, std::size_t N>
std::array<T, N> normalize_to_prob_dist(const std::array<T, N>& arr);

template <typename T, std::size_t N>
std::size_t argmax(const std::array<T, N>& arr);

#endif