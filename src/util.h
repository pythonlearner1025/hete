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
std::array<T, N> normalize_to_prob_dist(const std::array<T, N>& arr) {
    T sum = std::accumulate(arr.begin(), arr.end(), static_cast<T>(0));
    std::array<T, N> normalized;
    if (sum > 0) {
        for (std::size_t i = 0; i < N; ++i) {
            normalized[i] = arr[i] / sum;
        }
    } else {
        T uniform_prob = static_cast<T>(1) / static_cast<T>(N);
        normalized.fill(uniform_prob);
    }
    return normalized;
}

template <typename T, std::size_t N>
std::size_t argmax(const std::array<T, N>& arr) {
    return std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));
}

// taking actions
int sample_action(const std::array<double, NUM_ACTIONS>& strat);
void take_action(PokerEngine* engine, int player, int act);
bool verify_action(PokerEngine* engine, int player, int act);
void get_cards(PokerEngine& game, int player, Infoset& I);
torch::Tensor regret_match_batched(const torch::Tensor& batched_logits);
#endif