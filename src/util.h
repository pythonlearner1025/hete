#ifndef UTIL_H
#define UTIL_H
#include <torch/torch.h>
#include "engine.h"
#include "debug.h"
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

struct BetHistory {
    // nx4xMAX_ROUND_BETS where n=NUM_PLAYERS
    std::array<std::array<std::array<double, MAX_ROUND_BETS>, NUM_PLAYERS>, 4> amounts;
    std::array<std::array<std::array<bool, MAX_ROUND_BETS>, NUM_PLAYERS>, 4> status;

    // helper to create pre-flattened torch tensors 
    std::pair<torch::Tensor, torch::Tensor> to_tensors() const {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        
        // create contiguous tensors with right shape
        auto amounts_tensor = torch::zeros({4, NUM_PLAYERS, MAX_ROUND_BETS}, options);
        auto status_tensor = torch::zeros({4, NUM_PLAYERS, MAX_ROUND_BETS}, options);

        // fill tensors (they're contiguous in memory)
        auto amounts_acc = amounts_tensor.accessor<float,3>();
        auto status_acc = status_tensor.accessor<float,3>();
        
        for(int r = 0; r < 4; r++) {
            for(int p = 0; p < NUM_PLAYERS; p++) {
                for(int b = 0; b < MAX_ROUND_BETS; b++) {
                    amounts_acc[r][p][b] = amounts[r][p][b];
                    status_acc[r][p][b] = status[r][p][b];
                }
            }
        }

        // flatten preserving memory layout
        return {amounts_tensor.flatten(), status_tensor.flatten()};
    }
};

BetHistory construct_history(PokerEngine& engine);

struct State {
    std::array<int, 2> hand{};
    std::array<int, 3> flop{};
    std::array<int, 1> turn{};
    std::array<int, 1> river{};
    torch::Tensor bet_fracs{};
    torch::Tensor bet_status{};
};

void update_tensors(
    const State S, 
    torch::Tensor hand, 
    torch::Tensor flop, 
    torch::Tensor turn, 
    torch::Tensor river, 
    torch::Tensor bet_fracs, 
    torch::Tensor bet_status,
    int batch = 0 
);

void get_state(
    PokerEngine& game,
    State* state,
    int player
);

float sample_uniform(); 
std::array<double, NUM_ACTIONS> sample_prob(const torch::Tensor& logits, float beta); 
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
int sample_iter(size_t iter);
void take_action(PokerEngine* engine, int player, int act);
bool verify_action(PokerEngine* engine, int player, int act);

torch::Tensor regret_match_batched(const torch::Tensor& batched_logits);

torch::Tensor init_batched_hands(int BS);
torch::Tensor init_batched_flops(int BS);
torch::Tensor init_batched_turns(int BS);
torch::Tensor init_batched_rivers(int BS);
torch::Tensor init_batched_status(int BS);
torch::Tensor init_batched_fracs(int BS);
torch::Tensor init_batched_advs(int BS);
torch::Tensor init_batched_iters(int BS);
#endif
