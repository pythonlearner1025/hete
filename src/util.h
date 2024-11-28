#ifndef UTIL_H
#define UTIL_H
#include <torch/torch.h>
#include "engine.h"
#include "debug.h"
#include <mlx/mlx.h>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

std::vector<float> get_bets(PokerEngine& engine);

struct State {
    std::vector<float> hands;
    std::vector<float> bets;
};

void get_state(
    PokerEngine& game,
    State* state,
    int player
);
float sample_uniform(); 
std::array<double, NUM_ACTIONS> sample_prob(const mlx::core::array logits); 
std::array<double, NUM_ACTIONS> regret_match(const mlx::core::array logits);

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
bool verify_action(PokerEngine* engine, int player, int act, std::string logfile);
void save_model(std::map<std::string, std::optional<mlx::core::array>> params, const std::string& filepath);
std::map<std::string, std::optional<mlx::core::array>> load_model(const std::string& filepath);
#endif
