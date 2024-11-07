#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/script.h>
#include "model/model.h"
#include <array>

namespace py = pybind11;

torch::Tensor init_batched_hands(int BS) {
    std::vector<int64_t> hand_shape = {BS, 2};
    auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
    return torch::zeros(hand_shape, options).contiguous();
}

torch::Tensor init_batched_flops(int BS) {
    std::vector<int64_t> flop_shape = {BS, 3};
    auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
    return torch::zeros(flop_shape, options).contiguous();
}

torch::Tensor init_batched_turns(int BS) {
    std::vector<int64_t> turn_shape = {BS, 2};
    auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
    return torch::zeros(turn_shape, options).contiguous();
}

torch::Tensor init_batched_rivers(int BS) {
    std::vector<int64_t> river_shape = {BS, 2};
    auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
    return torch::zeros(river_shape, options).contiguous();
}

torch::Tensor init_batched_fracs(int BS, int size) {
    std::vector<int64_t> batched_fracs_shape = {BS, size};
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(false);
    return torch::zeros(batched_fracs_shape, options).contiguous();
}

torch::Tensor init_batched_status(int BS, int size) {
    std::vector<int64_t> batched_status_shape = {BS, size};
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(false);
    return torch::zeros(batched_status_shape, options).contiguous();
}

torch::Tensor regret_match(const torch::Tensor& batched_logits) {
    // Apply ReLU to ensure non-negative logits
    auto relu_logits = torch::relu(batched_logits); // [batch_size, num_actions]
    auto logits_sum = relu_logits.sum(1, true); // [batch_size, 1]
    auto positive_mask = logits_sum.squeeze(1) > 0; // [batch_size]
    auto strategy = torch::zeros_like(batched_logits); // [batch_size, num_actions]

    // Handle positive sums
    if (positive_mask.any().item<bool>()) {
        auto positive_indices = torch::nonzero(positive_mask).squeeze(1); // [num_positive]
        auto positive_logits = relu_logits.index_select(0, positive_indices); // [num_positive, num_actions]
        auto positive_logits_sum = logits_sum.index_select(0, positive_indices); // [num_positive, 1]
        auto positive_strategy = positive_logits / positive_logits_sum; // [num_positive, num_actions]
        strategy.index_copy_(0, positive_indices, positive_strategy);
    }

    // Handle non-positive sums
    auto negative_mask = ~positive_mask; // [batch_size]
    if (negative_mask.any().item<bool>()) {
        auto negative_indices = torch::nonzero(negative_mask).squeeze(1); // [num_negative]
        auto negative_logits = relu_logits.index_select(0, negative_indices); // [num_negative, num_actions]
        auto max_indices = torch::argmax(negative_logits, 1); // [num_negative]
        auto one_hot = torch::zeros({negative_indices.size(0), batched_logits.size(1)}, batched_logits.options());
        one_hot.scatter_(1, max_indices.unsqueeze(1), 1);
        strategy.index_copy_(0, negative_indices, one_hot);
    }

    return strategy;
}
std::vector<float> forward(
    const std::string& model_path,
    const std::vector<int>& hand, 
    const std::vector<int>& flops,
    int turn,
    int river,
    const std::vector<float>& fracs,
    const std::vector<int>& status
) {
    int total_round_size = fracs.size();
    auto b_hands = init_batched_hands(1);
    auto b_flops = init_batched_flops(1);
    auto b_turns = init_batched_turns(1);
    auto b_rivers = init_batched_rivers(1);
    auto b_fracs = init_batched_fracs(1, total_round_size);
    auto b_status = init_batched_status(1, total_round_size);
        
    int batch = 0;
    // Get accessors
    auto hand_a = b_hands.accessor<int32_t, 2>();
    auto flops_a = b_flops.accessor<int32_t, 2>();
    auto turn_a = b_turns.accessor<int32_t, 2>();
    auto river_a = b_rivers.accessor<int32_t, 2>();
    auto bet_fracs_a = b_fracs.accessor<float, 2>();
    auto bet_status_a = b_status.accessor<float, 2>();

    // Update hand cards (first two cards)
    for (int i = 0; i < 2; ++i) {
        hand_a[batch][i] = hand[i];
    }

    // Update flop cards (next three cards)
    for (int i = 0; i < 3; ++i) {
        flops_a[batch][i] = flops[i];
    }

    // Update turn card
    turn_a[batch][0] = turn;

    // Update river card
    river_a[batch][0] = river;

    // Update bet fractions
    for (int i = 0; i < total_round_size; ++i) {
        bet_fracs_a[batch][i] = fracs[i];
    }

    // Update bet status
    for (int i = 0; i < total_round_size; ++i) {
        bet_status_a[batch][i] = status[i];
    }

    DeepCFRModel model;
    torch::Tensor logits;

    try {
        // Try loading and inferencing on CPU first
        torch::load(model, model_path);
        logits = model->forward(b_hands, b_flops, b_turns, b_rivers, b_fracs, b_status);
    } catch (const c10::Error& e) {
        // If CPU loading fails, try GPU
        torch::load(model, model_path, torch::kCUDA);
        // Move input tensors to GPU
        b_hands = b_hands.to(torch::kCUDA);
        b_flops = b_flops.to(torch::kCUDA);
        b_turns = b_turns.to(torch::kCUDA);
        b_rivers = b_rivers.to(torch::kCUDA);
        b_fracs = b_fracs.to(torch::kCUDA);
        b_status = b_status.to(torch::kCUDA);
        
        logits = model->forward(b_hands, b_flops, b_turns, b_rivers, b_fracs, b_status);
        logits = logits.to(torch::kCPU);  // Move result back to CPU
    }

    // Convert the logits tensor to a std::vector<float>
    std::vector<float> logits_values(logits.size(1));
    auto logits_accessor = logits.accessor<float, 2>();
    for (int i = 0; i < logits.size(1); ++i) {
        logits_values[i] = logits_accessor[0][i];
    }
    
    return logits_values;
}

PYBIND11_MODULE(poker_inference, m) {
    m.doc() = "Python bindings for libtorch inference";
    m.def("forward", &forward, "Get forward logits",
          py::arg("model_path"),
          py::arg("hand"),
          py::arg("flops"),
          py::arg("turn"),
          py::arg("river"),
          py::arg("fracs"),
          py::arg("status"));
}