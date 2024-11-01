#include "util.h"

void update_tensors(
    const State S, 
    torch::Tensor hand, 
    torch::Tensor flop, 
    torch::Tensor turn, 
    torch::Tensor river, 
    torch::Tensor bet_fracs, 
    torch::Tensor bet_status,
    int batch 
) {
    // Get accessors
    auto hand_a = hand.accessor<int32_t, 2>();
    auto flop_a = flop.accessor<int32_t, 2>();
    auto turn_a = turn.accessor<int32_t, 2>();
    auto river_a = river.accessor<int32_t, 2>();
    auto bet_fracs_a = bet_fracs.accessor<float, 2>();
    auto bet_status_a = bet_status.accessor<float, 2>();

    // Update hand cards (first two cards)
    for (int i = 0; i < 2; ++i) {
        hand_a[batch][i] = S.hand[i];
    }

    // Update flop cards (next three cards)
    for (int i = 0; i < 3; ++i) {
        flop_a[batch][i] = S.flop[i];
    }

    // Update turn card
    turn_a[batch][0] = S.turn[0];

    // Update river card
    river_a[batch][0] = S.river[0];

    // Update bet fractions
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_fracs_a[batch][i] = S.bet_fracs[i];
    }

    // Update bet status
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_status_a[batch][i] = S.bet_status[i];
    }
}

int sample_iter(size_t iter) {
    // compute softmax
    double sum;
    for (int i = 1; i <= iter; ++i) {
        sum += exp(i);
    }

    std::vector<double> softmax_iters{};
    softmax_iters.resize(iter);

    for (int i = 1; i <= iter; ++i) {
        softmax_iters[i] = exp(i)/sum;
    }

    // sample
    double r = static_cast<double>(rand()) / RAND_MAX;
    double cumulative = 0.0;
    for (int i = 1; i <= iter; ++i) {
        cumulative += softmax_iters[i];
        if (r <= cumulative) {
            return i;
        }
    }

    return softmax_iters[softmax_iters.size()-1];
}

// Sample an action according to the strategy probabilities
int sample_action(const std::array<double, NUM_ACTIONS>& strat) {
    double r = static_cast<double>(rand()) / RAND_MAX;
    double cumulative = 0.0;
    DEBUG_INFO("r is " << r);
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        DEBUG_INFO("strat " << i << " has p=" << strat[i]);
        cumulative += strat[i];
        if (r <= cumulative) {
            DEBUG_INFO("returning " << i);
            return i;
        }
    }
    DEBUG_INFO("returning " << (NUM_ACTIONS - 1));
    return NUM_ACTIONS - 1; // Return last valid action if none selected
}

void take_action(PokerEngine* engine, int player, int act) {
    DEBUG_INFO("Chosen act: " << act);
    if (act == 0) {
        engine->fold(player);
        return;
    }
    if (act == 1) {
        engine->check_or_call(player); 
        return;
    }
    double inc = engine->get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            engine->bet_or_raise(player, bet_amt);
            return;
        }
        bet_amt += inc;
    }
}

bool verify_action(PokerEngine* engine, int player, int act) {
    if (act == 0) {
        return engine->can_fold(player);
    }
    if (act == 1) {
        return engine->can_check_or_call(player); 
    }
    double inc = engine->get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            return engine->can_bet_or_raise(player, bet_amt);
        }
        bet_amt += inc;
    }
    return false;
}

void get_state(
    PokerEngine& game,
    State* state,
    int player
) {
    auto history = game.construct_history();
    std::array<int, 2> hand = game.players[player].hand;
    std::array<int, 5> board = game.get_board();

    // Copy the bet_status and bet_fracs from history to state
    for (size_t i = 0; i < NUM_PLAYERS * MAX_ROUND_BETS * 4; ++i) {
        state->bet_status[i] = history.first[i];
        state->bet_fracs[i] = history.second[i];
    }

    // Assign hand, flop, turn, and river
    state->hand = hand;
    for (int i = 0; i < 3; ++i) {
        state->flop[i] = board[i];
    }

    state->turn[0] = board[3];
    state->river[0] = board[4];
}

// Must return a probability distribution
std::array<double, NUM_ACTIONS> regret_match(const torch::Tensor& logits) {
    auto relu_logits = torch::relu(logits);
    
    double logits_sum = relu_logits.sum().item<double>();
    
    std::array<double, NUM_ACTIONS> strat{};
    
    // If the sum is positive, calculate the strategy
    if (logits_sum > 0) {
        auto strategy_tensor = relu_logits / logits_sum;
        auto strat_data = strategy_tensor.data_ptr<float>();
        for (int i = 0; i < NUM_ACTIONS; ++i) {
            strat[i] = strat_data[i];
        }
    } 
    // If the sum is zero or negative, return a one-hot vector for the max logit
    else {
        auto max_index = torch::argmax(relu_logits).item<int>();
        std::fill(strat.begin(), strat.end(), 0.0);
        strat[max_index] = 1.0;
    }
    return strat;
}

torch::Tensor regret_match_batched(const torch::Tensor& batched_logits) {
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

torch::Tensor init_batched_fracs(int BS) {
    std::vector<int64_t> batched_fracs_shape = {BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(false);
    return torch::zeros(batched_fracs_shape, options).contiguous();
}

torch::Tensor init_batched_status(int BS) { 
    std::vector<int64_t> batched_status_shape = {BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(false);
    return torch::zeros(batched_status_shape, options).contiguous();
}

torch::Tensor init_batched_advs(int BS) {
    std::vector<int64_t> batched_advs_shape = {BS, NUM_ACTIONS};
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(false);
    return torch::zeros(batched_advs_shape, options).contiguous();
}

torch::Tensor init_batched_iters(int BS) {
    std::vector<int64_t> batched_iters_shape = {BS, 1};
    auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
    return torch::zeros(batched_iters_shape, options).contiguous();
}