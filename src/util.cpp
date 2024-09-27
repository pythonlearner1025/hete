#include "util.h"

void update_tensors(
    const State* S, 
    torch::Tensor* hand, 
    torch::Tensor* flop, 
    torch::Tensor* turn, 
    torch::Tensor* river, 
    torch::Tensor* bet_fracs, 
    torch::Tensor* bet_status,
    int batch = 0
) {
    // Get accessors
    auto hand_a = hand->accessor<int32_t, 2>();
    auto flop_a = flop->accessor<int32_t, 2>();
    auto turn_a = turn->accessor<int32_t, 2>();
    auto river_a = river->accessor<int32_t, 2>();
    auto bet_fracs_a = bet_fracs->accessor<float, 2>();
    auto bet_status_a = bet_status->accessor<float, 2>();

    // Update hand cards (first two cards)
    for (int i = 0; i < 2; ++i) {
        hand_a[batch][i] = S->hand[i];
    }

    // Update flop cards (next three cards)
    for (int i = 0; i < 3; ++i) {
        flop_a[batch][i] = S->flop[i];
    }

    // Update turn card
    turn_a[batch][0] = S->turn[0];

    // Update river card
    river_a[batch][0] = S->river[0];

    // Update bet fractions
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_fracs_a[batch][i] = S->bet_fracs[i];
    }

    // Update bet status
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_status_a[batch][i] = S->bet_status[i];
    }
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
