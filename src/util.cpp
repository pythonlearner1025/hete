#include "util.h"

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

void get_cards(PokerEngine& game, int player, Infoset& I) {
    std::array<int, 5> board = game.get_board();
    int board_size = 0;
    for (size_t i = 0; i < 5; ++i) {
        if (board[i] != -1) board_size++;
    }
    
    DEBUG_NONE("Player hand: " << game.players[player].hand[0] << ", " << game.players[player].hand[1]);
    
    // Hand cards
    DEBUG_NONE(I.cards[0].sizes());
    DEBUG_NONE(I.cards[1].sizes());
    DEBUG_NONE(I.cards[2].sizes());
    DEBUG_NONE(I.cards[3].sizes());
    auto hand_accessor = I.cards[0].accessor<int, 1>();
    DEBUG_NONE("worked");
    hand_accessor[0] = game.players[player].hand[0];
    hand_accessor[1] = game.players[player].hand[1];
    
    
    // Flop cards
    auto flop_accessor = I.cards[1].accessor<int, 2>();
    if (board_size >= 3) {
        DEBUG_INFO("Flop: " << board[0] << ", " << board[1] << ", " << board[2]);
        flop_accessor[0][0] = board[0];
        flop_accessor[0][1] = board[1];
        flop_accessor[0][2] = board[2];
    } else {
        flop_accessor[0][0] = -1;
        flop_accessor[0][1] = -1;
        flop_accessor[0][2] = -1;
    }
    
    // Turn card
    auto turn_accessor = I.cards[2].accessor<int, 2>();
    if (board_size >= 4) {
        DEBUG_INFO("Turn: " << board[3]);
        turn_accessor[0][0] = board[3];
    } else {
        turn_accessor[0][0] = -1;
    }
    
    // River card
    auto river_accessor = I.cards[3].accessor<int, 2>();
    if (board_size >= 5) {
        DEBUG_INFO("River: " << board[4]);
        river_accessor[0][0] = board[4];
    } else {
        river_accessor[0][0] = -1;
    }
}

Infoset prepare_infoset(
    PokerEngine& game,
    int player
) {
    Infoset I;
    auto history = game.construct_history();
    get_cards(game, player, I);

    torch::Tensor bet_status = torch::empty({1, history.first.size()}, torch::kInt);
    auto status_accessor = bet_status.accessor<int,2>();

    torch::Tensor bet_fracs = torch::empty({1, history.second.size()}, torch::kFloat);
    auto frac_accessor = bet_fracs.accessor<float,2>();
    for (size_t i = 0; i < NUM_PLAYERS * 4 * MAX_ROUND_BETS; ++i) {
        status_accessor[0][i] = history.first[i]; 
        frac_accessor[0][i] = history.second[i];
    }

    I.bet_fracs = bet_fracs;
    I.bet_status = bet_status;

    return I;
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
    // Sum logits across actions
    auto logits_sum = relu_logits.sum(1, true); // [batch_size, 1]
    // Identify batches where the sum is positive
    auto positive_mask = logits_sum.squeeze(1) > 0; // [batch_size]
    // Initialize the strategy tensor
    auto strategy = torch::zeros_like(batched_logits); // [batch_size, num_actions]

    // Handle positive sums
    if (positive_mask.any().item<bool>()) {
        // Indices of positive batches
        auto positive_indices = torch::nonzero(positive_mask).squeeze(1); // [num_positive]
        // Select positive logits and sums
        auto positive_logits = relu_logits.index_select(0, positive_indices); // [num_positive, num_actions]
        auto positive_logits_sum = logits_sum.index_select(0, positive_indices); // [num_positive, 1]
        // Compute the strategy for positive batches
        auto positive_strategy = positive_logits / positive_logits_sum; // [num_positive, num_actions]
        // Update the strategy tensor
        strategy.index_copy_(0, positive_indices, positive_strategy);
    }

    // Handle non-positive sums
    auto negative_mask = ~positive_mask; // [batch_size]
    if (negative_mask.any().item<bool>()) {
        // Indices of negative batches
        auto negative_indices = torch::nonzero(negative_mask).squeeze(1); // [num_negative]
        // Select negative logits
        auto negative_logits = relu_logits.index_select(0, negative_indices); // [num_negative, num_actions]
        // Find the action with the maximum logit
        auto max_indices = torch::argmax(negative_logits, 1); // [num_negative]
        // Create one-hot vectors
        auto one_hot = torch::zeros({negative_indices.size(0), batched_logits.size(1)}, batched_logits.options());
        one_hot.scatter_(1, max_indices.unsqueeze(1), 1);
        // Update the strategy tensor
        strategy.index_copy_(0, negative_indices, one_hot);
    }

    return strategy;
}
