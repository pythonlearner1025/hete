#include "util.h"

BetHistory construct_history(PokerEngine& engine) {
    BetHistory hist;
    double pot = engine.small_blind + engine.big_blind;
    
    for(int r = 0; r < 4; r++) {
        for(int p = 0; p < NUM_PLAYERS; p++) {
            for(int b = 0; b < MAX_ROUND_BETS; b++) {
                if(engine.players[p].bets_per_round[r][b] >= 0) {
                    hist.amounts[r][p][b] = engine.players[p].bets_per_round[r][b] / pot;
                    hist.status[r][p][b] = engine.players[p].bets_per_round[r][b] > 0;
                    pot += engine.players[p].bets_per_round[r][b];
                }
            }
        }
    }
    return hist;
}

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

    // Fill CPU tensors using accessors
    auto hand_a = hand.accessor<int32_t, 2>();
    auto flop_a = flop.accessor<int32_t, 2>();
    auto turn_a = turn.accessor<int32_t, 2>();
    auto river_a = river.accessor<int32_t, 2>();

    // Update hand cards
    for (int i = 0; i < 2; ++i) {
        hand_a[0][i] = S.hand[i];
    }

    // Update flop cards
    for (int i = 0; i < 3; ++i) {
        flop_a[0][i] = S.flop[i];
    }

    // Update turn card
    turn_a[0][0] = S.turn[0];

    // Update river card
    river_a[0][0] = S.river[0];

    bet_fracs = S.bet_fracs;
    bet_status = S.bet_status;
}

float sample_uniform() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng); // Thread-safe random float between 0 and 1
}

int sample_iter(size_t max_iter) {
    if (max_iter == 0) {
        return 0;
    }
    
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, static_cast<int>(max_iter));
    
    try {
        return dist(rng);
    } catch (const std::exception& e) {
        std::cerr << "Error in sample_iter: " << e.what() << " max_iter: " << max_iter << std::endl;
        return 1;  // Return safe default
    }
}
// Sample an action according to the strategy probabilities
int sample_action(const std::array<double, NUM_ACTIONS>& strat) {
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng); // Thread-safe random number between 0 and 1
    double cumulative = 0.0;
    for (size_t i = 0; i < strat.size(); ++i) {
        cumulative += strat[i];
        if (r <= cumulative) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(strat.size() - 1);
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
    auto history = construct_history(game);
    std::array<int, 2> hand = game.players[player].hand;
    std::array<int, 5> board = game.get_board();

    auto [bet_fracs, bet_status] = history.to_tensors();
    state->bet_fracs = bet_fracs;
    state->bet_status = bet_status;

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


// Must return a probability distribution
std::array<double, NUM_ACTIONS> sample_prob(const torch::Tensor& logits, float beta) {
    double logits_sum = logits.sum().item<double>() + beta;
    std::array<double, NUM_ACTIONS> strat{};
    auto strategy_tensor = (logits+beta) / logits_sum;
    auto strat_data = strategy_tensor.data_ptr<float>();
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        strat[i] = strat_data[i];
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