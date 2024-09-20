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

void take_action(PokerEngine& engine, int player, int act) {
    DEBUG_INFO("Chosen act: " << act);
    if (act == 0) {
        engine.fold(player);
        return;
    }
    if (act == 1) {
        engine.check_or_call(player); 
        return;
    }
    double inc = engine.get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            engine.bet_or_raise(player, bet_amt);
            return;
        }
        bet_amt += inc;
    }
}

bool verify_action(PokerEngine& engine, int player, int act) {
    if (act == 0) {
        return engine.can_fold(player);
    }
    if (act == 1) {
        return engine.can_check_or_call(player); 
    }
    double inc = engine.get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            return engine.can_bet_or_raise(player, bet_amt);
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

    DEBUG_INFO("Player hand: " << game.players[player].hand[0] << ", " << game.players[player].hand[1]);

    I.cards[0] = torch::tensor({static_cast<int64_t>(game.players[player].hand[0]), 
                                static_cast<int64_t>(game.players[player].hand[1])}).view({1, 2});

    DEBUG_INFO("Board size: " << board_size);

    if (board_size >= 3) {
        DEBUG_INFO("Flop: " << board[0] << ", " << board[1] << ", " << board[2]);
        I.cards[1] = torch::tensor({static_cast<int64_t>(board[0]), 
                                    static_cast<int64_t>(board[1]), 
                                    static_cast<int64_t>(board[2])}).view({1, 3});
    } else {
        I.cards[1] = torch::tensor({-1, -1, -1}).view({1, 3});
    }

    if (board_size >= 4) {
        DEBUG_INFO("Turn: " << board[3]);
        I.cards[2] = torch::tensor({static_cast<int64_t>(board[3])}).view({1, 1});
    } else {
        I.cards[2] = torch::tensor({-1}).view({1, 1});
    }

    if (board_size >= 5) {
        DEBUG_INFO("River: " << board[4]);
        I.cards[3] = torch::tensor({static_cast<int64_t>(board[4])}).view({1, 1});
    } else {
        I.cards[3] = torch::tensor({-1}).view({1, 1});
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