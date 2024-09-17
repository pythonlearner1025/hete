#include "util.h"

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

    // Convert bet_fracs array to tensor
    I.bet_fracs = torch::from_blob(history.second.data(), 
                                   {1, static_cast<long long>(history.second.size())}, 
                                   torch::kFloat32);

    // Convert bet_status array to tensor
    I.bet_status = torch::from_blob(history.first.data(), 
                                    {1, static_cast<long long>(history.first.size())}, 
                                    torch::kBool).to(torch::kFloat32);

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



template <typename T, std::size_t N>
std::array<T, N> normalize_to_prob_dist(const std::array<T, N>& arr) {
    // First, we'll use reduce to sum all elements
    T sum = std::reduce(arr.begin(), arr.end());

    // Check if the sum is zero to avoid division by zero
    if (std::abs(sum) < std::numeric_limits<T>::epsilon()) {
        // If sum is zero, return a uniform distribution
        std::array<T, N> result;
        std::fill(result.begin(), result.end(), static_cast<T>(1.0 / N));
        return result;
    }

    // Now, we'll use transform to divide each element by the sum
    std::array<T, N> result;
    std::transform(arr.begin(), arr.end(), result.begin(),
                   [sum](const T& val) { return val / sum; });

    return result;
}

template <typename T, std::size_t N>
std::size_t argmax(const std::array<T, N>& arr) {
    return std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));
}