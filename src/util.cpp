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
    int player,
    int max_bets_per_player
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