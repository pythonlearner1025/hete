#include "eval.h"
#include <array>
#include <algorithm>
#include <bitset>
#include <stdexcept>
#include <iostream>
#include "eval.h"


const std::array<std::array<int, 2>, MAX_OPP_RANGE> card_lookup_table = construct_card_lookup();

constexpr std::array<std::array<int, 2>, MAX_OPP_RANGE> construct_card_lookup() {
    std::array<std::array<int, 2>, MAX_OPP_RANGE> table{};
    int idx = 0;
    for (int i = 0; i < 52; ++i) {
        for (int j = i + 1; j < 52; ++j) {
            table[idx++] = {i, j};
        }
    }
    return table;
}

constexpr int MAX_REMAINING_CARDS = 52 - 2 - 5;
constexpr int MAX_OPP_RANGE = MAX_OPP_RANGE;

// NOTE -- calculating a six-player showdown is Comb(40,5)*1326^6, which is intractable
// a more tractable option would be to do MC rollouts and get avg win
double wp_rollout(
    const std::array<int, 2>& player_hand,
    const std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS>& opp_ranges,
    const std::array<int, 5>& board,
    const std::array<uint8_t, 52>& deck_cards
) {
    int board_size = std::count_if(board.begin(), board.end(), [](int card) { return card != NULL_CARD; });
    int cards_to_deal = 5 - board_size;

    if (cards_to_deal < 0 || cards_to_deal > MAX_REMAINING_CARDS) {
        throw std::runtime_error("Invalid number of cards to deal");
    }

    std::bitset<52> used_cards;
    for (int card : player_hand) used_cards.set(card);
    for (int card : board) {
        if (card != NULL_CARD) used_cards.set(card);
    }

    std::array<int, MAX_REMAINING_CARDS> remaining_cards;
    int remaining_count = 0;
    for (int card : deck_cards) {
        if (!used_cards.test(card)) {
            remaining_cards[remaining_count++] = card;
        }
    }

    std::array<bool, MAX_REMAINING_CARDS> v{};
    std::fill_n(v.begin() + (remaining_count - cards_to_deal), cards_to_deal, true);

    omp::Hand initial_player_hand = omp::Hand::empty();
    for (int card : player_hand) initial_player_hand += omp::Hand(card);

    std::array<int, 5> new_board;
    std::copy(board.begin(), board.end(), new_board.begin());

    std::bitset<52> board_bitset;
    for (int card : board) {
        if (card != NULL_CARD) {
            board_bitset.set(card);
        }
    }

    double total_won = 0.0;
    int total_boards = 0;

    do {
        int new_board_size = board_size;
        for (int i = 0; i < remaining_count; ++i) {
            if (v[i]) {
                new_board[new_board_size++] = remaining_cards[i];
                board_bitset.set(remaining_cards[i]);
            }
        }

        omp::Hand my_hand_board = initial_player_hand;
        omp::Hand opp_hand_board = omp::Hand::empty();
        omp::HandEvaluator evaluator;

        for (int i = 0; i < new_board_size; ++i) {
            my_hand_board += omp::Hand(new_board[i]);
            opp_hand_board += omp::Hand(new_board[i]);
        }
        int my_strength = evaluator.evaluate(my_hand_board);

        double won = 0.0;
        for (size_t j = 0; j < opp_range.size(); ++j) {
            double prob = opp_range[j];
            const auto& opp_hand = card_lookup_table[j];

            if (prob > 0 && 
                !board_bitset.test(opp_hand[0]) && !board_bitset.test(opp_hand[1])) {
                omp::Hand current_opp_hand = opp_hand_board;
                for (int card : opp_hand) current_opp_hand += omp::Hand(card);

                int opp_strength = evaluator.evaluate(current_opp_hand);
                if (my_strength > opp_strength) won += prob;
            }
        }
        total_won += won;
        ++total_boards;

        for (int i = board_size; i < new_board_size; ++i) {
            board_bitset.reset(new_board[i]);
        }
    } while (std::next_permutation(v.begin(), v.end()));

    if (total_boards == 0) {
        throw std::runtime_error("No valid boards evaluated");
    }
    std::cout << "Total boards evaluated: " << total_boards << std::endl;
    return total_won / total_boards;
}

int get_lbr_act(
    PokerEngine& engine,
    void* policy_net,
    std::array<std::array<int, 2>, NUM_PLAYERS> player_hands, 
    int player,
    std::array<std::vector<Infoset>, NUM_PLAYERS> opp_histories, // list of each infoset opp saw before acting 
    std::array<int, 52> deck_cards
) {

    std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS> opp_ranges{}; 
    std::array<int, 5> curr_board = engine.get_board();
    std::array<uint8_t, 52> curr_deck = engine.get_deck();

    // count invalids
    std::array<int, NUM_PLAYERS> player_invalids{};
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
            std::array<int, 2> opp_hand = card_lookup_table[i];
            bool on_board = false;

            // check board for invalids
            for (size_t k=0; k<5; ++k) {
                if (curr_board[k] == opp_hand[0] || curr_board[k] == opp_hand[1]) {
                    player_invalids[i]++;
                    on_board = true;
                }
            }

            // check other player hand for invalids
            for (size_t k=0; k<NUM_PLAYERS; ++k) {
                if (player_hands[k][0] == opp_hand[0] || player_hands[k][1] == opp_hand[1]) {
                    if (on_board) throw std::runtime_error("duplicate card found in board & player " + k + "hand");
                    else {
                        player_invalids[i]++;
                        on_board = true;
                    }
                }
            }

            if (!on_board) opp_ranges[i][j] = 1.0;
            else opp_ranges[i][j] = 0.0;
        }
    }

    // init opp ranges 
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
            if (opp_ranges[i][j] != 0.0) 
                opp_ranges[i][j] = 1.0/static_cast<float>(player_invalids[i]);
        }

        // update opp range to reflect game history
        if (opp_histories[i].size() > 0) {
            for (auto infoset : opp_histories[i]) {
                for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
                    float curr_hand_prob = opp_ranges[i][j]; 
                    std::array<int, 2> opp_hand = card_lookup_table[j];
                    torch::Tensor logits = deep_cfr_model_forward(policy_net, infoset.cards, infoset.bet_fracs, infoset.bet_status);
                    int n_acts = get_action_head_dim(policy_net);
                    std::array<double, MAX_ACTIONS> strat = regret_match(logits, n_acts);    
                    double max_prob = 0.0;
                    int max_act = 0;
                    for (size_t j=0; j<MAX_ACTIONS; ++j) {
                        if (strat[i] > max_prob) {
                            max_prob = strat[i];
                            max_act = i;
                        }
                    }
                    opp_ranges[i][j] = curr_hand_prob * strat[max_act];
                }
            }
        }
    }

    

    // calc first wprollout 
    double wp = wp_rollout(player_hands[player], opp_ranges, curr_board, curr_deck);

    
}