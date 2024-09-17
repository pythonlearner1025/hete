#include "eval.h"
#include <array>
#include <algorithm>
#include <bitset>
#include <stdexcept>
#include <iostream>
#include "eval.h"
#include <random>
#include <numeric>

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

// Monte Carlo wp_rollout function for multiple opponents
double wp_rollout_monte_carlo(
    const std::array<int, 2>& player_hand,
    const std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS - 1>& opp_ranges,
    const std::array<int, 5>& board,
    const std::array<uint8_t, 52>& deck_cards,
    int num_samples = 10000  // Number of Monte Carlo samples
) {
    int board_size = std::count_if(board.begin(), board.end(), [](int card) { return card != NULL_CARD; });
    int cards_to_deal = 5 - board_size;

    if (cards_to_deal < 0) {
        throw std::runtime_error("Invalid number of cards to deal");
    }

    // Collect used cards (player's hand and the board)
    std::bitset<52> used_cards;
    for (int card : player_hand) used_cards.set(card);
    for (int card : board) {
        if (card != NULL_CARD) used_cards.set(card);
    }

    // Prepare remaining cards for dealing
    std::vector<int> remaining_cards;
    for (int card : deck_cards) {
        if (!used_cards.test(card)) {
            remaining_cards.push_back(card);
        }
    }

    // Prepare cumulative distribution functions (CDFs) for opponent ranges
    std::array<std::vector<double>, NUM_PLAYERS - 1> opp_cdfs;
    std::array<std::vector<int>, NUM_PLAYERS - 1> opp_hand_indices;
    for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
        const auto& opp_range = opp_ranges[opp_idx];
        double cumulative = 0.0;
        std::vector<double> cdf;
        std::vector<int> hand_indices;
        for (int h = 0; h < MAX_OPP_RANGE; ++h) {
            double prob = opp_range[h];
            if (prob > 0.0) {
                const auto& opp_hand = card_lookup_table[h];
                if (!used_cards.test(opp_hand[0]) && !used_cards.test(opp_hand[1])) {
                    cumulative += prob;
                    cdf.push_back(cumulative);
                    hand_indices.push_back(h);
                }
            }
        }
        if (cumulative == 0.0) {
            throw std::runtime_error("Opponent's range is empty after excluding used cards.");
        }
        // Normalize the CDF
        for (double& val : cdf) {
            val /= cumulative;
        }
        opp_cdfs[opp_idx] = cdf;
        opp_hand_indices[opp_idx] = hand_indices;
    }

    // Initialize random engine
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double total_wins = 0.0;
    int valid_samples = 0;

    omp::HandEvaluator evaluator;

    for (int sample = 0; sample < num_samples; ++sample) {
        // Sample remaining board cards
        std::vector<int> sampled_board_cards;
        if (cards_to_deal > 0) {
            std::shuffle(remaining_cards.begin(), remaining_cards.end(), rng);
            sampled_board_cards.assign(remaining_cards.begin(), remaining_cards.begin() + cards_to_deal);
        }

        // Prepare full board
        std::array<int, 5> full_board = board;
        int idx = board_size;
        for (int card : sampled_board_cards) {
            full_board[idx++] = card;
        }

        // Collect used cards for this sample
        std::bitset<52> sample_used_cards = used_cards;
        for (int card : sampled_board_cards) {
            sample_used_cards.set(card);
        }

        // Sample opponent hands without conflicts
        std::vector<int> sampled_opp_hand_indices(NUM_PLAYERS - 1, -1);
        bool valid_sample = true;

        for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
            const auto& cdf = opp_cdfs[opp_idx];
            const auto& hand_indices = opp_hand_indices[opp_idx];

            // Attempt to sample an opponent hand without conflicts
            bool hand_found = false;
            const int max_retries = 10;
            int retries = 0;
            while (retries < max_retries) {
                // Sample a random value
                double rand_val = dist(rng);
                // Find the corresponding hand index
                auto it = std::lower_bound(cdf.begin(), cdf.end(), rand_val);
                if (it == cdf.end()) {
                    ++retries;
                    continue;
                }
                int idx_in_cdf = std::distance(cdf.begin(), it);
                int hand_idx = hand_indices[idx_in_cdf];
                const auto& opp_hand = card_lookup_table[hand_idx];

                // Check for card conflicts with already used cards
                if (!sample_used_cards.test(opp_hand[0]) && !sample_used_cards.test(opp_hand[1])) {
                    // No conflict
                    sample_used_cards.set(opp_hand[0]);
                    sample_used_cards.set(opp_hand[1]);
                    sampled_opp_hand_indices[opp_idx] = hand_idx;
                    hand_found = true;
                    break;
                }
                ++retries;
            }
            if (!hand_found) {
                valid_sample = false;
                break;
            }
        }

        if (!valid_sample) {
            // Skip this sample if conflicts couldn't be resolved
            continue;
        }

        // Evaluate player's hand
        omp::Hand player_full_hand = omp::Hand::empty();
        for (int card : player_hand) player_full_hand += omp::Hand(card);
        for (int card : full_board) player_full_hand += omp::Hand(card);
        int player_strength = evaluator.evaluate(player_full_hand);

        // Evaluate opponent hands
        std::vector<int> opp_strengths(NUM_PLAYERS - 1);
        for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
            int hand_idx = sampled_opp_hand_indices[opp_idx];
            const auto& opp_hand = card_lookup_table[hand_idx];
            omp::Hand opp_full_hand = omp::Hand::empty();
            opp_full_hand += omp::Hand(opp_hand[0]);
            opp_full_hand += omp::Hand(opp_hand[1]);
            for (int card : full_board) opp_full_hand += omp::Hand(card);
            opp_strengths[opp_idx] = evaluator.evaluate(opp_full_hand);
        }

        // Determine winner(s)
        int max_strength = player_strength;
        std::vector<int> winners = { -1 }; // -1 for the player
        for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
            if (opp_strengths[opp_idx] > max_strength) {
                max_strength = opp_strengths[opp_idx];
                winners = { opp_idx };
            } else if (opp_strengths[opp_idx] == max_strength) {
                winners.push_back(opp_idx);
            }
        }

        // Calculate win share for the player
        double player_win_share = 0.0;
        if (std::find(winners.begin(), winners.end(), -1) != winners.end()) {
            player_win_share = 1.0 / winners.size();
        }

        total_wins += player_win_share;
        ++valid_samples;
    }

    if (valid_samples == 0) {
        throw std::runtime_error("No valid samples generated in Monte Carlo simulation.");
    }

    double win_probability = total_wins / valid_samples;
    return win_probability;
}

// Exhaustive wp_rollout function for multiple opponents
double wp_rollout_exhaustive(
    const std::array<int, 2>& player_hand,
    const std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS - 1>& opp_ranges,
    const std::array<int, 5>& board,
    const std::array<uint8_t, 52>& deck_cards
) {
    int board_size = std::count_if(board.begin(), board.end(), [](int card) { return card != NULL_CARD; });
    int cards_to_deal = 5 - board_size;

    if (cards_to_deal < 0) {
        throw std::runtime_error("Invalid number of cards to deal");
    }

    // Collect used cards (player's hand and the board)
    std::bitset<52> used_cards;
    for (int card : player_hand) used_cards.set(card);
    for (int card : board) {
        if (card != NULL_CARD) used_cards.set(card);
    }

    // Prepare remaining cards for dealing
    std::vector<int> remaining_cards;
    for (int card : deck_cards) {
        if (!used_cards.test(card)) {
            remaining_cards.push_back(card);
        }
    }

    // Generate all possible combinations of remaining board cards
    std::vector<std::vector<int>> board_combinations;
    if (cards_to_deal == 0) {
        board_combinations.push_back({});
    } else {
        std::vector<bool> v(remaining_cards.size(), false);
        std::fill(v.end() - cards_to_deal, v.end(), true);
        do {
            std::vector<int> new_board_cards;
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i]) {
                    new_board_cards.push_back(remaining_cards[i]);
                }
            }
            board_combinations.push_back(new_board_cards);
        } while (std::next_permutation(v.begin(), v.end()));
    }

    // Prepare possible hands for each opponent
    std::vector<std::vector<std::pair<int, double>>> opponents_possible_hands(NUM_PLAYERS - 1);
    for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
        const auto& opp_range = opp_ranges[opp_idx];
        for (int h = 0; h < MAX_OPP_RANGE; ++h) {
            double prob = opp_range[h];
            if (prob > 0.0) {
                const auto& opp_hand = card_lookup_table[h];
                if (!used_cards.test(opp_hand[0]) && !used_cards.test(opp_hand[1])) {
                    opponents_possible_hands[opp_idx].emplace_back(h, prob);
                }
            }
        }
    }

    // Generate all combinations of opponent hands
    std::vector<std::vector<int>> opponent_hand_indices;
    std::function<void(int, std::vector<int>&)> generate_combinations = [&](int idx, std::vector<int>& current) {
        if (idx == NUM_PLAYERS - 1) {
            opponent_hand_indices.push_back(current);
            return;
        }
        for (const auto& [hand_idx, prob] : opponents_possible_hands[idx]) {
            // Check for card overlaps with already selected hands
            const auto& opp_hand = card_lookup_table[hand_idx];
            bool overlap = false;
            for (int i = 0; i < idx; ++i) {
                const auto& prev_hand = card_lookup_table[current[i]];
                if (opp_hand[0] == prev_hand[0] || opp_hand[0] == prev_hand[1] ||
                    opp_hand[1] == prev_hand[0] || opp_hand[1] == prev_hand[1]) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                current[idx] = hand_idx;
                generate_combinations(idx + 1, current);
            }
        }
    };

    std::vector<int> current_combination(NUM_PLAYERS - 1);
    generate_combinations(0, current_combination);

    // Evaluate each combination
    double total_weighted_wins = 0.0;
    double total_weight = 0.0;

    omp::HandEvaluator evaluator;

    for (const auto& board_cards : board_combinations) {
        // Prepare full board
        std::array<int, 5> full_board = board;
        int idx = board_size;
        for (int card : board_cards) {
            full_board[idx++] = card;
        }

        // Evaluate player's hand
        omp::Hand player_full_hand = omp::Hand::empty();
        for (int card : player_hand) player_full_hand += omp::Hand(card);
        for (int card : full_board) player_full_hand += omp::Hand(card);
        int player_strength = evaluator.evaluate(player_full_hand);

        // Evaluate each opponent hand combination
        for (const auto& opp_hands_indices : opponent_hand_indices) {
            // Check for overlaps with new board cards
            bool valid = true;
            std::bitset<52> used_in_this_combination = used_cards;
            for (int card : board_cards) {
                used_in_this_combination.set(card);
            }

            // Prepare opponent hands and check for overlaps
            std::vector<omp::Hand> opp_full_hands(NUM_PLAYERS - 1);
            double combination_prob = 1.0;

            for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
                int hand_idx = opp_hands_indices[opp_idx];
                const auto& opp_hand = card_lookup_table[hand_idx];

                if (used_in_this_combination.test(opp_hand[0]) || used_in_this_combination.test(opp_hand[1])) {
                    valid = false;
                    break;
                }

                used_in_this_combination.set(opp_hand[0]);
                used_in_this_combination.set(opp_hand[1]);

                omp::Hand opp_full_hand = omp::Hand::empty();
                opp_full_hand += omp::Hand(opp_hand[0]);
                opp_full_hand += omp::Hand(opp_hand[1]);
                for (int card : full_board) opp_full_hand += omp::Hand(card);
                opp_full_hands[opp_idx] = opp_full_hand;

                // Multiply probabilities
                double prob = opp_ranges[opp_idx][hand_idx];
                combination_prob *= prob;
            }

            if (!valid) continue;

            // Evaluate opponent hand strengths
            std::vector<int> opp_strengths(NUM_PLAYERS - 1);
            for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
                opp_strengths[opp_idx] = evaluator.evaluate(opp_full_hands[opp_idx]);
            }

            // Determine winner(s)
            int max_strength = player_strength;
            std::vector<int> winners = { -1 }; // -1 for the player
            for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
                if (opp_strengths[opp_idx] > max_strength) {
                    max_strength = opp_strengths[opp_idx];
                    winners = { opp_idx };
                } else if (opp_strengths[opp_idx] == max_strength) {
                    winners.push_back(opp_idx);
                }
            }

            // Calculate win share for the player
            double player_win_share = 0.0;
            if (std::find(winners.begin(), winners.end(), -1) != winners.end()) {
                player_win_share = 1.0 / winners.size();
            }

            total_weighted_wins += combination_prob * player_win_share;
            total_weight += combination_prob;
        }
    }

    if (total_weight == 0.0) {
        throw std::runtime_error("Total weight is zero, no valid combinations evaluated.");
    }

    double win_probability = total_weighted_wins / total_weight;
    return win_probability;
}


int get_lbr_act(
    PokerEngine& engine,
    void* policy_net,
    std::array<std::array<int, 2>, NUM_PLAYERS> player_hands, 
    int player,
    std::array<std::vector<Infoset>, NUM_PLAYERS-1> opp_histories, // list of each infoset opp saw before acting 
    std::array<int, 52> deck_cards
) {

    std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS-1> opp_ranges{}; 
    std::array<int, 5> curr_board = engine.get_board();
    std::array<uint8_t, 52> curr_deck = engine.get_deck();

    // count invalids
    std::array<int, NUM_PLAYERS-1> opp_invalids{};
    for (size_t i=0; i<NUM_PLAYERS-1; ++i) {
        for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
            std::array<int, 2> opp_hand = card_lookup_table[i];
            bool on_board = false;

            // check board for invalids
            for (size_t k=0; k<5; ++k) {
                if (curr_board[k] == opp_hand[0] || curr_board[k] == opp_hand[1]) {
                    opp_invalids[i]++;
                    on_board = true;
                }
            }

            // check other player hand for invalids
            for (size_t k=0; k<NUM_PLAYERS; ++k) {
                if (player_hands[k][0] == opp_hand[0] || player_hands[k][1] == opp_hand[1]) {
                    if (on_board) throw std::runtime_error("duplicate card found in board & player " + k + "hand");
                    else {
                        opp_invalids[i]++;
                        on_board = true;
                    }
                }
            }

            if (!on_board) opp_ranges[i][j] = 1.0;
            else opp_ranges[i][j] = 0.0;
        }
    }

    // init opp ranges 
    for (size_t i=0; i<NUM_PLAYERS-1; ++i) {
        for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
            if (opp_ranges[i][j] != 0.0) 
                opp_ranges[i][j] = 1.0/static_cast<float>(opp_invalids[i]);
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
    double wp = wp_rollout_monte_carlo(player_hands[player], opp_ranges, curr_board, curr_deck);

}