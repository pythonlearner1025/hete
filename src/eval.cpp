#include "eval.h"
#include <array>
#include <algorithm>
#include <bitset>
#include <stdexcept>
#include <iostream>
#include "eval.h"
#include <random>
#include <numeric>

constexpr int MAX_REMAINING_CARDS = 52 - 2 - 5;
constexpr int MAX_OPP_RANGE = 1326;

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

const std::array<std::array<int, 2>, MAX_OPP_RANGE> card_lookup_table = construct_card_lookup();

// Monte Carlo wp_rollout function for multiple opponents
double wp_rollout_monte_carlo(
    PokerEngine& engine,
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

    // Prepare cumulative distribution functions (CDFs) for opponent ranges
    std::array<std::vector<double>, NUM_PLAYERS - 1> opp_cdfs;
    std::array<std::vector<int>, NUM_PLAYERS - 1> opp_hand_indices;
    for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
        const auto& opp_range = opp_ranges[opp_idx];
        double cumulative = 0.0;
        std::vector<double> cdf;
        std::vector<int> hand_indices;
        if (engine.players[opp_idx].status == PokerEngine::PlayerStatus::Playing) {
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

    for (int sample = 0; sample < WP_MC_SAMPLES; ++sample) {
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

std::array<torch::Tensor, 4> init_batched_cards(std::array<int, 5> board) {
    // Define shapes for each tensor
    std::vector<int64_t> hand_shape = {BS, 2};
    std::vector<int64_t> flop_shape = {BS, 3};
    std::vector<int64_t> turn_shape = {BS, 1};
    std::vector<int64_t> river_shape = {BS, 1};
    
    std::array<torch::Tensor, 4> batched_cards;

    // Initialize hand tensor with placeholder values (-1)
    batched_cards[0] = torch::full(hand_shape, -1, torch::kInt32);

    // Initialize flop tensor
    std::vector<int> flop_cards = {board[0], board[1], board[2]};
    batched_cards[1] = torch::from_blob(flop_cards.data(), {1, 3}, torch::kInt32).expand(flop_shape);

    // Initialize turn tensor
    int turn_card = board[3];
    batched_cards[2] = torch::full(turn_shape, turn_card, torch::kInt32);

    // Initialize river tensor
    int river_card = board[4];
    batched_cards[3] = torch::full(river_shape, river_card, torch::kInt32);

    return batched_cards;
}

torch::Tensor init_batched_fracs(torch::Tensor batched_fracs) {
    std::vector<int64_t> batched_fracs_shape = {BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return batched_fracs.expand(batched_fracs_shape);
}

torch::Tensor init_batched_status(torch::Tensor batched_status) { 
    std::vector<int64_t> batched_status_shape = {BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return batched_status.expand(batched_status_shape);
}

double get_bet_amt(PokerEngine& engine, int player, int act) {
    if (act == 0) return 0.0;
    if (act == 1) return engine.get_call_amount(player);
    double inc = engine.get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            return bet_amt;
        }
        bet_amt += inc;
    }
    return -1.0;
}

int get_lbr_act(
    PokerEngine& engine,
    void* policy_net,
    std::array<std::array<int, 2>, NUM_PLAYERS> player_hands, 
    int player,
    std::array<std::vector<Infoset>, NUM_PLAYERS-1> opp_histories, // list of each infoset opp saw before acting 
    std::array<uint8_t, 52> deck_cards
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
                    if (on_board) throw std::runtime_error("duplicate card found in board & player " + std::to_string(k) + "hand");
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
    // TODO init for those who didn't fold
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
                    std::array<double, NUM_ACTIONS> strat = regret_match(logits);    
                    double max_prob = 0.0;
                    int max_act = 0;
                    for (size_t j=0; j<NUM_ACTIONS; ++j) {
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
    double wp = wp_rollout_monte_carlo(engine, player_hands[player], opp_ranges, curr_board, curr_deck);

    double max_bet = 0.0;
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        if (i != player && engine.players[i].total_bet > max_bet) {
            max_bet = engine.players[i].total_bet;
        }
    }
    double current_bet = engine.players[player].total_bet;
    double asked_amt = max_bet - current_bet;
    double call_util = wp * engine.get_pot() - (1-wp) * asked_amt;

    std::array<float, NUM_ACTIONS> action_utils{};
    action_utils[0] = 0.0;
    action_utils[1] = call_util;

    // init original game state trackers
    Infoset infoset = prepare_infoset(engine, player);
    std::array<torch::Tensor, 4> cards = infoset.cards;

    for (size_t a=2; a<NUM_ACTIONS; ++a) {
        // init cf game states
        torch::Tensor cf_bet_frac = infoset.bet_fracs;
        torch::Tensor cf_bet_status = infoset.bet_status;
        std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS-1> cf_opp_ranges;
        std::copy(opp_ranges.begin(), opp_ranges.end(), cf_opp_ranges.begin());

        // if everyone folds in betting round, game is already over
        // get the bet_index of the last bet - this would be bet of non-folded opponent to player's left
        // get the player_index using (bet_index % (NUM_ROUND_BETS * MAX_PLAYERS)) % MAX_PLAYERS
        // get the player - player_index
        int last_bet_idx = 0;
        for (size_t j=MAX_ROUND_BETS*NUM_PLAYERS*4-1; j>=0; --j) {
            if (cf_bet_frac[j].item<double>() > 0.0) {
                last_bet_idx = j;
                break;
            }
        }

        int last_player_idx = (last_bet_idx % (MAX_ROUND_BETS * NUM_PLAYERS)) % MAX_PLAYERS;
        // if player = 0, last_player = 4, max_players = 5
        // if last_player > player, shift = (max_players - last_player + player)
        // else shift = player - last_player
        // if player = 1, last_player = 0, max_players = 3
        int shift = 0;
        if (last_player_idx > player) shift = NUM_PLAYERS - last_player_idx + player;
        else shift = player - last_player_idx;
        int player_bet_idx = last_bet_idx + shift;
        double bet_amt = get_bet_amt(engine, player, a);
        if (last_player_idx + shift < MAX_ROUND_BETS*NUM_PLAYERS*4) {
            cf_bet_frac[player_bet_idx] = bet_amt;
            cf_bet_status[player_bet_idx] = (bet_amt > 0 ? 1 : 0);
        } else throw std::runtime_error("shift calc is incorrect");

        // prepare for mc_rollouts of game after this act 
        std::array<std::vector<double>, NUM_PLAYERS - 1> opp_cdfs;
        std::array<std::vector<int>, NUM_PLAYERS - 1> opp_hand_indices;
        int live_opps = 0;
        for (int opp_idx = 0; opp_idx < NUM_PLAYERS; ++opp_idx) {
            const auto& opp_range = opp_ranges[opp_idx];
            double cumulative = 0.0;
            std::vector<double> cdf;
            std::vector<int> hand_indices;
            if (engine.players[opp_idx].status == PokerEngine::PlayerStatus::Playing) {
                live_opps++;
                for (int h = 0; h < MAX_OPP_RANGE; ++h) {
                    double prob = opp_range[h];
                    if (prob > 0.0) {
                        const auto& opp_hand = card_lookup_table[h];
                        cumulative += prob;
                        cdf.push_back(cumulative);
                        hand_indices.push_back(h);
                    }
                }
                if (cumulative == 0.0) {
                    throw std::runtime_error("Opponent's range is empty after excluding used cards.");
                }
                // Normalize the CDF
                for (double& val : cdf) {
                    val /= cumulative;
                }
            }
            opp_cdfs[opp_idx] = cdf;
            opp_hand_indices[opp_idx] = hand_indices;
        }

        // fold percent of opp if player takes action a
        double fp = 0.0;
        int batch_idx = 0;

        // batch tensors
        std::array<int, BS> batched_hand_indices{}; 
        std::array<torch::Tensor, 4> batched_cards = init_batched_cards(curr_board);
        torch::Tensor batched_bet_fracs = init_batched_fracs(infoset.bet_fracs);
        torch::Tensor batched_bet_status = init_batched_status(infoset.bet_status);

        // Initialize random engine
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        std::bitset<52> board_cards;
        for (int card : engine.get_board()) board_cards.set(card);
        const int max_handpick_retries = 10;

        for (int sample = 0; sample < ACT_MC_SAMPLES; ++sample) {
            // collect used cards for this sample
            std::bitset<52> sample_used_cards = board_cards;

             // Sample opponent hands without conflicts
            std::array<int, NUM_PLAYERS-1> sampled_opp_hand_indices{-1};
            bool valid_sample = true;

            for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
                const auto& cdf = opp_cdfs[opp_idx];
                const auto& hand_indices = opp_hand_indices[opp_idx];

                // skip dead opponents
                if (cdf.size() == 0) continue;

                // Attempt to sample an opponent hand without conflicts
                bool hand_found = false;
                int retries = 0;
                while (retries < max_handpick_retries) {
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

            // Skip this sample if conflicts couldn't be resolved
            if (!valid_sample) {
                continue;
            }
            
            // fill in batches
            int dead = 0;
            for (int opp_idx = 0; opp_idx < NUM_PLAYERS - 1; ++opp_idx) {
                int opp_hand_index = sampled_opp_hand_indices[opp_idx];
                // skip dead opponents
                if (opp_hand_index < 0) {
                    dead++;
                    continue; 
                }
                std::array<int, 2> opp_hand = card_lookup_table[opp_hand_index]; 
                int idx = batch_idx*live_opps+(opp_idx-dead);
                // hand is still player hand, so update to this opp's hand
                batched_cards[idx][0][0] = opp_hand[0];
                batched_cards[idx][0][1] = opp_hand[1];

                // todo this can just be initialized in beginning without change
                batched_bet_fracs[idx] = cf_bet_frac;
                batched_bet_status[idx] = cf_bet_status;

                // store hand idx for convenience
                batched_hand_indices[idx] = opp_hand_index;
            }

            batch_idx++;

            // do batched inference and update counterfactual opponent ranges 
            if (batch_idx * live_opps == BS) {
                auto batched_logits = deep_cfr_model_forward(policy_net, batched_cards, batched_bet_fracs, batched_bet_status);
                for (size_t b = 0; b < BS; ++b) {
                    auto logits = batched_logits[b];
                    std::array<double, NUM_ACTIONS> regrets = regret_match(logits);
                    int opp_idx = b % live_opps;
                    int opp_hand_idx = batched_hand_indices[b*live_opps+opp_idx];
                    double opp_hand_prob = cf_opp_ranges[opp_idx][opp_hand_idx];
                    fp += opp_hand_prob * (1-regrets[0]);
                    cf_opp_ranges[opp_idx][opp_hand_idx] = opp_hand_prob * (1-regrets[0]);
                }

                // normalize ranges
                for (size_t i=0; i<NUM_PLAYERS-1; ++i) {
                    if (engine.players[i].status == PokerEngine::PlayerStatus::Playing) {
                        cf_opp_ranges[i] = normalize_to_prob_dist(cf_opp_ranges[i]);
                    }
                }

                batch_idx = 0;
            }
        }

        // rollout with updated cf_opp_ranges
        double wp = wp_rollout_monte_carlo(engine, player_hands[player], cf_opp_ranges, curr_board, curr_deck);

        // calc current action's ev
        double pot = engine.get_pot();
        double action_util = fp * engine.get_pot() + (1-fp) * wp * (pot + bet_amt)\ 
            - (1-wp) * (asked_amt + bet_amt);
        action_utils[a] = action_util;
    }

    // return action with max ev
    int max_act = argmax(action_utils);
    if (action_utils[max_act] > 0) return max_act;
    // fold
    return 0;
}

double evaluate(
    void* policy_net,
    int player
) {
    // init game 
    std::array<double, NUM_PLAYERS> starting_stacks{};
    std::array<double, NUM_PLAYERS> antes{};
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        starting_stacks[i] = 100.0;
        antes[i] = 0.0;
    } 
    double small_bet = 0.5;
    double big_bet = 1.0;
    int starting_actor = 0;

    // init history
    std::array<std::vector<Infoset>, NUM_PLAYERS-1> opp_histories;

    // avg mbb winnings
    double total_mbb = 0.0;

    for (size_t i = 0; i < EVAL_MC_SAMPLES; ++i) {

        // randomize the player's & opps hands  
        PokerEngine engine(
            starting_stacks, 
            antes,
            starting_actor,
            small_bet, 
            big_bet, 
            false
        );

        std::array<std::array<int, 2>, NUM_PLAYERS> player_hands{};
        for (size_t p=0;p<NUM_PLAYERS;++p) {
            player_hands[p] = engine.players[p].hand;
        }

        while (!engine.get_game_status() || !engine.is_playing(player)) {
            if (engine.turn() == player) {
                DEBUG_INFO("Player's turn");
                Infoset I = prepare_infoset(engine, player);
                torch::Tensor logits = deep_cfr_model_forward(policy_net, I.cards, I.bet_fracs, I.bet_status);
                std::array<double, NUM_ACTIONS> strat = regret_match(logits);
                int act = sample_action(strat);
                  // Verify and adjust action if necessary
                while (!verify_action(engine, player, act)) {
                    act = (act - 1) % NUM_ACTIONS;
                }
                // Take action
                take_action(engine, player, act);
                DEBUG_INFO("I selected action: " << action_index);
            } else {
                int opp = engine.turn();
                Infoset I = prepare_infoset(engine, player);
                opp_histories[opp].push_back(I);

                int act = get_lbr_act(
                    engine, 
                    policy_net, 
                    player_hands, 
                    opp, 
                    opp_histories, 
                    engine.get_deck()
                );

                while (!verify_action(engine, player, act)) {
                    act = (act - 1) % NUM_ACTIONS;
                }
                // Take action
                take_action(engine, player, act);
                DEBUG_INFO("I selected action: " << act); 
            }
        }

        total_mbb += (engine.get_payoffs()[player] / big_bet) * 1000;
        // calculate the player's winnings
    }

    double avg_mbb = total_mbb / EVAL_MC_SAMPLES;
    DEBUG_NONE("Policy net is evaluated at: " << avg_mbb << "mbb");
    return avg_mbb;
}