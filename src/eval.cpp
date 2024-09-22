#include "eval.h"
#include <array>
#include <algorithm>
#include <bitset>
#include <stdexcept>
#include <iostream>
#include "eval.h"
#include <random>
#include <numeric>

/*
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
    int player,
    const std::array<int, 2>& player_hand,
    const std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS - 1>& opp_ranges,
    const std::array<int, NUM_PLAYERS - 1>& opp_idxs,
    const std::array<int, 5>& board,
    const std::array<int, 52>& deck_cards
) {
    int board_size = std::count_if(board.begin(), board.end(), [](int card) { return card != NULL_CARD; });
    int cards_to_deal = 5 - board_size;

    if (cards_to_deal < 0) {
        throw std::runtime_error("Invalid number of cards to deal");
    }

    // Collect used cards (player's hand and the board)
    std::bitset<52> used_cards;
    for (int card : player_hand) {
        //DEBUG_INFO("setting card " << card << " as used");
        used_cards.set(card);
    }
    for (int card : board) {
        if (card != NULL_CARD) {
            //DEBUG_INFO("setting card " << card << " as used");
            used_cards.set(card);
        }
    }

    // Prepare remaining cards for dealing
    std::array<int, 52> remaining_cards{NULL_CARD};
    for (int card : deck_cards) {
        //DEBUG_INFO("testing card " << card << "");
        if (card != NULL_CARD && !used_cards.test(card)) {
            remaining_cards[card] = card;
        }
    }

    // sanity check on remaining cards
    if (remaining_cards.size() > 52 || remaining_cards.empty()) {
        throw std::runtime_error("Invalid remaining_cards size: " + std::to_string(remaining_cards.size()));
    } 

    // Prepare cumulative distribution functions (CDFs) for opponent ranges
    std::array<std::vector<double>, NUM_PLAYERS - 1> opp_cdfs;
    std::array<std::vector<int>, NUM_PLAYERS - 1> opp_hand_indices;
    for (int i = 0; i < NUM_PLAYERS - 1; ++i) {
        int opp_idx = opp_idxs[i];
        DEBUG_INFO("i: " << i << " opp_idx: " << opp_idx);
        if (opp_idx == player) continue;
        const auto& opp_range = opp_ranges[i];
        double cumulative = 0.0;
        std::vector<double> cdf;
        std::vector<int> hand_indices;
        if (engine.players[opp_idx].status == PokerEngine::PlayerStatus::Playing) {
            for (int h = 0; h < MAX_OPP_RANGE; ++h) {
                double prob = opp_range[h];
                const auto& opp_hand = card_lookup_table[h];
                DEBUG_INFO("opp hand" << opp_hand[0] << "," << opp_hand[1] << "prob: " << prob);
                if (prob > 0.0) {
                    DEBUG_INFO("testing opp hand " << opp_hand[0] << "," << opp_hand[1]);
                    if (!used_cards.test(opp_hand[0]) && !used_cards.test(opp_hand[1])) {
                        cumulative += prob;
                        cdf.push_back(cumulative);
                        hand_indices.push_back(h);
                        DEBUG_INFO("cumulated");
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
            DEBUG_INFO("normalize cdf");
        }
        opp_cdfs[i] = cdf;
        opp_hand_indices[i] = hand_indices;
        DEBUG_INFO("assign cdf");
    }

    // Initialize random engine
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    DEBUG_INFO("init random");

    double total_wins = 0.0;
    int valid_samples = 0;

    omp::HandEvaluator evaluator;
    DEBUG_INFO("Begin wp mc sampling");

    // sanity check on remaining cards
    if (remaining_cards.size() > 52 || remaining_cards.empty()) {
        throw std::runtime_error("Invalid remaining_cards size: " + std::to_string(remaining_cards.size()));
    } 
    DEBUG_INFO("Address of remaining_cards: " << (void*)&remaining_cards);
    DEBUG_INFO("Address of remaining_cards.data(): " << (void*)remaining_cards.data());
    DEBUG_INFO("remaining card count: " << remaining_cards.size());
    DEBUG_INFO("Before shuffle, remaining_cards.size(): " << remaining_cards.size());
    std::shuffle(remaining_cards.begin(), remaining_cards.end(), rng);
    DEBUG_INFO("After shuffle, remaining_cards.size(): " << remaining_cards.size());

    for (int sample = 0; sample < WP_MC_SAMPLES; ++sample) {

        // Sample remaining board cards
        std::vector<int> sampled_board_cards{};
        if (cards_to_deal > 0) {
            DEBUG_INFO("shuffling");
            DEBUG_INFO("remaining card count: " << remaining_cards.size());
            DEBUG_INFO("cards to deal: " << cards_to_deal);
            std::shuffle(remaining_cards.begin(), remaining_cards.end(), rng);
            int dealt_cards = 0;
            int card_idx = 0;
            while (dealt_cards < cards_to_deal) {
                if (remaining_cards[card_idx] != NULL_CARD) {
                    sampled_board_cards.push_back(remaining_cards[card_idx]);
                    dealt_cards++;
                    card_idx++;
                }
            }
            //DEBUG_INFO("assigning");
            //sampled_board_cards.assign(remaining_cards.begin(), remaining_cards.begin() + cards_to_deal);
        }

        DEBUG_INFO("sampled board cards: " << sampled_board_cards);

        // Prepare full board
        std::array<int, 5> full_board = board;
        int idx = board_size;
        for (int card : sampled_board_cards) {
            full_board[idx++] = card;
        }
        DEBUG_INFO("prep full board");

        // Collect used cards for this sample
        std::bitset<52> sample_used_cards = used_cards;
        for (int card : sampled_board_cards) {
            sample_used_cards.set(card);
        }
        //DEBUG_INFO("collected used cards, size: " << sample_used_cards);

        // Sample opponent hands without conflicts
        std::vector<int> sampled_opp_hand_indices(NUM_PLAYERS - 1, -1);
        bool valid_sample = true;
        for (int i = 0; i < NUM_PLAYERS - 1; ++i) {
            const auto& cdf = opp_cdfs[i];
            const auto& hand_indices = opp_hand_indices[i];

            // Attempt to sample an opponent hand without conflicts
            bool hand_found = false;
            const int max_retries = 10;
            int retries = 0;
            while (retries < max_retries) {
                // Sample a random value
                double rand_val = dist(rng);
                DEBUG_INFO("rand val: " << rand_val);
                DEBUG_INFO("opp idx: " << i);
                DEBUG_INFO("player idx: " << player);
                DEBUG_INFO("cdf size: " << cdf.size());
                if (cdf.size() == 0) throw std::runtime_error("cdf size is zero");
                DEBUG_INFO("hand indices: " << opp_hand_indices.size());
                //std::cout << "Press Enter to continue...";
                //std::cin.get(); // Waits for the user to press Enter
                //std::cout << "Continuing execution..." << std::endl;
                // Find the corresponding hand index
                auto it = std::lower_bound(cdf.begin(), cdf.end(), rand_val);
                if (it == cdf.end()) {
                    DEBUG_INFO("it == cdf.end()");
                    ++retries;
                    continue;
                }
                int idx_in_cdf = std::distance(cdf.begin(), it);
                int hand_idx = hand_indices[idx_in_cdf];
                const auto& opp_hand = card_lookup_table[hand_idx];
                DEBUG_INFO("sampled opp hand: " << opp_hand[0] << "," << opp_hand[1]);

                // Check for card conflicts with already used cards
                if (!sample_used_cards.test(opp_hand[0]) && !sample_used_cards.test(opp_hand[1])) {
                    // No conflict
                    sample_used_cards.set(opp_hand[0]);
                    sample_used_cards.set(opp_hand[1]);
                    sampled_opp_hand_indices[i] = hand_idx;
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
            DEBUG_INFO("no valid hand found");
            continue;
        }
        DEBUG_INFO("sampled opp hands w/o conflict");

        // Evaluate player's hand
        omp::Hand player_full_hand = omp::Hand::empty();
        for (int card : player_hand) player_full_hand += omp::Hand(card);
        for (int card : full_board) player_full_hand += omp::Hand(card);
        int player_strength = evaluator.evaluate(player_full_hand);
        DEBUG_INFO("eval player hand");

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
        DEBUG_INFO("eval opp hand");

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
        DEBUG_INFO("get winner");

        // Calculate win share for the player
        double player_win_share = 0.0;
        if (std::find(winners.begin(), winners.end(), -1) != winners.end()) {
            player_win_share = 1.0 / winners.size();
        }
        DEBUG_INFO("calc win share");

        total_wins += player_win_share;
        ++valid_samples;
    }

    //DEBUG_INFO("no valid samples");

    if (valid_samples == 0) {
        throw std::runtime_error("No valid samples generated in Monte Carlo simulation.");
    }

    double win_probability = total_wins / valid_samples;
    return win_probability;
}

std::array<torch::Tensor, 4> init_batched_cards(std::array<int, 5> board) {
    // Define shapes for each tensor
    std::vector<int64_t> hand_shape = {EVAL_BS, 2};
    std::vector<int64_t> flop_shape = {EVAL_BS, 3};
    std::vector<int64_t> turn_shape = {EVAL_BS, 1};
    std::vector<int64_t> river_shape = {EVAL_BS, 1};
    
    std::array<torch::Tensor, 4> batched_cards;

    // Initialize hand tensor with placeholder values (-1)
    batched_cards[0] = torch::full(hand_shape, -1, torch::kInt32);

    // Initialize flop tensor
    std::vector<int> flop_cards = {board[0], board[1], board[2]};
    batched_cards[1] = torch::tensor(flop_cards, torch::kInt32).expand(flop_shape);

    // Initialize turn tensor
    int turn_card = board[3];
    batched_cards[2] = torch::full(turn_shape, turn_card, torch::kInt32);

    // Initialize river tensor
    int river_card = board[4];
    batched_cards[3] = torch::full(river_shape, river_card, torch::kInt32);

    return batched_cards;
}

torch::Tensor init_batched_fracs(torch::Tensor batched_fracs) {
    std::vector<int64_t> batched_fracs_shape = {EVAL_BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return batched_fracs.expand(batched_fracs_shape).clone();
}

torch::Tensor init_batched_status(torch::Tensor batched_status) { 
    std::vector<int64_t> batched_status_shape = {EVAL_BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return batched_status.expand(batched_status_shape).clone();
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
    std::array<int, NUM_PLAYERS-1> opp_idxs,
    int player,
    std::array<std::vector<Infoset>, NUM_PLAYERS-1> opp_histories, // list of each infoset opp saw before acting 
    std::array<int, 52> deck_cards
) {
    std::array<std::array<double, MAX_OPP_RANGE>, NUM_PLAYERS-1> opp_ranges{}; 
    std::array<int, 5> curr_board = engine.get_board();
    std::array<int, 52> curr_deck = engine.get_deck();

    for (size_t i=0; i<NUM_PLAYERS-1; ++i) {
        int opp_idx = opp_idxs[i];
        for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
            std::array<int, 2> opp_hand = card_lookup_table[j];
            bool on_board = false;

            // check board for invalids
            for (size_t k=0; k<5; ++k) {
                if (curr_board[k] == opp_hand[0] || curr_board[k] == opp_hand[1]) {
                    on_board = true;
                    break;
                }
            }

            // check other player hand for invalids
            for (size_t k=0; k<NUM_PLAYERS; ++k) {
                // skip ur own hand
                if (k == opp_idx) continue;

                if ((player_hands[k][0] == opp_hand[0] || player_hands[k][0] == opp_hand[1]) || 
                    (player_hands[k][1] == opp_hand[1] || player_hands[k][1] == opp_hand[0])) {
                    if (on_board) continue;
                    else {
                        on_board = true;
                        break;
                    }
                }
            }

            if (!on_board) {
                opp_ranges[i][j] = 1.0;
            } else {
                opp_ranges[i][j] = 0.0;
            }
        }
    }

    // normalize
    for (size_t i=0; i<NUM_PLAYERS-1; ++i) {
        double total_prob = 0.0;
        for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
            total_prob += opp_ranges[i][j];
        }
        DEBUG_INFO("total prob: " << total_prob);
        
        if (total_prob > 0) {
            for (size_t j=0; j<MAX_OPP_RANGE; ++j) {
                opp_ranges[i][j] /= total_prob;
            }
        } else {
            throw std::runtime_error("No valid hands for opponent " + std::to_string(i));
        }
    }

    // init opp ranges 
    // TODO init for those who didn't fold
    for (size_t i=0; i<NUM_PLAYERS-1; ++i) {
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
                    for (size_t a=0; a<NUM_ACTIONS; ++a) {
                        if (strat[a] > max_prob) {
                            max_prob = strat[a];
                            max_act = a;
                        }
                    }
                    opp_ranges[i][j] = curr_hand_prob * strat[max_act];
                }
            }
        }
    }

    DEBUG_INFO("Updated opp ranges");

    // calc first wprollout 
    double wp = wp_rollout_monte_carlo(engine, player, player_hands[player], opp_ranges, opp_idxs, curr_board, curr_deck);

    DEBUG_INFO("wp: " << wp);

    double max_bet = 0.0;
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        if (i != player && engine.players[i].total_bet > max_bet) {
            max_bet = engine.players[i].total_bet;
        }
    }
    double current_bet = engine.players[player].total_bet;
    double asked_amt = max_bet - current_bet;
    double call_util = wp * engine.get_pot() - (1-wp) * asked_amt;

    DEBUG_INFO("call util: " << call_util);

    std::array<float, NUM_ACTIONS> action_utils{};
    action_utils[0] = 0.0;
    action_utils[1] = call_util;

    // init original game state trackers
    Infoset infoset = prepare_infoset(engine, player);
    std::array<torch::Tensor, 4> cards = infoset.cards;

    for (size_t a=2; a<NUM_ACTIONS; ++a) {
        DEBUG_INFO("calc act: " << a << " util");
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
            if (cf_bet_frac[0][j].item<double>() > 0.0) {
                last_bet_idx = j;
                break;
            }
        }

        int last_player_idx = (last_bet_idx % (MAX_ROUND_BETS * NUM_PLAYERS)) % MAX_PLAYERS;
        DEBUG_INFO("last player idx: " << last_player_idx);
        // if player = 0, last_player = 4, max_players = 5
        // if last_player > player, shift = (max_players - last_player + player)
        // else shift = player - last_player
        // if player = 1, last_player = 0, max_players = 3
        int shift = 0;
        if (last_player_idx > player) shift = NUM_PLAYERS - last_player_idx + player;
        else shift = player - last_player_idx;
        int player_bet_idx = last_bet_idx + shift;
        double bet_amt = get_bet_amt(engine, player, a);
        DEBUG_INFO("player_bet_idx: " << player_bet_idx);
        DEBUG_INFO("max_bet_length: " << cf_bet_frac.size(1));
        if (player_bet_idx < MAX_ROUND_BETS*NUM_PLAYERS*4) {
            // must index in at 0 due to shape [1, total_bets]
            cf_bet_frac[0][player_bet_idx] = bet_amt;
            cf_bet_status[0][player_bet_idx] = (bet_amt > 0 ? 1 : 0);
        } else throw std::runtime_error("shift calc is incorrect");

        DEBUG_INFO("calc shift");

        // prepare for mc_rollouts of game after this act 
        std::array<std::vector<double>, NUM_PLAYERS - 1> opp_cdfs;
        std::array<std::vector<int>, NUM_PLAYERS - 1> opp_hand_indices;
        int live_opps = 0;
        for (int i = 0; i < NUM_PLAYERS - 1 ; ++i) {
            // get idx in engine.players
            int opp_idx = opp_idxs[i];
            if (opp_idx == player) {
                throw std::runtime_error("Somehow, player is in opp_idx");
            };

            const auto& opp_range = opp_ranges[i];
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

            opp_cdfs[i] = cdf;
            opp_hand_indices[i] = hand_indices;
        }

        // fold percent of opp if player takes action a
        double fp = 0.0;
        int batch_idx = 0;

        // batch tensors
        std::array<int, EVAL_BS> batched_hand_indices{}; 
        std::array<torch::Tensor, 4> batched_cards = init_batched_cards(curr_board);
        DEBUG_INFO("check batched cards at init");
        DEBUG_INFO(batched_cards[1][0]);
        torch::Tensor batched_bet_fracs = init_batched_fracs(infoset.bet_fracs);
        torch::Tensor batched_bet_status = init_batched_status(infoset.bet_status);

        // Initialize random engine
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        std::bitset<52> board_cards;
        for (int card : engine.get_board()) {
            if (card != NULL_CARD) board_cards.set(card);
        }
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
            DEBUG_INFO("begin filling batches");
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
                DEBUG_INFO("batch_idx: " << idx);
                // hand is still player hand, so update to this opp's hand
                DEBUG_INFO(batched_cards[0][idx].sizes());
                DEBUG_INFO("[2]");
                batched_cards[0][idx][0] = opp_hand[0];
                batched_cards[0][idx][1] = opp_hand[1];
                DEBUG_INFO("fill cards");

                // todo this can just be initialized in beginning without change
                DEBUG_INFO(batched_bet_fracs[idx].sizes());
                DEBUG_INFO(cf_bet_frac.squeeze().sizes());
                batched_bet_fracs[idx] = cf_bet_frac.squeeze().clone();
                DEBUG_INFO(batched_bet_status[idx].sizes());
                DEBUG_INFO(batched_bet_status[idx].sizes());
                batched_bet_status[idx] = cf_bet_status.squeeze().clone();
                DEBUG_INFO("fill bets");

                // store hand idx for convenience
                batched_hand_indices[idx] = opp_hand_index;
                DEBUG_INFO("fill ts");
            }

            batch_idx++;

            // do batched inference and update counterfactual opponent ranges 
            if (batch_idx * live_opps == EVAL_BS) {
                auto batched_logits = deep_cfr_model_forward(policy_net, batched_cards, batched_bet_fracs, batched_bet_status);
                for (size_t b = 0; b < EVAL_BS; ++b) {
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
        double wp = wp_rollout_monte_carlo(engine, player, player_hands[player], cf_opp_ranges, opp_idxs, curr_board, curr_deck);

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

    // avg mbb winnings
    double total_mbb = 0.0;

    DEBUG_INFO("begin eval mc simulations");
    for (size_t i = 0; i < EVAL_MC_SAMPLES; ++i) {
        // init history
        std::array<std::vector<Infoset>, NUM_PLAYERS> all_histories{};
        for (auto& history : all_histories) {
            history.reserve(MAX_ROUND_BETS * 4);
        }
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

        while (engine.get_game_status() && engine.is_playing(player)) {
            if (engine.turn() == player) {
                DEBUG_INFO("Player's turn");
                Infoset I = prepare_infoset(engine, player);
                all_histories[player].push_back(I);
                torch::Tensor logits = deep_cfr_model_forward(policy_net, I.cards, I.bet_fracs, I.bet_status);
                std::array<double, NUM_ACTIONS> strat = regret_match(logits);
                int act = 0;
                double max_prob = 0.0;
                for (size_t a=0; a<NUM_ACTIONS; ++a) {
                    if (strat[a] > max_prob) {
                        max_prob = strat[a];
                        act = a;
                    }
                }
                DEBUG_INFO("Player sampled act: " << act);
                  // Verify and adjust action if necessary
                while (!verify_action(engine, player, act)) {
                    act = (act - 1) % NUM_ACTIONS;
                }
                // Take action
                take_action(engine, player, act);
                DEBUG_INFO("I selected action: " << act);
            } else {
                int opp = engine.turn();
                DEBUG_INFO("Opp " << opp << " turn");
                Infoset I = prepare_infoset(engine, player);
                all_histories[opp].push_back(I);

                std::array<std::vector<Infoset>, NUM_PLAYERS-1> opp_histories{};
                std::array<int, NUM_PLAYERS-1> opp_idxs{};
                for (size_t i=0; i<NUM_PLAYERS;++i) {
                    // opp is 0
                    if (i != opp) {
                        // so i = 1
                        if (i > opp) {
                            // opp_histories[0] = all
                            // opp_idxs[0] = 0
                            opp_histories[i-1] = all_histories[i];
                            opp_idxs[i-1] = i;
                        } else {
                            opp_histories[i] = all_histories[i];
                            opp_idxs[i] = i;
                        }
                    }
                }

                DEBUG_INFO("opp_histories made");

                int act = get_lbr_act(
                    engine, 
                    policy_net, 
                    player_hands, 
                    opp_idxs,
                    opp, 
                    opp_histories, 
                    engine.get_deck()
                );

                DEBUG_INFO("Lbr chosen act: " << act);

                while (!verify_action(engine, opp, act)) {
                    act = (act - 1) % NUM_ACTIONS;
                }
                // Take action
                take_action(engine, opp, act);
                DEBUG_INFO("Opp selected action: " << act); 
            }
        }

        total_mbb += (engine.get_payoffs()[player] / big_bet) * 1000;
        // calculate the player's winnings
    }

    double avg_mbb = total_mbb / EVAL_MC_SAMPLES;
    DEBUG_NONE("Policy net is evaluated at: " << avg_mbb << "mbb");
    return avg_mbb;
}
*/