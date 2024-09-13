#include "engine.h"
#include <iostream>

const int PokerEngine::MAX_PLAYERS;
const int PokerEngine::BOARD_SIZE;

PokerEngine::PokerEngine(std::vector<double> starting_stacks, int n_players, double small_blind, double big_blind, int max_round_bets)
    : 
      n_players(n_players),
      small_blind(small_blind),
      big_blind(big_blind),
      max_round_bets(max_round_bets),
      round(0),
      actor(0),
      bet_idx(0),
      pot(0),
      game_status(true)
{
    if (this->n_players < 2 || this->n_players > MAX_PLAYERS) {
        throw std::invalid_argument("Invalid number of players");
    }

    if (this->small_blind < 0 || this->big_blind < 0) {
        throw std::invalid_argument("Blinds must be non-negative");
    }

    if (this->small_blind * 2 != this->big_blind) {
        throw std::invalid_argument("Big blind must be twice the small blind");
    }

    if (this->n_players != starting_stacks.size()) {
        throw std::invalid_argument("starting_stacks size must equal n_players"); 
    }
    
    // Initialize players
    for (int i = 0; i < this->n_players; ++i) {
        this->players[i].stack = starting_stacks[i];  // stack will be set later, no bet, in hand
    }
    
    // Initialize other member variables
    int total_bets = this->max_round_bets * this->n_players * 4;
    this->bet_status.resize(total_bets, 0);
    this->bet_history_raw.resize(total_bets, 0.0);
    this->bet_history_frac.resize(total_bets, 0.0);
    this->status.resize(this->n_players, true);
    std::fill(this->payoffs.begin(), this->payoffs.end(), 0.0);
    std::fill(this->board.begin(), this->board.end(), 0);

    // Initialize blinds
    this->players[0].stack -= this->small_blind;
    this->players[0].bet = this->small_blind;
    this->players[1].stack -= this->big_blind;
    this->players[1].bet = this->big_blind;
    this->pot = this->small_blind + this->big_blind;

    // Set bet_status, bet_history_raw, and bet_history_frac for blinds
    this->bet_status[0] = true;  
    this->bet_status[1] = true;  
    this->bet_history_raw[0] = this->small_blind;
    this->bet_history_raw[1] = this->big_blind;
    this->bet_history_frac[0] = this->small_blind / this->players[0].stack;
    this->bet_history_frac[1] = this->big_blind / this->players[1].stack;

    // deal player hands 
    std::set<int> chosen_hands;
    std::mt19937 rng(std::random_device{}());

    for (size_t player = 0; player < this->n_players; ++player) {
        for (size_t _ = 0; _ < 2; ++_) {
            std::vector<int> remaining_deck;
            for (int card = 0; card < 52; ++card) {
                if (chosen_hands.find(card) == chosen_hands.end()) {
                    remaining_deck.push_back(card);
                }
            }
            std::uniform_int_distribution<> dist(0, remaining_deck.size() - 1);
            int card = remaining_deck[dist(rng)];
            chosen_hands.insert(card);
            this->players[player].hand += Hand(card);
        }
    }    

    // initialize deck
    this->deck.clear();
    for (size_t card = 0; card < 52; ++card) {
        if (chosen_hands.find(card) == chosen_hands.end()) {
            this->deck.push_back(card);
        }
    }
}

/* 
    PUBLIC ACTIONS 

    - on entering fn, we assume bet_idx was correctly incremented
    - on exiting fn, we call next_state to perform state transition
*/

void PokerEngine::fold(int player) {
    if (player != this->actor) {
        throw std::runtime_error("Player mismatch: folding player is not the current actor");
    }
    this->players[player].in_hand = false;
    this->status[player] = false;
    this->pot += this->players[player].bet;
    this->players[player].folded = true;
    this->bet_idx++;

    move_action();
}

// side pot:
// when a player calls or bets when 
// their total stack < min_bet_amt 
void PokerEngine::bet_or_raise(int player, int amount) {
    // valid if player.bet + amount > agressor.bet 
    if (player != this->actor) {
        throw std::runtime_error("Player mismatch: folding player is not the current actor");
    }

    double min_bet_amt = calc_min_bet_amt(player);
    int bet_idx = this->bet_idx;

    // side pot
    if (!verify_sufficient_funds(player, min_bet_amt) 
        && amount == this->players[player].stack ) {
        this->players[player].stack -= amount;
        this->players[player].bet += amount;
        double old_pot = this->pot;
        this->pot += amount;
        this->bet_status[bet_idx] = true;
        this->bet_history_raw[bet_idx] = amount;
        this->bet_history_frac[bet_idx] = amount/old_pot;

        next_state();
        return;
    }

    if (should_force_check_or_call(player)) {
        // if we have to force someone to check when they 
        // wanted to do a valid bet, it means they were gonna raise
        // and NOT fold. therefore we are safe to proceed 
        if (amount + this->players[player].bet < min_bet_amt) {
            // Something is wrong, but we won't raise an exception
            std::cerr << "Warning: Player " << player << " attempted to bet less than the minimum required amount." << std::endl;
            fold(player);
        } else {
            check_or_call(player);
        }
        return;
    }

    this->players[player].stack -= amount;
    this->players[player].bet += amount;
    double old_pot = this->pot;
    this->pot += amount;

    this->bet_status[bet_idx] = true;
    this->bet_history_raw[bet_idx] = amount;
    this->bet_history_frac[bet_idx] = amount/old_pot;

    next_state();
}

void PokerEngine::check_or_call(int player) {
    if (player != this->actor) {
        throw std::runtime_error("Player mismatch: folding player is not the current actor");
    }
    // get the min raise amt
    double min_bet_amt = calc_min_bet_amt(player);
    double call_amt = min_bet_amt - this->players[player].bet;
    this->players[player].stack -= call_amt;
    this->players[player].bet += call_amt;
    double old_pot = this->pot;
    this->pot += call_amt;

    int bet_idx = this->bet_idx;
    this->bet_status[bet_idx] = call_amt > 0;
    this->bet_history_raw[bet_idx] = call_amt;
    this->bet_history_frac[bet_idx] = call_amt/old_pot;

    next_state();
}

// get the min amt player's bet must be to keep playing
double PokerEngine::calc_min_bet_amt(int player) {
    double min_bet = this->players[player].bet;
    for (size_t i=0; i < this->n_players; ++i) {
        if (i != player && this->players[i].in_hand) {
            if (this->players[i].bet > min_bet) min_bet = this->players[i].bet;
        }
    }  
    return min_bet;
}

bool PokerEngine::verify_min_raise(int player, double amount) const {
    if (!verify_sufficient_funds(player, amount)) return false;
    return amount + this->players[player].bet >= calc_min_bet_amt(player); 
}

bool PokerEngine::verify_sufficient_funds(int player, double amount) const {
    if (this->players[player].stack < amount) return false;
    return true;
}

void PokerEngine::reset() {
    // TODO: Implement
}

/*
PokerEngine PokerEngine::copy() const {
    // TODO: Implement
    return PokerEngine(0, 0);  // Placeholder return
}
*/

/* 
    PRIVATE GAME FLOW 

    the main state transition function is next_state,
    all other functions are utilities for it

*/
void PokerEngine::deal_cards() {
    std::random_device rd;
    std::mt19937 rng(rd());

    switch (this->round) {
        case 1: // Flop
            for (int i = 0; i < 3; ++i) {
                std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
                int idx = dist(rng);
                this->board[i] = this->deck[idx];
                this->deck.erase(this->deck.begin() + idx);
            }
            break;
        case 2: // Turn
            {
                std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
                int idx = dist(rng);
                this->board[3] = this->deck[idx];
                this->deck.erase(this->deck.begin() + idx);
            }
            break;
        case 3: // River
            {
                std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
                int idx = dist(rng);
                this->board[4] = this->deck[idx];
                this->deck.erase(this->deck.begin() + idx);
            }
            break;
        default:
            throw std::runtime_error("Invalid round for dealing cards");
    }

    this->round++;
}

bool PokerEngine::is_everyone_all_in() const {
    int bet_idx = this->bet_idx;
    bool all_all_in = true;

    for (size_t i = 0; i < this->n_players; ++i) {
        if (this->players[i].in_hand 
            && this->players[i].stack != 0.0) all_all_in = false;
    }

    return all_all_in;
}

bool PokerEngine::is_round_complete() const {
    // round is over if every bet status between 
    // current actor's current bet (inclusive)   
    // and current actor's last bet (exclusive) is false.
    // everyone either checked or folded.
    int bet_idx = this->bet_idx; 
    bool complete = true;
    for (size_t i = 0; i < this->n_players-1; ++i) {
        if (this->bet_status[bet_idx-i]) complete = false;
    }
    return complete;
}

void PokerEngine::next_state() {
    if (is_everyone_all_in()) {
        showdown();
        return;
    }

    if (is_round_complete()) {
        if (this->round == 4) {
            showdown();
            return;
        } 
        deal_cards();
    }

    int curr_actor = this->actor;
    bool live_player_exists = false;
    this->bet_idx++;
    for (size_t i=curr_actor+1; i<curr_actor+this->n_players; ++i) {
        if (!this->status[i % this->n_players]) {
            // skip bet idxs of dead players
            this->bet_idx++;
        } else {
            // first live player found is actor 
            live_player_exists = true;
            this->actor = i % this->n_players;
            break;
        }
    }

    // if game ended with this player folding
    if (!live_player_exists) {
        for (size_t i=0; i<this->n_players; ++i) {
            if (this->players[i].in_hand && !this->players[i].folded) {
                this->payoffs[i] = this->pot;
                this->players[i].stack += this->pot;
                break;
            }
        }
        this->game_status = false;    
    }
}

// case 1: round 4 ended
// case 2: all live players all-ined
void PokerEngine::showdown() {
    // get all live players + side pot, 
    // evaluate each of their hands against board,
    // get the winner(s), then distribute to payoffs

    // deal remaining cards at showdown
    if (this->board.size() < 5) {
        std::mt19937 rng(std::random_device{}());
        while (this->board.size() < 5) {
            std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
            int idx = dist(rng);
            uint8_t card = this->deck[idx];
            this->board[this->board.size()] = card;
            this->deck.erase(this->deck.begin() + idx);
        }
    }

    // populate player hand strengths
    std::vector<std::tuple<int, int>> hand_strengths;

    for (size_t i=0; i < this->n_players; ++i) {
        if (this->players[i].in_hand || !this->players[i].folded) {
            // add board card to live player hands
            for (size_t j=0; j < this->board.size(); ++j) {
                this->players[i].hand += this->board[j];
            }
            // eval strength
            int hand_strength = evaluator.evaluate(this->players[i].hand);
            // add to vec
            hand_strengths.push_back({i, hand_strength});
        }
    }

    // sort the vector, descending order
    std::sort(vec.begin(), vec.end(),
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
    });

    if (hand_strengths.size() < 2) {
        throw std::runtime_error("must be at least 2 live players or side pot players at showdown");
    }

    // out of n live players,
    // there can be (n-x) pot winners, and x side pot winners worst case
    std::vector<int> pot_winners; 
    std::vector<int> side_pot_winners;
    int winner = std::get<1>(hand_strengths[0]);

    // side pot
    if (!this->players[winner].in_hand) {
        int next = winner + 1;
        while (next < hand_strengths.size() && 
            std::get<1>(hand_strengths[++next]) == std::get<1>(hand_strengths[0])) {
            int player = std::get<0>(hand_strengths[next]);
            if (!this->players[player].in_hand) {
                // case tie with someone w side pot
                side_pot_winners.push_back(player);
            } else {
                // case tie with someone in main pot
                pot_winners.push_back(player);
            }
        }
    } else {
        // main pot wins, ignore side pot
        int next = winner + 1;
        while (next < hand_strenghts.size() && 
            std::get<1>(hand_strengths[++next]) == std::get<1>(hand_strenghts[0])) {
            int player = std::get<0>(hand_strengths[next]);
            pot_winners.push_back(player); 
        } 
    }

    // side pot payoff calc: https://en.wikipedia.org/wiki/Betting_in_poker
    // scenario A sb 5, B bb 10, C bet 15, A fold, B all in 10, C check 
    // pot = 40
    // B wins side, C wins main:
    // B gets 5 + 10 + 10
    // C wins 5 + 10
    for (size_t i=0; i<side_pot_winners.size(); ++i) {
        int player = std::get<0>(side_pot_winners[i]);
        double entitled = this->players[player].bet; 
        double payoff = entitled;
        for (size_t j=0; j<this->n_players; ++j) { 
            if (j != player) {
                payoff += this->players[j].bet <= entitled ? this->players[j].bet : entitled;
            } 
        }
        this->payoffs[player] = payoff;
        this->pot -= payoff;
        this->players[player].stack += payoff;
    }
        
    // split remaining pot evenly between pot_winners
    double split_pot = this->pot / static_cast<double>(pot_winners.size());
    for (size_t i=0; i<pot_winners.size(); ++i) {
        int player = std::get<0>(pot_winners[i]);
        this->payoffs[player] = split_pot;
        this->players[player].stack += split_pot;
    }

    this->game_status = false;
}


/* PUBLIC VERIFICATIONS  */

bool PokerEngine::should_force_check_or_call() const {
    int bets_per_round = this->n_players * this->max_round_bets;
    // ex: bet_idx = 9; bets_per_round = 12; n_players = 3
    // then 12-9 <= 3, meaning 9, 10, 11 should force check or call 
    if (bets_per_round - (this->bet_idx % bets_per_round) <= this->n_players) {
        return true;
    }     
    return false;
}

/* PUBLIC QUERIES */ 

void PokerEngine::set_stack(std::vector<double> stack) {
    
}

bool PokerEngine::is_in(int player) const {
    // TODO: Implement
    return false;
}

int PokerEngine::turn() const {
    // TODO: Implement
    return 0;
}

