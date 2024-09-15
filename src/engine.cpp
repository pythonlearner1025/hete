#include "engine.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>

PokerEngine::PokerEngine(
    std::vector<double> starting_stacks, 
    std::vector<double> antes, 
    int actor, 
    int n_players, 
    double small_blind, 
    double big_blind, 
    bool manual
    )
    : 
      n_players(n_players),
      small_blind(small_blind),
      big_blind(big_blind),
      round(0),
      actor(actor),
      pot(0.0),
      game_status(true),
      manual(manual)
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
    for (int i = 0; i < n_players; ++i) {
        this->players[i].stack = starting_stacks[i];
        this->players[i].status = PlayerStatus::Playing;
        // Initialize bets_per_round for each player
        for (int r = 0; r < 4; ++r) {
            this->players[i].bets_per_round[r].reserve(MAX_ROUND_BETS);
        }
    }

    // Collect Antes
    for (int i = 0; i < n_players; ++i) {
        double ante = antes[i];
        if (ante > 0) {
            this->players[i].stack -= ante;
            this->pot += ante;
        }
    }
    
    // Initialize payoffs
    std::fill(this->payoffs.begin(), this->payoffs.end(), 0.0);

    // Initialize blinds
    this->players[0].stack -= small_blind;
    this->players[0].total_bet += small_blind;
    this->players[0].bets_per_round[0].push_back(small_blind);

    this->players[1].stack -= big_blind;
    this->players[1].total_bet += big_blind;
    this->players[1].bets_per_round[0].push_back(big_blind);

    this->pot += (small_blind + big_blind);

    // Initialize and shuffle the deck once
    if (!manual) {
        std::vector<int> deck(52);
        std::iota(deck.begin(), deck.end(), 0);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(deck.begin(), deck.end(), rng);

        // Deal cards to players
        size_t card_index = 0;
        for (size_t player = 0; player < this->n_players; ++player) {
            this->players[player].hand[0] = deck[card_index++];
            this->players[player].hand[1] = deck[card_index++];
        }
        // Remaining deck starts from card_index
        this->deck.assign(deck.begin() + card_index, deck.end());
    }
}

void PokerEngine::reset_actions() {
    for (size_t i=0; i<this->n_players; ++i) {
        if (this->players[i].status == PlayerStatus::Playing) {
            this->players[i].acted = false;
        }
    }
}
/* 
    PUBLIC ACTIONS 
*/

bool PokerEngine::can_fold(int player) const {
    return true;
}

void PokerEngine::fold(int player) {
    if (player != this->actor) {
        std::cout << "Player mismatch: folding player " + std::to_string(player) + " is not the current actor " + std::to_string(this->actor) << std::endl;
        throw std::runtime_error("Player mismatch: folding player " + std::to_string(player+1) + " is not the current actor " + std::to_string(this->actor+1));
    }
    this->players[player].status = PlayerStatus::Folded;
    // No need to adjust pot or bets here since the player forfeits their current bet
    next_state();
}

bool PokerEngine::can_bet_or_raise(int player, double amount) const {
    return !verify_sufficient_funds(player, amount);
}
// side pot:
// when a player calls or bets when 
// their total stack < min_bet_amt 
void PokerEngine::bet_or_raise(int player, double amount) {
    if (player != this->actor) {
        throw std::runtime_error("Player mismatch: betting player " + std::to_string(player + 1) + " is not the current actor " + std::to_string(this->actor+1));
    }

    double min_bet_amt = calc_min_bet_amt(player);

    // Handle all-in scenarios
    if (!verify_sufficient_funds(player, min_bet_amt) && amount == this->players[player].stack) {
        double bet_amount = this->players[player].stack;
        this->players[player].stack = 0.0;
        this->players[player].total_bet += bet_amount;
        this->players[player].bets_per_round[this->round].push_back(bet_amount);
        this->pot += bet_amount;
        this->players[player].status = PlayerStatus::AllIn;
        reset_actions();
        this->players[this->actor].acted = true;
        next_state();
        return;
    }

    // Validate bet amount
    if (amount + this->players[player].total_bet < min_bet_amt) {
        std::cerr << "Warning: Player " << player << " attempted to bet less than the minimum required amount." << std::endl;
        fold(player);
        return;
    }

    // Process the bet
    this->players[player].stack -= amount;
    this->players[player].total_bet += amount;
    this->players[player].bets_per_round[this->round].push_back(amount);
    this->pot += amount;

    if (this->players[player].stack == 0) {
        this->players[player].status = PlayerStatus::AllIn;
    }
    reset_actions();
    this->players[this->actor].acted = true;
    next_state();
}

bool PokerEngine::can_check_or_call(int player) const {
    return true;
}

void PokerEngine::check_or_call(int player) {
    if (player != this->actor) {
        throw std::runtime_error("Player mismatch: checking/calling player " + std::to_string(player+1) + " is not the current actor " + std::to_string(this->actor+1));
    }
    double min_bet_amt = calc_min_bet_amt(player);
    double call_amt = min_bet_amt - this->players[player].total_bet;

    // Handle all-in if the player can't cover the call amount
    if (call_amt > this->players[player].stack) {
        call_amt = this->players[player].stack;
        this->players[player].status = PlayerStatus::AllIn;
    }

    if (call_amt > 0.0) {
        this->players[player].stack -= call_amt;
        this->players[player].total_bet += call_amt;
        this->players[player].bets_per_round[this->round].push_back(call_amt);
        this->pot += call_amt;
    } else {
        // Player checks; record zero bet
        this->players[player].bets_per_round[this->round].push_back(0.0);
    }
    
    this->players[player].acted = true;
    next_state();
}


// get the min amt player's bet must be to keep playing
double PokerEngine::calc_min_bet_amt(int player) const {
    double min_bet = 0.0;
    for (size_t i = 0; i < this->n_players; ++i) {
        if (i != player && (this->players[i].status == PlayerStatus::Playing || this->players[i].status == PlayerStatus::AllIn)) {
            if (this->players[i].total_bet > min_bet) min_bet = this->players[i].total_bet;
        }
    }  
    return min_bet;
}


bool PokerEngine::verify_min_raise(int player, double amount) const {
    if (!verify_sufficient_funds(player, amount)) return false;
    return amount + this->players[player].total_bet >= calc_min_bet_amt(player); 
}

bool PokerEngine::verify_sufficient_funds(int player, double amount) const {
    if (this->players[player].stack < amount) return false;
    return true;
}

void PokerEngine::reset() {
    // TODO: Implement
}


/* 
    PRIVATE GAME FLOW 

    the main state transition function is next_state,
    all other functions are utilities for it

*/
void PokerEngine::deal_cards() {

    if (this->manual) {
        this->round++;
        reset_actions();
        return;
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    switch (this->round) {
        case 0: // Flop
            for (int i = 0; i < 3; ++i) {
                std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
                int idx = dist(rng);
                uint8_t card = this->deck[idx];
                this->board.push_back(card);
                this->deck.erase(this->deck.begin() + idx);
            }
            break;
        case 1: // Turn
            {
                std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
                int idx = dist(rng);
                uint8_t card = this->deck[idx];
                this->board.push_back(card);
                this->deck.erase(this->deck.begin() + idx);
            }
        case 2: // River
            {
                std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
                int idx = dist(rng);
                uint8_t card = this->deck[idx];
                this->board.push_back(card);
                this->deck.erase(this->deck.begin() + idx);
            }
            break;
        default:
            throw std::runtime_error("Invalid round for dealing cards");
    }

    this->round++;
    reset_actions();
}

bool PokerEngine::is_everyone_all_in() const {
    bool all_all_in = true;

    for (size_t i = 0; i < this->n_players; ++i) {
        if (this->players[i].status == PlayerStatus::Playing) {
            all_all_in = false;
            break;
        }
    }

    return all_all_in;
}

double PokerEngine::get_current_max_bet() const {
    double max_bet = 0.0;
    for (int i = 0; i < n_players; ++i) {
        if (players[i].status == PlayerStatus::Playing || players[i].status == PlayerStatus::AllIn) {
            if (players[i].total_bet > max_bet) {
                max_bet = players[i].total_bet;
            }
        }
    }
    return max_bet;
}

bool PokerEngine::is_round_complete() const {

    // if even one alive player hasn't acted, round isn't complete
    for (size_t i = 0; i < this->n_players; ++i) {
        const auto& player = this->players[i];

        if (player.status == PlayerStatus::Playing && !player.acted) {
            return false;
        }
        //std::cout << "Player " << i << ": Status = " << static_cast<int>(player.status) 
        //         << ", Acted = " << (player.acted ? "true" : "false") << std::endl;
    }

    return true;
}

void PokerEngine::next_state() {
    int n_live_players = 0;
    int live_player;

    for (int i = 0; i < n_players; ++i){
        if (this->players[i].status == PlayerStatus::Playing || 
            this->players[i].status == PlayerStatus::AllIn) {
            n_live_players++;
            live_player = i;
        }
    }

    if (n_live_players == 1) {
        std::cout << "Only 1 live player remaining, ending game" << std::endl;
        // All players are folded or all-in
        this->payoffs[live_player] = this->pot;
        this->players[live_player].stack += this->pot;
        this->game_status = false;
        return;
    }

    if (is_everyone_all_in()) {
        std::cout << "everyone is all in, proceed to showdown";
        showdown();
        return;
    }

    if (is_round_complete()) {
        std::cout << "Round " << this->round << " is complete." << std::endl;
        if (this->round == 3 && !this->manual) {
            showdown();
            return;
        } else {
            // Proceed to next betting round
            deal_cards();
        }
        this->actor = 0;
        if (this->players[0].status == PlayerStatus::Playing) return;
    }

    // Find the next player to act
    int curr_actor = this->actor;
    std::cout << "Current actor: Player " << curr_actor+1 << std::endl;

    for (int i = 1; i <= n_players; ++i) {
        int next_player = (curr_actor + i) % n_players;
        if (players[next_player].status == PlayerStatus::Playing) {
            this->actor = next_player;
            std::cout << "New actor: Player " << next_player+1 << std::endl;
            break;
        } else {
            std::cout << "Skipped Player " << next_player+1 << " (Status: ";
            switch (players[next_player].status) {
                case PlayerStatus::Folded:
                    std::cout << "Folded";
                    break;
                case PlayerStatus::AllIn:
                    std::cout << "All-In";
                    break;
                case PlayerStatus::Out:
                    std::cout << "Out";
                    break;
                default:
                    std::cout << "Unknown";
            }
            std::cout << ")" << std::endl;
        }
    }
}


// case 1: round 4 ended
// case 2: all live players all-ined
void PokerEngine::showdown() {
    std::cout << "Debug: Entering showdown()" << std::endl;
    if (!this->game_status) {
        std::cout << "showdown already triggered, skipping" << std::endl;
        return;
    }
    // get all live players + side pot, 
    // evaluate each of their hands against board,
    // get the winner(s), then distribute to payoffs

    std::cout << "Debug: begin deal remaining cards at showodwn" << std::endl;
    // deal remaining cards at showdown
    if (this->board.size() < 5) {
        std::mt19937 rng(std::random_device{}());
        while (this->board.size() < 5) {
            std::uniform_int_distribution<> dist(0, this->deck.size() - 1);
            int idx = dist(rng);
            uint8_t card = this->deck[idx];
            this->board.push_back(card);
            this->deck.erase(this->deck.begin() + idx);
        }
    }

    std::cout << "Debug: Board size after dealing: " << this->board.size() << std::endl;

    // populate player hand strengths
    std::vector<std::tuple<int, int>> hand_strengths;
    omp::HandEvaluator evaluator;
    for (size_t i=0; i < this->n_players; ++i) {
        if (this->players[i].status == PlayerStatus::Playing || this->players[i].status == PlayerStatus::AllIn) {
            omp::Hand hand = omp::Hand::empty();
            // add hole cards 
            hand += omp::Hand(this->players[i].hand[0]);
            hand += omp::Hand(this->players[i].hand[1]);
            // add board card 
            for (size_t j=0; j < this->board.size(); ++j) {
                hand += omp::Hand(this->board[j]);
            }
            // eval strength
            int hand_strength = evaluator.evaluate(hand);
            // add to vec
            hand_strengths.push_back({i, hand_strength});
        }
    }
    std::cout << "Debug: Number of players with evaluated hands: " << hand_strengths.size() << std::endl;

    // sort the vector, descending order
    std::sort(hand_strengths.begin(), hand_strengths.end(),
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
    });
    std::cout << "Debug: Hand strengths sorted" << std::endl;

    if (hand_strengths.size() < 2) {
        throw std::runtime_error("must be at least 2 live players or side pot players at showdown");
    }

    // out of n live players,
    // there can be (n-x) pot winners, and x side pot winners worst case
    std::vector<int> pot_winners; 
    std::vector<int> side_pot_winners;
    int winner = std::get<0>(hand_strengths[0]);

    // side pot
    if (this->players[winner].status == PlayerStatus::AllIn) {
        int next = winner + 1;
        while (next < hand_strengths.size() && 
            std::get<1>(hand_strengths[++next]) == std::get<1>(hand_strengths[0])) {
            int player = std::get<0>(hand_strengths[next]);
            if (this->players[player].status == PlayerStatus::AllIn) {
                // case tie with someone w side pot
                side_pot_winners.push_back(player);
            } else {
                // case tie with someone in main pot
                pot_winners.push_back(player);
            }
        }
    } else {
        // main pot wins, ignore side pot
        pot_winners.push_back(winner);
        int next = winner + 1;
        while (next < hand_strengths.size() && 
            std::get<1>(hand_strengths[++next]) == std::get<1>(hand_strengths[0])) {
            int player = std::get<0>(hand_strengths[next]);
            pot_winners.push_back(player); 
        } 
    }

    std::cout << "Debug: Number of pot winners: " << pot_winners.size() << ", Number of side pot winners: " << side_pot_winners.size() << std::endl;

    // side pot payoff calc: https://en.wikipedia.org/wiki/Betting_in_poker
    // scenario A sb 5, B bb 10, C bet 15, A fold, B all in 10, C check 
    // pot = 40
    // B wins side, C wins main:
    // B gets 5 + 10 + 10
    // C wins 5 + 10
    for (size_t i=0; i<side_pot_winners.size(); ++i) {
        int player = side_pot_winners[i];
        double entitled = this->players[player].total_bet; 
        double payoff = entitled;
        for (size_t j=0; j<this->n_players; ++j) { 
            if (j != player) {
                payoff += this->players[j].total_bet <= entitled ? this->players[j].total_bet : entitled;
            } 
        }
        this->payoffs[player] = payoff;
        this->pot -= payoff;
        this->players[player].stack += payoff;
    }
    std::cout << "Debug: Side pot payoffs calculated. Remaining pot: " << this->pot << std::endl;
        
    // split remaining pot evenly between pot_winners
    double split_pot = this->pot / static_cast<double>(pot_winners.size());
    for (size_t i=0; i<pot_winners.size(); ++i) {
        int player = pot_winners[i];
        this->payoffs[player] = split_pot;
        this->players[player].stack += split_pot;
    }
    std::cout << "Debug: Main pot split. Each winner receives: " << split_pot << std::endl;

    this->game_status = false;
    std::cout << "Debug: Game status set to false" << std::endl;
}


/* PUBLIC VERIFICATIONS  */

bool PokerEngine::should_force_check_or_call() const {
    std::cout << "Debug: Checking if should force check or call for Player " << (actor + 1) << std::endl;
    std::cout << "Debug: Current round: " << this->round << ", Max round bets: " << MAX_ROUND_BETS << std::endl;
    std::cout << "Debug: Player's bets this round: " << this->players[actor].bets_per_round[this->round].size() << std::endl;

    int actor = this->actor;
    if (this->players[actor].bets_per_round[this->round].size() == MAX_ROUND_BETS - 1) {
        return true;
    }     
    return false;
}

/* PUBLIC QUERIES */ 

std::array<int, 5> PokerEngine::get_board() const {
    std::array<int, 5> ret;
    for (size_t i = 0; i < 5; ++i) {
        if (i < this->board.size()) {
            ret[i] = this->board[i];
        } else {
            ret[i] = -1;
        }
    }
    return ret;
}

std::pair<std::array<bool, PokerEngine::MAX_PLAYERS * 4 * PokerEngine::MAX_ROUND_BETS>, std::array<double, PokerEngine::MAX_PLAYERS * 4 * PokerEngine::MAX_ROUND_BETS>> PokerEngine::construct_history() const {
    constexpr int total_bets = 4 * MAX_PLAYERS * MAX_ROUND_BETS;

    std::array<bool, total_bets> bet_status{};
    std::array<double, total_bets> bet_fracs{};
    std::fill(bet_status.begin(), bet_status.end(), false);
    std::fill(bet_fracs.begin(), bet_fracs.end(), 0.0);

    int index = 0;
    double pot = this->small_blind + this->big_blind;

    for (int r = 0; r < 4; ++r) {
        for (uint b = 0; b < MAX_ROUND_BETS * this->n_players; ++b) {
            uint p = b % this->n_players; 
            uint n_bet = static_cast<uint>(std::floor(static_cast<double>(b) / this->n_players));

            if (this->players[p].bets_per_round[0].size() == 0) {
                continue;
            }

            if (n_bet < this->players[p].bets_per_round[r].size()) {
                double bet_amount = this->players[p].bets_per_round[r][n_bet]; 

                bet_status[b] = (bet_amount > 0.0) ? 1 : 0;
                bet_fracs[b] = bet_amount / pot;

                pot += bet_amount;
            }
        }
    }

    return std::make_pair(bet_status, bet_fracs);
}

/* MANUAL MODE */

// Manual Dealing: Assign a Specific Hand to a Player
void PokerEngine::manual_deal_hand(int player, std::array<int, 2> hand) {
    if (!manual) {
        throw std::runtime_error("Cannot deal manually when not in manual mode.");
    }

    if (player < 0 || player >= n_players) {
        throw std::invalid_argument("Invalid player index.");
    }

       // Assign the new hand to the player
    this->players[player].hand[0] = hand[0];
    this->players[player].hand[1] = hand[1];
}

// Manual Dealing: Assign Specific Cards to the Board
void PokerEngine::manual_deal_board(const std::array<int, 5> board_cards) {
    if (!manual) {
        throw std::runtime_error("Cannot deal manually when not in manual mode.");
    }

    // Assign the new board cards
    for (int card : board_cards) {
        if (card >= 0) {
            this->board.push_back(card);
        }
    }

    // Optionally, update the round if the board is fully dealt
    if (this->board.size() == 3 && this->round == 0) { // Flop
        this->round = 1;
    } else if (this->board.size() == 4 && this->round == 1) { // Turn
        this->round = 2;
    } else if (this->board.size() == 5 && this->round == 2) { // River
        this->round = 3;
    } else if (this->board.size() > 5) {
        throw std::runtime_error("Board cannot have more than 5 cards.");
    }
}

/* queries */
std::array<double, PokerEngine::MAX_PLAYERS> PokerEngine::get_finishing_stacks() const {
    std::array<double, MAX_PLAYERS> stacks;
    for(int i = 0; i < n_players; ++i){
        stacks[i] = players[i].stack;
    }
    return stacks;
}

std::array<double, PokerEngine::MAX_PLAYERS> PokerEngine::get_payoffs() const {
    return this->payoffs;
}

bool PokerEngine::is_playing(int player) const {
    return this->players[player].status == PlayerStatus::Playing;
}

bool PokerEngine::get_game_status() const {
    // TODO: Implement
    return this->game_status;
}

int PokerEngine::turn() const {
    // TODO: Implement
    return this->actor;
}

double PokerEngine::get_big_blind() const {
    return this->big_blind;
}

double PokerEngine::get_pot() const {
    return this->pot;
}

// copy 
PokerEngine PokerEngine::copy() const {
    return *this;
}

