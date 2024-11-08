#ifndef POKER_ENGINE_H
#define POKER_ENGINE_H

#include <array>
#include <vector>
#include <cstdint>
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"
#include "constants.h"
#include <random>

// minimal No Limit Texas Hold'em poker engine

class PokerEngine {
public:
    // constructor
    PokerEngine(
        std::array<double, NUM_PLAYERS> starting_stacks,
        std::array<double, NUM_PLAYERS> antes,
        int actor,
        double small_blind, 
        double big_blind,
        bool manual
        );

    ~PokerEngine() = default;

    // queries
    bool get_game_status() const;
    std::array<double, NUM_PLAYERS> get_payoffs() const;
    int turn() const;
    double get_big_blind() const;
    bool is_playing(int player) const;
    std::array<double, NUM_PLAYERS> get_finishing_stacks() const;
    double get_pot() const;
    std::array<int, 5> get_board() const;
    std::array<int, 52> get_deck() const;
    double get_call_amount(int player) const;

    // action verification functions
    bool can_fold(int player) const;
    bool can_check_or_call(int player) const;
    bool can_bet_or_raise(int player, double amount) const;

    // Actions
    void fold(int player);
    void bet_or_raise(int player, double amount);
    void check_or_call(int player);

    // Action verifications
    bool verify_min_raise(int player, double amount) const;
    bool verify_sufficient_funds(int player, double amount) const;

    // Utility functions
    void reset(
        std::array<double, NUM_PLAYERS> starting_stacks, 
        std::array<double, NUM_PLAYERS> antes, 
        int actor, 
        double small_blind, 
        double big_blind, 
        bool manual
    );

    // manual mode
    void manual_deal_hand(int player, std::array<int, 2> hand);
    void manual_deal_board(const std::array<int, 5> board_cards);

    // game act
    void showdown();

    // Construct bet history
    std::pair<std::array<int, NUM_PLAYERS * 4 * MAX_ROUND_BETS>, std::array<double, NUM_PLAYERS * 4 * MAX_ROUND_BETS>> construct_history() const;

    enum class PlayerStatus { Playing, Folded, AllIn, Out };
    struct Player {
        std::array<int, 2> hand;
        double stack;
        double total_bet = 0.0; // total bet across all rounds
        bool acted = false;
        PlayerStatus status;
        std::array<std::array<double, MAX_ROUND_BETS>, 4> bets_per_round; // 4 betting rounds
    };

    std::array<Player, NUM_PLAYERS> players;

    double small_blind;
    double big_blind;

    PokerEngine(const PokerEngine& other);
    PokerEngine& operator=(const PokerEngine& other);
    PokerEngine copy() const;
private:
    static std::random_device rd;  // static class member for global seed source
    static std::mt19937 master_rng; // master rng that seeds instance rngs
    std::mt19937 rng; // instance-specific rng

    // member variables
    int n_players;
    int round;
    int actor;
    int bet_idx;
    double pot;
    bool manual;
    bool game_status;

    std::array<double, NUM_PLAYERS> payoffs;
    std::array<int, 5> board; 
    std::array<int, 52> deck;

    // Game flow
    void deal_cards();
    int get_next_card();

    double calc_min_bet_amt(int player) const;
    bool is_round_complete() const;
    bool is_everyone_all_in() const;
    bool should_force_check_or_call() const;
    void reset_actions();

    void next_state();

    // Helper function to get the maximum bet in the current round
    double get_current_max_bet() const;
};

#endif // POKER_ENGINE_H