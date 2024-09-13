#ifndef POKER_ENGINE_H
#define POKER_ENGINE_H

#include <array>
#include <vector>
#include <cstdint>
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"

// minimal No Limit Texas Hold'em poker engine

class PokerEngine {
public:
    // constants
    static const int MAX_PLAYERS = 6;
    static const int BOARD_SIZE = 5;

    // constructor
    PokerEngine(
        std::vector<double> starting_stacks,
        int actor,
        int n_players, 
        double small_blind, 
        double big_blind,
        int max_round_bets,
        bool manual
        );
    ~PokerEngine() = default;

    // queries
    bool is_in(int player) const;
    int turn() const;

    // Actions
    void fold(int player);
    void bet_or_raise(int player, double amount);
    void check_or_call(int player);

    // Action verifications
    bool verify_min_raise(int player, double amount) const;
    bool verify_sufficient_funds(int player, double amount) const;

    // Utility functions
    const std::array<double, MAX_PLAYERS> get_payoffs() const;
    void reset();
    PokerEngine copy() const;

    // manual mode
    void manual_deal_hand(int player, const omp::Hand& hand);
    void manual_deal_board(const std::vector<int>& board_cards);

    // game act
    void showdown();

    // Construct bet history
    void construct_history(std::vector<int>& bet_status, std::vector<double>& bet_fracs) const;

    enum class PlayerStatus { Playing, Folded, AllIn, Out };
    struct Player {
        omp::Hand hand;
        double stack;
        double total_bet = 0.0; // total bet across all rounds
        PlayerStatus status;
        std::array<std::vector<double>, 4> bets_per_round; // 4 betting rounds
    };

    std::array<double, MAX_PLAYERS> get_finishing_stacks() const;

    std::array<Player, MAX_PLAYERS> players;

private:
    // member variables
    int n_players;
    double small_blind;
    double big_blind;
    int max_round_bets;
    int round;
    int actor;
    int bet_idx;
    int pot;
    bool game_status;
    bool manual;

    std::vector<int> bet_status;
    std::vector<double> bet_history_raw;
    std::vector<double> bet_history_frac;
    std::array<double, MAX_PLAYERS> payoffs;
    std::vector<bool> status;
    std::vector<uint8_t> board; // In the class definition
    std::vector<uint8_t> deck;

    // Game flow
    void deal_cards();
    void move_action();

    double calc_min_bet_amt(int player) const;
    bool is_round_complete() const;
    bool is_everyone_all_in() const;
    bool should_force_check_or_call() const;

    void next_state();
    void end_game();

    // Helper function to get the maximum bet in the current round
    double get_current_max_bet() const;
};

#endif // POKER_ENGINE_H