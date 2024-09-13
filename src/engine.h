#ifndef POKER_ENGINE_H
#define POKER_ENGINE_H

#include <array>
#include <vector>
#include <cstdint>
#include "omp/Hand.h"

// minimal No Limit Texas Hold'em poker engine

class PokerEngine {
public:
    // constants
    static const int MAX_PLAYERS = 6;
    static const int BOARD_SIZE = 5;

    // constructor
    PokerEngine(
        std::vector<double> starting_stacks,
        int n_players, 
        double small_blind, 
        double big_blind,
        int max_round_bets
        );
    ~PokerEngine() = default;

    // setters
    void set_stack(std::vector<double> stack);

    // queries
    bool is_in(int player) const;
    int turn() const;

    // Actions
    void fold(int player);
    void bet_or_raise(int player, double amount);
    void check_or_call(int player);

    // Action verifications
    bool verify_min_raise(int player, double amount) const;
    bool verify_sufficient_funds(int player, int amount) const;

    // Utility functions
    std::array<double, MAX_PLAYERS> get_payoffs();
    void reset();
    PokerEngine copy() const;

private:
    struct Player {
        Hand hand;
        double stack;
        double bet = 0.0;
        bool in_hand = true;
        bool folded = false;
    };

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

    std::array<Player, MAX_PLAYERS> players;
    std::vector<int> bet_status;
    std::vector<double> bet_history_raw;
    std::vector<double> bet_history_frac;
    std::array<double, MAX_PLAYERS> payoffs;
    std::vector<bool> status;
    std::array<uint8_t, BOARD_SIZE> board;
    std::vector<uint8_t> deck;

    // Game flow
    void deal_cards();
    void move_action();
    double calc_min_bet_amt(int player);
    bool is_round_complete() const;

    void is_everyone_all_in();
    void is_round_complete();
    void next_state();
    void showdown();
    void end_game();
};

#endif // POKER_ENGINE_H