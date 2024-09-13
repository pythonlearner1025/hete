#include "engine.h"
#include <iostream>

void test_poker() {
    
}

int main() {
    std::vector<double> starting_stacks = {1000.0, 1000.0, 1000.0, 1000.0};
    int n_players = 4;
    int small_blind = 5;
    int big_blind = 10;
    int max_round_bets = 3;

    PokerEngine engine(starting_stacks, n_players, small_blind, big_blind, max_round_bets);
    
    std::cout << "success" << std::endl;
    
    return 0;
}