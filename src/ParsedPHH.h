// ParsedPHH.h
#ifndef PARSED_PHH_H
#define PARSED_PHH_H

#include <vector>
#include <string>
#include "constants.h"

struct ParsedPHH {
    std::array<double, PHH_NUM_PLAYERS> starting_stacks;
    std::array<double, PHH_NUM_PLAYERS> finishing_stacks;
    int n_players;
    double small_bet;
    double big_bet;
    std::array<double, PHH_NUM_PLAYERS> antes;
    std::vector<std::string> actions;
};

#endif // PARSED_PHH_H
