// ParsedPHH.h
#ifndef PARSED_PHH_H
#define PARSED_PHH_H

#include <vector>
#include <string>

struct ParsedPHH {
    std::vector<double> starting_stacks;
    std::vector<double> finishing_stacks;
    int n_players;
    double small_bet;
    double big_bet;
    std::vector<double> antes;
    std::vector<std::string> actions;
};

#endif // PARSED_PHH_H
