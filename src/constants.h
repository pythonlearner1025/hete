#ifndef CONSTANTS_H
#define CONSTANTS_H

// cfr
constexpr int NUM_PLAYERS = 2;
constexpr int NUM_TRAVERSALS = 1000;
constexpr int MAX_ADVS = 1e8; // Adjust this based on your expected maximum number of advantages
constexpr int MAX_ACTIONS = 5;
constexpr int MAX_PLAYERS = 6;
constexpr int CFR_ITERS = 1;

// engine
constexpr int MAX_ROUND_BETS = 3;
constexpr int BOARD_SIZE = 5;

//tests
constexpr int TEST_MAX_ACTIONS = 6;
constexpr int PHH_NUM_PLAYERS = NUM_PLAYERS;

#endif // CONSTANTS_H

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif