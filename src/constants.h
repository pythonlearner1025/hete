#ifndef CONSTANTS_H
#define CONSTANTS_H

// cfr
constexpr size_t NUM_PLAYERS = 2;
constexpr size_t NUM_TRAVERSALS = 1000;
constexpr size_t MAX_ADVS = 1e8; // Adjust this based on your expected maximum number of advantages
constexpr size_t NUM_ACTIONS = 5;
constexpr size_t MAX_PLAYERS = 6;
constexpr size_t CFR_ITERS = 1;

// engine
constexpr int MAX_ROUND_BETS = 3;
constexpr int BOARD_SIZE = 5;

// model 
constexpr int64_t MODEL_DIM = 256;

// training
constexpr size_t TRAIN_BS = 1024;
constexpr size_t TRAIN_EPOCHS = 10;

// eval
constexpr size_t BS = 10000;
constexpr size_t WP_MC_SAMPLES = 30000;
constexpr size_t ACT_MC_SAMPLES = 10000;

// shared
constexpr int NULL_CARD = 69; 

// optimization
constexpr bool ESTIMATE_SHOWDOWN = true; 

//tests
constexpr int TEST_MAX_ACTIONS = 6;
constexpr int PHH_NUM_PLAYERS = NUM_PLAYERS;

#endif // CONSTANTS_H

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif