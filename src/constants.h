#ifndef CONSTANTS_H
#define CONSTANTS_H

// cfr
constexpr size_t NUM_PLAYERS = 2;
constexpr size_t NUM_TRAVERSALS = 1000;
constexpr size_t MAX_SIZE = 1e7; // Adjust this based on your expected maximum number of advantages
constexpr size_t CFR_MAX_SIZE = 1e6;
constexpr size_t NUM_ACTIONS = 6;
constexpr size_t MAX_PLAYERS = 2;
constexpr size_t CFR_ITERS = 1000;
constexpr size_t NUM_THREADS = 1;

constexpr float BETA = 0;
constexpr double EPSILON = 0.20;
constexpr int PRINT_PER = 1;

// training
constexpr size_t TRAIN_BS = 1024;
constexpr size_t TRAIN_EPOCHS = 50;
constexpr size_t TRAIN_ITERS = 5000;

// engine
constexpr int MAX_ROUND_BETS = 6;
constexpr int BOARD_SIZE = 5;

// model 
constexpr int64_t MODEL_DIM = 128;
constexpr size_t N_HEADS = 4;
constexpr size_t N_LAYERS = 1;
constexpr int HEAD_DIM = MODEL_DIM / N_HEADS; 

// eval
constexpr size_t EVAL_BS = 100000;
constexpr size_t WP_MC_SAMPLES = 10000;
constexpr size_t ACT_MC_SAMPLES = 10000;
constexpr size_t EVAL_MC_SAMPLES = 1; 

// shared
constexpr int NULL_CARD = -1; 
constexpr int USED_CARD = -2;

// optimization
constexpr bool ESTIMATE_SHOWDOWN = true; 

//tests
constexpr int TEST_MAX_ACTIONS = 6;
constexpr int PHH_NUM_PLAYERS = NUM_PLAYERS;

#endif // CONSTANTS_H

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif
