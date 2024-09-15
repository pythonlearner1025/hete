// cfr.cpp

#include "cfr.h"
#include "debug.h"  // Include the debug header
#include <torch/torch.h>
#include <tuple>
#include <thread>
#include <atomic>
#include <set>
#include <algorithm>
#include <cstdlib> // For rand()
#include <ctime>   // For seeding rand()
#include <chrono>
#include <iomanip> // For std::put_time
#include <sstream> // For std::stringstream

// Helper function for getting current timestamp
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %X");
    return ss.str();
}

// Initialize random seed
struct RandInit {
    RandInit() { std::srand(static_cast<unsigned int>(std::time(nullptr))); }
} rand_init;

void get_cards(PokerEngine& game, int player, Infoset& I) {
    std::array<int, 5> board = game.get_board();
    int board_size = 0;
    for (size_t i = 0; i < 5; ++i) {
        if (board[i] != -1) board_size++;
    }

    DEBUG_INFO("Player hand: " << game.players[player].hand[0] << ", " << game.players[player].hand[1]);

    I.cards[0] = torch::tensor({static_cast<int64_t>(game.players[player].hand[0]), 
                                static_cast<int64_t>(game.players[player].hand[1])}).view({1, 2});

    DEBUG_INFO("Board size: " << board_size);

    if (board_size >= 3) {
        DEBUG_INFO("Flop: " << board[0] << ", " << board[1] << ", " << board[2]);
        I.cards[1] = torch::tensor({static_cast<int64_t>(board[0]), 
                                    static_cast<int64_t>(board[1]), 
                                    static_cast<int64_t>(board[2])}).view({1, 3});
    } else {
        I.cards[1] = torch::tensor({-1, -1, -1}).view({1, 3});
    }

    if (board_size >= 4) {
        DEBUG_INFO("Turn: " << board[3]);
        I.cards[2] = torch::tensor({static_cast<int64_t>(board[3])}).view({1, 1});
    } else {
        I.cards[2] = torch::tensor({-1}).view({1, 1});
    }

    if (board_size >= 5) {
        DEBUG_INFO("River: " << board[4]);
        I.cards[3] = torch::tensor({static_cast<int64_t>(board[4])}).view({1, 1});
    } else {
        I.cards[3] = torch::tensor({-1}).view({1, 1});
    }
}

Infoset prepare_infoset(
    PokerEngine& game,
    int player,
    int max_bets_per_player
) {
    Infoset I;
    auto history = game.construct_history();
    get_cards(game, player, I);

    // Convert bet_fracs array to tensor
    I.bet_fracs = torch::from_blob(history.second.data(), 
                                   {1, static_cast<long long>(history.second.size())}, 
                                   torch::kFloat32);

    // Convert bet_status array to tensor
    I.bet_status = torch::from_blob(history.first.data(), 
                                    {1, static_cast<long long>(history.first.size())}, 
                                    torch::kBool).to(torch::kFloat32);

    return I;
}

// must return a prob dist
std::array<double, MAX_ACTIONS> regret_match(const torch::Tensor& logits, int n_acts) {
    auto relu_logits = torch::relu(logits);
    
    double logits_sum = relu_logits.sum().item<double>();
    
    std::array<double, MAX_ACTIONS> strat;
    
    // If the sum is positive, calculate the strategy
    if (logits_sum > 0) {
        auto strategy_tensor = relu_logits / logits_sum;
        std::copy(strategy_tensor.data_ptr<float>(), strategy_tensor.data_ptr<float>() + n_acts, strat.begin());
    } 
    // If the sum is zero or negative, return a one-hot vector for the max logit
    else {
        auto max_index = torch::argmax(relu_logits).item<int>();
        std::fill(strat.begin(), strat.end(), 0.0);
        strat[max_index] = 1.0;
    }
    return strat;
}

// Sample an action according to the strategy probabilities
int sample_action(const std::array<double, MAX_ACTIONS>& strat, int n_acts) {
    double r = static_cast<double>(rand()) / RAND_MAX;
    double cumulative = 0.0;
    DEBUG_INFO("r is " << r);
    for (int i = 0; i < n_acts; ++i) {
        DEBUG_INFO("strat " << i << " has p=" << strat[i]);
        cumulative += strat[i];
        if (r <= cumulative) {
            DEBUG_INFO("returning " << i);
            return i;
        }
    }
    DEBUG_INFO("returning " << (n_acts - 1));
    return n_acts - 1; // Return last valid action if none selected
}

void take_action(PokerEngine& engine, int player, int act, int n_acts) {
    DEBUG_INFO("Chosen act: " << act);
    if (act == 0) {
        engine.fold(player);
        return;
    }
    if (act == 1) {
        engine.check_or_call(player); 
        return;
    }
    double inc = engine.get_pot() * 1.0 / static_cast<double>(n_acts);
    double bet_amt = inc;
    for (int a = 2; a < n_acts; ++a) {
        if (a == act) {
            engine.bet_or_raise(player, bet_amt);
            return;
        }
        bet_amt += inc;
    }
}

bool verify_action(PokerEngine& engine, int player, int act, int n_acts) {
    if (act == 0) {
        return engine.can_fold(player);
    }
    if (act == 1) {
        return engine.can_check_or_call(player); 
    }
    double inc = engine.get_pot() * 1.0 / static_cast<double>(n_acts);
    double bet_amt = inc;
    for (int a = 2; a < n_acts; ++a) {
        if (a == act) {
            return engine.can_bet_or_raise(player, bet_amt);
        }
        bet_amt += inc;
    }
    return false;
}

// The traverse function implementation
double traverse(
    PokerEngine& engine, 
    int player,
    std::array<std::array<void*, CFR_ITERS>, NUM_PLAYERS>& nets,
    int t,
    std::array<TraverseAdvantage, MAX_ADVS>& all_traverse_advs,
    std::atomic<size_t>& all_traverse_advs_index,
    std::mutex& advs_mutex
) {
    DEBUG_INFO(get_timestamp() << " Entering traverse()");
    DEBUG_INFO(get_timestamp() << " Game status: " << engine.get_game_status() << ", Is player playing: " << engine.is_playing(player));

    // Check if game is done for this player
    if (!engine.get_game_status() || !engine.is_playing(player)) {
        DEBUG_INFO(get_timestamp() << " Game over");
        std::array<double, MAX_PLAYERS> payoff = engine.get_payoffs();

        /*
        for (size_t i=0; i<NUM_PLAYERS; ++i) {
            DEBUG_INFO("player " << i << " payoff: " << payoff[i]);
        }
        DEBUG_INFO(""); // For additional newline
        */

        double bb = engine.get_big_blind();
        return payoff[player] / bb;
    }
    else if (engine.turn() == player) {
        DEBUG_INFO(get_timestamp() << " My turn");

        void* net_ptr = nets[player][CFR_ITERS-1];

        Infoset I = prepare_infoset(engine, player, MAX_ROUND_BETS);

        torch::Tensor logits = deep_cfr_model_forward(net_ptr, I.cards, I.bet_fracs, I.bet_status);

        DEBUG_INFO(get_timestamp() << " Neural network logits: " << logits);

        int n_acts = get_action_head_dim(net_ptr);
        // Regret matching
        std::array<double, MAX_ACTIONS> strat = regret_match(logits, n_acts);

        // Define possible actions
        std::array<double, MAX_ACTIONS> values{};  // Initialize all elements to 0.0
        std::array<double, MAX_ACTIONS> advs{};    // Initialize all elements to 0.0
        std::array<bool, MAX_ACTIONS> is_illegal{};  // Initialize all elements to false
        double ev = 0.0;

        for (size_t a = 0; a < n_acts; ++a) {

            // Verify action
            // ensure game state doesn't change
            if (!verify_action(engine, player, a, n_acts)) {
                is_illegal[a] = true;
                DEBUG_INFO("action is illegal, skipping: " << a);
                continue;
            } else {
                DEBUG_INFO("action is legal, modeling: " << a);
            }

            // Create a copy of the game state
            PokerEngine new_engine = engine.copy();

            is_illegal[a] = false;

            // Take action
            take_action(new_engine, player, a, n_acts);

            // Recursive call to traverse
            double value = traverse(
                new_engine, 
                player, 
                nets, 
                t, 
                all_traverse_advs,
                all_traverse_advs_index,
                advs_mutex
            );
            DEBUG_INFO("value: " << value << " strat %: " << strat[a]);
            values[a] = value;
            ev += value * strat[a];
        }

        DEBUG_INFO(get_timestamp() << " Calculated advantages: ");
        for (size_t a = 0; a < n_acts; ++a) {
            if (!is_illegal[a]) {
                double adv = values[a] - ev;
                advs[a] = (adv > 0.0) ? adv : (1.0 / static_cast<double>(n_acts));
            } else {
                // TODO what should illegal act adv be? 
                advs[a] = 0.0;
            }
            DEBUG_INFO(advs[a] << " ");
        }
        DEBUG_INFO(""); // For newline

        // Add to traverse_advs with synchronization
        TraverseAdvantage ta;
        ta.infoset = I;
        ta.iteration = t;
        ta.advantages = advs;

        // When adding an advantage:
        size_t index = all_traverse_advs_index.fetch_add(1);
        if (index < MAX_ADVS) {
            std::lock_guard<std::mutex> lock(advs_mutex);
            all_traverse_advs[index] = TraverseAdvantage{I, t, advs};
        } else {
            // Handle the case where we've exceeded MAX_ADVS
            DEBUG_INFO("Warning: Exceeded MAX_ADVS");
        }

        // Return expected value
        DEBUG_INFO(get_timestamp() << "ev: " << ev);
        return ev;
    }
    else {
        DEBUG_INFO(get_timestamp() << " Opponent's turn");
        // Opponent's turn
        int actor = engine.turn();
        void* net_ptr = nets[actor][CFR_ITERS-1];
        int n_acts = get_action_head_dim(net_ptr);
        // Prepare infoset
        Infoset I = prepare_infoset(engine, actor, MAX_ROUND_BETS);
        // Forward pass
        torch::Tensor logits = deep_cfr_model_forward(net_ptr, I.cards, I.bet_fracs, I.bet_status);
        // Regret matching
        std::array<double, MAX_ACTIONS> strat = regret_match(logits, n_acts);

        // Sample action according to strat
        int action_index = sample_action(strat, n_acts);

        // Verify and adjust action if necessary
        while (!verify_action(engine, player, action_index, n_acts)) {
            action_index = (action_index - 1) % n_acts;
        }
        // Take action
        take_action(engine, actor, action_index, n_acts);

        DEBUG_INFO(get_timestamp() << " Opponent's turn. Selected action: " << action_index);

        // Recursive call
        return traverse(
            engine,
            player,
            nets,
            t,
            all_traverse_advs,
            all_traverse_advs_index,
            advs_mutex
        );
    }
}

// Function to run multiple traversals in parallel
// TODO - why is pot always no greater than big blind?
int main(){
    int player = 0;
    double small_bet = 0.5;
    double big_bet = 1;
    DEBUG_NONE("hello");
    std::array<double, NUM_PLAYERS> antes = {0.0, 0.0};
    std::array<double, NUM_PLAYERS> starting_stacks = {100.0, 100.0};
    DEBUG_NONE("init antes & satcks");
    void* init_model = create_deep_cfr_model(NUM_PLAYERS, MAX_ROUND_BETS, MAX_ACTIONS);
    DEBUG_NONE("init model");
    int t = 0;
    std::array<std::array<void*, CFR_ITERS>, NUM_PLAYERS> nets;
    for (size_t i = 0; i<NUM_PLAYERS; ++i) {
        for (size_t j = 0; j <CFR_ITERS; ++j){
            nets[i][j] = init_model;
        }
    }
    DEBUG_NONE("init nets");
    std::array<TraverseAdvantage, MAX_ADVS> all_traverse_advs;
    DEBUG_NONE("init all advs");
    std::atomic<size_t> all_traverse_advs_index(0);
    std::mutex advs_mutex;

    // Determine the number of available hardware threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback to 4 if unable to detect

    DEBUG_NONE("Total threads: " << num_threads);

    auto thread_func = [&](int traversals_per_thread, int thread_id) {
        for(int k = 0; k < traversals_per_thread; ++k) {
            // Extract parameters
            int starting_actor = 0;

            // Initialize PokerEngine
            PokerEngine engine(
                starting_stacks, 
                antes,
                starting_actor,
                NUM_PLAYERS, 
                small_bet, 
                big_bet, 
                false
            );

            try {
                traverse(engine, player, nets, t, all_traverse_advs, all_traverse_advs_index, advs_mutex);
            }
            catch(const std::exception& e) {
                DEBUG_INFO(e.what());
                continue;
            }
        }
    };

    // Calculate traversals per thread
    int traversals_per_thread = NUM_TRAVERSALS / num_threads;
    DEBUG_NONE("Traversals per thread: " << traversals_per_thread);
    int remaining_traversals = NUM_TRAVERSALS % num_threads;
    DEBUG_NONE("Remaining traversals: " << remaining_traversals);

    // Create threads
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for(unsigned int i = 0; i < num_threads; ++i) {
        int traversals_to_run = traversals_per_thread + (i < remaining_traversals ? 1 : 0);
        threads.emplace_back([&thread_func, traversals_to_run, i]() {
            thread_func(traversals_to_run, i);
        });
    }

    // Wait for all threads to finish
    for(auto& thread : threads) {
        if(thread.joinable()) {
            thread.join();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    size_t total_traversals = all_traverse_advs_index.fetch_add(1);
    auto throughput = static_cast<float>(total_traversals) / duration.count();

    DEBUG_NONE("All jobs done");
    DEBUG_NONE(get_timestamp() << " Total traversals: " << total_traversals);
    DEBUG_NONE(get_timestamp() << " Size of all_traverse_advs: " << all_traverse_advs.size());
    DEBUG_NONE(get_timestamp() << " Throughput: " << throughput << " traversals/s");

    return 0;
}
