#include "cfr.h"
#include "debug.h"
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

//std::atomic<int> total_advantages(0);
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

// Sample an action according to the strategy probabilities
int sample_action(const std::array<double, NUM_ACTIONS>& strat) {
    double r = static_cast<double>(rand()) / RAND_MAX;
    double cumulative = 0.0;
    DEBUG_INFO("r is " << r);
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        DEBUG_INFO("strat " << i << " has p=" << strat[i]);
        cumulative += strat[i];
        if (r <= cumulative) {
            DEBUG_INFO("returning " << i);
            return i;
        }
    }
    DEBUG_INFO("returning " << (NUM_ACTIONS - 1));
    return NUM_ACTIONS - 1; // Return last valid action if none selected
}

void take_action(PokerEngine& engine, int player, int act) {
    DEBUG_INFO("Chosen act: " << act);
    if (act == 0) {
        engine.fold(player);
        return;
    }
    if (act == 1) {
        engine.check_or_call(player); 
        return;
    }
    double inc = engine.get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            engine.bet_or_raise(player, bet_amt);
            return;
        }
        bet_amt += inc;
    }
}

bool verify_action(PokerEngine& engine, int player, int act) {
    if (act == 0) {
        return engine.can_fold(player);
    }
    if (act == 1) {
        return engine.can_check_or_call(player); 
    }
    double inc = engine.get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
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
    std::vector<TraverseAdvantage>& all_traverse_advs,
    std::atomic<size_t>& all_traverse_advs_index,
    std::mutex& advs_mutex
) {
    DEBUG_INFO(get_timestamp() << " Entering traverse()");
    DEBUG_INFO(get_timestamp() << " Game status: " << engine.get_game_status() << ", Is player playing: " << engine.is_playing(player));


    for (size_t p=0; p<NUM_PLAYERS; ++p){
        DEBUG_INFO("player " << p << " hand: " << engine.players[p].hand[0] << "," << engine.players[p].hand[1]);
    }

    // Check if game is done for this player
    if (!engine.get_game_status() || !engine.is_playing(player)) {
        DEBUG_INFO(get_timestamp() << " Game over");
        std::array<double, NUM_PLAYERS> payoff = engine.get_payoffs();

        double bb = engine.get_big_blind();
        return payoff[player] / bb;
    }
    else if (engine.turn() == player) {
        DEBUG_INFO(get_timestamp() << " My turn");

        void* net_ptr = nets[player][CFR_ITERS-1];

        Infoset I = prepare_infoset(engine, player, MAX_ROUND_BETS);

        torch::Tensor logits = deep_cfr_model_forward(net_ptr, I.cards, I.bet_fracs, I.bet_status);

        DEBUG_INFO(get_timestamp() << " Neural network logits: " << logits);

        // Regret matching
        std::array<double, NUM_ACTIONS> strat = regret_match(logits);

        // Define possible actions
        std::array<double, NUM_ACTIONS> values{};  // Initialize all elements to 0.0
        std::array<double, NUM_ACTIONS> advs{};    // Initialize all elements to 0.0
        std::array<bool, NUM_ACTIONS> is_illegal{};  // Initialize all elements to false
        double ev = 0.0;

        for (size_t a = 0; a < NUM_ACTIONS; ++a) {

            // Verify action
            // ensure game state doesn't change
            if (!verify_action(engine, player, a)) {
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
            take_action(new_engine, player, a);

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
        for (size_t a = 0; a < NUM_ACTIONS; ++a) {
            if (!is_illegal[a]) {
                double adv = values[a] - ev;
                advs[a] = (adv > 0.0) ? adv : (1.0 / static_cast<double>(NUM_ACTIONS));
            } else {
                advs[a] = 0.0;
            }
            DEBUG_INFO(advs[a] << " ");
        }
        DEBUG_INFO(""); // For newline

         // Add to traverse_advs with synchronization
        size_t index = all_traverse_advs_index.fetch_add(1);
        if (index < all_traverse_advs.size()) {
            // No need for a mutex since each thread writes to a unique index
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
        // Prepare infoset
        Infoset I = prepare_infoset(engine, actor);
        // Forward pass
        torch::Tensor logits = deep_cfr_model_forward(net_ptr, I.cards, I.bet_fracs, I.bet_status);
        // Regret matching
        std::array<double, NUM_ACTIONS> strat = regret_match(logits);

        // Sample action according to strat
        int action_index = sample_action(strat);

        // Verify and adjust action if necessary
        while (!verify_action(engine, actor, action_index)) {
            action_index = (action_index - 1) % NUM_ACTIONS;
        }
        // Take action
        take_action(engine, actor, action_index);

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
int y(){
    int player = 0;
    double small_bet = 0.5;
    double big_bet = 1;
    std::array<double, NUM_PLAYERS> antes = {0.0, 0.0};
    std::array<double, NUM_PLAYERS> starting_stacks = {100.0, 100.0};
    int t = 0;
    
    std::vector<TraverseAdvantage> all_traverse_advs;
    all_traverse_advs.resize(MAX_ADVS); // Preallocate capacity
    std::atomic<size_t> all_traverse_advs_index(0);
    std::mutex advs_mutex;

    // Determine the number of available hardware threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback to 4 if unable to detect
    DEBUG_NONE("NUM_PLAYERS = " << NUM_PLAYERS);
    DEBUG_NONE("TRAVERSALS = " << NUM_TRAVERSALS);
    DEBUG_NONE("NUM_ACTIONS = " << NUM_ACTIONS);
    DEBUG_NONE("MAX_ROUND_BETS = " << MAX_ROUND_BETS);
    DEBUG_NONE("NUM_THREADS = " << num_threads);

    auto thread_func = [&](int traversals_per_thread, int thread_id) {
        for(int k = 0; k < traversals_per_thread; ++k) {
            // Thread-specific memory
            int starting_actor = 0;
            void* init_model = create_deep_cfr_model();
            std::array<std::array<void*, CFR_ITERS>, NUM_PLAYERS> nets;
            for (size_t i = 0; i<NUM_PLAYERS; ++i) {
                for (size_t j = 0; j < CFR_ITERS; ++j){
                    nets[i][j] = init_model;
                }
            }
            // Initialize PokerEngine
            PokerEngine engine(
                starting_stacks, 
                antes,
                starting_actor,
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
    size_t total_traversals = all_traverse_advs_index.load();
    auto throughput_recur = static_cast<float>(total_traversals) / duration.count();
    auto throughput = static_cast<float>(NUM_TRAVERSALS) / duration.count();

    DEBUG_NONE("All jobs done");
    DEBUG_NONE(get_timestamp() << " Total traversals: " << total_traversals);
    //DEBUG_NONE(get_timestamp() << " Total advantages:: " << total_advantages);
    DEBUG_NONE(get_timestamp() << " Size of all_traverse_advs: " << all_traverse_advs.size());
    DEBUG_NONE(get_timestamp() << " traverse call throughput : " << throughput_recur << " calls/s");
    DEBUG_NONE(get_timestamp() << " traverse iter throughput : " << throughput << " traversals/s");

    return 0;
}
