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

struct rusage measure_usage;

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

// The traverse function implementation
double traverse(
    PokerEngine& engine, 
    int player,
    std::array<void*, NUM_PLAYERS>& nets,
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

        void* net_ptr = nets[player];

        Infoset I = prepare_infoset(engine, player);

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
        void* net_ptr = nets[actor];
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

std::array<torch::Tensor, 4> init_batched_cards() {
    // Define shapes for each tensor
    std::vector<int64_t> hand_shape = {TRAIN_BS, 2};
    std::vector<int64_t> flop_shape = {TRAIN_BS, 3};
    std::vector<int64_t> turn_shape = {TRAIN_BS, 1};
    std::vector<int64_t> river_shape = {TRAIN_BS, 1};
    
    std::array<torch::Tensor, 4> batched_cards;

    // Initialize all tensors with zeros
    batched_cards[0] = torch::zeros(hand_shape, torch::kInt32);
    batched_cards[1] = torch::zeros(flop_shape, torch::kInt32);
    batched_cards[2] = torch::zeros(turn_shape, torch::kInt32);
    batched_cards[3] = torch::zeros(river_shape, torch::kInt32);

    return batched_cards;
}

torch::Tensor init_batched_fracs() {
    std::vector<int64_t> batched_fracs_shape = {TRAIN_BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return torch::zeros(batched_fracs_shape, torch::kInt32);
}

torch::Tensor init_batched_status() { 
    std::vector<int64_t> batched_status_shape = {TRAIN_BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return torch::zeros(batched_status_shape, torch::kInt32);
}

torch::Tensor init_batched_advs() {
    std::vector<int64_t> batched_advs_shape = {TRAIN_BS, NUM_ACTIONS};
    return torch::zeros(batched_advs_shape, torch::kFloat);
}

torch::Tensor init_batched_iters() {
    std::vector<int64_t> batched_iters_shape = {TRAIN_BS, 1};
    return torch::zeros(batched_iters_shape, torch::kFloat);
}

int main() {
    std::vector<TraverseAdvantage> all_traverse_advs;
    std::vector<std::array<void*, NUM_PLAYERS>> total_nets;
    std::array<void*, NUM_PLAYERS> init_player_nets{};
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        init_player_nets[i] = create_deep_cfr_model();
    }
    total_nets.push_back(init_player_nets);
    //all_traverse_advs.resize(MAX_ADVS); // Preallocate capacity
    std::atomic<size_t> all_traverse_advs_index(0);

    // init concurrency
    std::mutex advs_mutex;
    unsigned int num_threads = std::thread::hardware_concurrency();
    int traversals_per_thread = NUM_TRAVERSALS / num_threads;
    DEBUG_NONE("Traversals per thread: " << traversals_per_thread);
    int remaining_traversals = NUM_TRAVERSALS % num_threads;
    DEBUG_NONE("Remaining traversals: " << remaining_traversals);
    std::array<double, NUM_PLAYERS> starting_stacks{};
    std::array<double, NUM_PLAYERS> antes{};
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        starting_stacks[i] = 100.0;
        antes[i] = 0.0;
    } 
    double small_bet = 0.5;
    double big_bet = 1.0;
    int starting_actor = 0;

    // todo init nets
    auto thread_func = [&](int traversals_per_thread, int thread_id, int player, int cfr_iter, std::array<void*, NUM_PLAYERS> nets) {
        for(int k = 0; k < traversals_per_thread; ++k) {
            
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
                traverse(engine, player, nets, cfr_iter, all_traverse_advs, all_traverse_advs_index, advs_mutex);
            }
            catch(const std::exception& e) {
                DEBUG_INFO(e.what());
                continue;
            }
        }
    };

    for (int cfr_iter=0; cfr_iter<CFR_ITERS; ++cfr_iter) {
        std::array<void*, NUM_PLAYERS> player_nets = total_nets[total_nets.size()-1];
        for (int player=0; player<NUM_PLAYERS; ++player) {
            // spawn threads
            std::vector<std::thread> threads;
            for(unsigned int thread_id = 0; thread_id < num_threads; ++thread_id) {
                // for each thread, create new deep copy of the total_nets[total_nets.size()-1]
                int traversals_to_run = traversals_per_thread + (thread_id < remaining_traversals ? 1 : 0);
                threads.emplace_back([&thread_func, traversals_to_run, thread_id, player, cfr_iter, &player_nets]() {
                    thread_func(traversals_to_run, thread_id, player, cfr_iter, player_nets);
                });
            }

            // Wait for all threads to finish
            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }


            void* player_model = player_nets[player];
            DEBUG_NONE("traversals complete");

            // todo train
            int batch_repeat = all_traverse_advs.size() / TRAIN_BS;
            int advs_idx = 0;
            for (int _ = 0; _ < batch_repeat; ++_) {
                // init batch tensors
                std::array<torch::Tensor, 4> batched_cards = init_batched_cards();
                torch::Tensor batched_fracs = init_batched_fracs();
                torch::Tensor batched_status = init_batched_status();
                torch::Tensor batched_advs = init_batched_advs();
                torch::Tensor batched_iters = init_batched_iters();
                DEBUG_NONE("init complete");

                // populate batches 
                DEBUG_NONE("collected advs count: " << all_traverse_advs.size());
                for (size_t i = 0; i < TRAIN_BS; ++i) {
                    Infoset infoset = all_traverse_advs[advs_idx].infoset;
                    int iteration = all_traverse_advs[advs_idx].iteration;
                    std::array<double, NUM_ACTIONS> advs = all_traverse_advs[advs_idx].advantages;

                    batched_cards[0][i] = infoset.cards[0].squeeze();
                    batched_cards[1][i] = infoset.cards[1].squeeze();
                    batched_cards[2][i] = infoset.cards[2].squeeze();
                    batched_cards[3][i] = infoset.cards[3].squeeze();

                    batched_fracs[i] = infoset.bet_fracs.squeeze();
                    DEBUG_NONE("bet fracs init");
                    batched_status[i] = infoset.bet_status.squeeze();
                    DEBUG_NONE("bet status init");
                    for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                        batched_advs[i][a] = advs[a];
                        DEBUG_NONE("fill action " << a);
                    }
                    batched_iters[i] = iteration;
                    advs_idx++;
                }
                // train
                torch::optim::Adam optimizer(get_model_parameters(player_model));
                DEBUG_NONE("begin training...");
                for (size_t epoch = 0; epoch < TRAIN_EPOCHS; ++epoch) {
                    optimizer.zero_grad();

                    torch::Tensor pred = deep_cfr_model_forward(
                        player_model, 
                        batched_cards, 
                        batched_fracs, 
                        batched_status
                    );

                    torch::Tensor loss = torch::nn::functional::mse_loss(
                        pred, 
                        batched_advs,
                        torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)
                    );

                    // Calculate mean loss for logging
                    torch::Tensor mean_loss = loss.mean();

                    loss *= batched_iters;
                    
                    loss.backward();
                    optimizer.step();

                    // Log the mean loss every epoch
                    DEBUG_NONE("Epoch " << epoch + 1 << "/" << TRAIN_EPOCHS << ", Loss: " << mean_loss.item<float>());
                }
            }

            // todo eval - implement game sim in eval.cpp
            double eval_mbb = evaluate(player_model, player);
        }
    }
}

// Function to run multiple traversals in parallel
int profile_cfr(){
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

    std::array<void*, NUM_PLAYERS> nets;
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        nets[i] = create_deep_cfr_model();
    }

    auto thread_func = [&](int traversals_per_thread, int thread_id) {
        for(int k = 0; k < traversals_per_thread; ++k) {
            // Thread-specific memory
            int starting_actor = 0;
            void* init_model = create_deep_cfr_model();
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
