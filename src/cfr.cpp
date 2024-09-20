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

torch::Tensor random_forward() {
    return torch::rand({NUM_ACTIONS}) * 2 - 1;  // Random values between -1 and 1
}

thread_local std::vector<TraverseAdvantage> local_advs;

// The traverse function implementation
double traverse(
    PokerEngine& engine, 
    int player,
    std::array<void*, NUM_PLAYERS>& nets,
    int t,
    std::vector<TraverseAdvantage>& all_traverse_advs,
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
       // torch::Tensor logits = random_forward();

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

        // No need for a mutex since each thread writes to a unique index
        // turns out, the mutex slows down things by 1000x. 
        /*
        {
            std::lock_guard<std::mutex> lock(advs_mutex);
            all_traverse_advs.push_back(TraverseAdvantage{I, t, advs});
        }
        */
       local_advs.push_back(TraverseAdvantage{I, t, advs});


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
        //torch::Tensor logits = random_forward();
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
    batched_cards[0] = torch::empty(hand_shape, torch::kInt32);
    batched_cards[1] = torch::empty(flop_shape, torch::kInt32);
    batched_cards[2] = torch::empty(turn_shape, torch::kInt32);
    batched_cards[3] = torch::empty(river_shape, torch::kInt32);

    return batched_cards;
}

torch::Tensor init_batched_fracs() {
    std::vector<int64_t> batched_fracs_shape = {TRAIN_BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return torch::empty(batched_fracs_shape, torch::kFloat);
}

torch::Tensor init_batched_status() { 
    std::vector<int64_t> batched_status_shape = {TRAIN_BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return torch::empty(batched_status_shape, torch::kFloat);
}

torch::Tensor init_batched_advs() {
    std::vector<int64_t> batched_advs_shape = {TRAIN_BS, NUM_ACTIONS};
    return torch::empty(batched_advs_shape, torch::kFloat);
}

torch::Tensor init_batched_iters() {
    std::vector<int64_t> batched_iters_shape = {TRAIN_BS, 1};
    return torch::empty(batched_iters_shape, torch::kFloat);
}

std::vector<TraverseAdvantage>& get_local_advs() {
    return local_advs;
}

// more cfr traversal steps converges faster but is sample inefficient
// 3000 cfr steps * 500 cfr iters converges same as 100,000 and 500 cfr iters
// so estimate around ~1 million total steps budget
// around 10,000 training iter steps needed per iter
int main() {
    std::vector<TraverseAdvantage> all_traverse_advs;
    std::vector<std::array<void*, NUM_PLAYERS>> total_nets;
    std::array<void*, NUM_PLAYERS> init_player_nets{};
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        init_player_nets[i] = create_deep_cfr_model();
    }
    total_nets.push_back(init_player_nets);
    //all_traverse_advs.resize(MAX_ADVS); // Preallocate capacity

    // init concurrency
    std::mutex advs_mutex;
    unsigned int num_threads = std::thread::hardware_concurrency();
    int traversals_per_thread = NUM_TRAVERSALS / num_threads;
    int remaining_traversals = NUM_TRAVERSALS % num_threads;
    DEBUG_NONE("Traversals per thread: " << traversals_per_thread);
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
            if (k > 0) {
                DEBUG_NONE("thread=" << thread_id << ", traversal=" << k << ", advs=" << all_traverse_advs.size());
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
                traverse(engine, player, nets, cfr_iter, all_traverse_advs, advs_mutex);
            }
            catch(const std::exception& e) {
                DEBUG_INFO(e.what());
                continue;
            }
        }
    };
    // for threading use intel tbb if all else fails
    std::vector<std::vector<TraverseAdvantage>> all_thread_advs;

    for (int cfr_iter=1; cfr_iter<CFR_ITERS+1; ++cfr_iter) {
        std::array<void*, NUM_PLAYERS> player_nets = total_nets[total_nets.size()-1];
        for (int player=0; player<NUM_PLAYERS; ++player) {
            // spawn threads
            std::vector<std::thread> threads;
            for(unsigned int thread_id = 0; thread_id < num_threads; ++thread_id) {
                // for each thread, create new deep copy of the total_nets[total_nets.size()-1]
                int traversals_to_run = traversals_per_thread + (thread_id < remaining_traversals ? 1 : 0);
                threads.emplace_back([&thread_func, traversals_to_run, thread_id, player, cfr_iter, &player_nets, &all_thread_advs]() {
                    thread_func(traversals_to_run, thread_id, player, cfr_iter, player_nets);
                    all_thread_advs.push_back(std::move(get_local_advs()));
                });
            }

            // Wait for all threads to finish
            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }

            // Merge results from all threads
            std::vector<TraverseAdvantage> global_advs;
            for (auto& thread_advs : all_thread_advs) {
                global_advs.insert(global_advs.end(), 
                                thread_advs.begin(), 
                                thread_advs.end());
            }

         

            void* player_model = player_nets[player];
            DEBUG_NONE("CFR ITER = " << cfr_iter);
            DEBUG_NONE("COLLECTED ADVS = " << all_traverse_advs.size());
            // todo train
            // draw training samples
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            std::shuffle(all_traverse_advs.begin(), all_traverse_advs.end(), rng);
            std::vector<TraverseAdvantage> training_advs;
            for (size_t i = 0; i < TRAIN_BS * TRAIN_EPOCHS; ++i) {
                if (i < all_traverse_advs.size()) {
                    training_advs.push_back(all_traverse_advs[i]);
                }
            }

            DEBUG_NONE("TOTAL TRAINING SAMPLES = " << training_advs.size());
            int batch_repeat = all_traverse_advs.size() / TRAIN_BS;
            int advs_idx = 0;
            for (int _ = 0; _ < batch_repeat; ++_) {
                // init batch tensors
                std::array<torch::Tensor, 4> batched_cards = init_batched_cards();
                torch::Tensor batched_fracs = init_batched_fracs();
                torch::Tensor batched_status = init_batched_status();
                torch::Tensor batched_advs = init_batched_advs();
                torch::Tensor batched_iters = init_batched_iters();
                // populate batches 
                for (size_t i = 0; i < TRAIN_BS; ++i) {
                    if (advs_idx >= all_traverse_advs.size()) {
                    //    DEBUG_NONE("advs_idx > all_traverse_advs");
                        break;
                    }
                    Infoset infoset = all_traverse_advs[advs_idx].infoset;
                    int iteration = all_traverse_advs[advs_idx].iteration;
                    std::array<double, NUM_ACTIONS> advs = all_traverse_advs[advs_idx].advantages;
                    batched_cards[0][i] = infoset.cards[0].squeeze();
                    batched_cards[1][i] = infoset.cards[1].squeeze();
                    batched_cards[2][i] = infoset.cards[2].squeeze();
                    batched_cards[3][i] = infoset.cards[3].squeeze();
                    batched_fracs[i] = infoset.bet_fracs.squeeze();
                    //batched_fracs[i] = infoset.bet_fracs.squeeze();
                    batched_status[i] = infoset.bet_status.squeeze();
                    auto batched_advs_accessor = batched_advs.accessor<float, 2>();
                    for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                        batched_advs_accessor[i][a] = advs[a];
                    }
                    batched_iters[i] = iteration;
                    advs_idx++;
                }

                // train
                torch::optim::Adam optimizer(get_model_parameters(player_model));

                // TODO random sampling from all_traverse_advs
                // maximum 50 million samples
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


                    loss *= batched_iters;

                    //torch::Tensor columnwise_mean_loss = loss.mean(1);
                    torch::Tensor batch_mean_loss = loss.mean(1).mean();
                    
                    batch_mean_loss.backward();
                    optimizer.step();

                    // Log the mean loss every epoch
                    DEBUG_NONE("Epoch " << epoch + 1 << "/" << TRAIN_EPOCHS << ", Loss: " << batch_mean_loss.item<float>());
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
                traverse(engine, player, nets, t, all_traverse_advs, advs_mutex);
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

    /*
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    auto throughput_recur = static_cast<float>(total_traversals) / duration.count();
    auto throughput = static_cast<float>(NUM_TRAVERSALS) / duration.count();

    DEBUG_NONE("All jobs done");
    DEBUG_NONE(get_timestamp() << " Total traversals: " << total_traversals);
    //DEBUG_NONE(get_timestamp() << " Total advantages:: " << total_advantages);
    DEBUG_NONE(get_timestamp() << " Size of all_traverse_advs: " << all_traverse_advs.size());
    DEBUG_NONE(get_timestamp() << " traverse call throughput : " << throughput_recur << " calls/s");
    DEBUG_NONE(get_timestamp() << " traverse iter throughput : " << throughput << " traversals/s");
    */

    return 0;
}
