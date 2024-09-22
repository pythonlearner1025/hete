#include "cfr.h"
#include "debug.h"
#include <torch/torch.h>
#include <torch/script.h>
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
#include <memory>

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

std::vector<TraverseAdvantage> global_advs{};
std::atomic<size_t> global_index(0);
const int MAX_SIZE = 40e6;

torch::Tensor init_batched_hands(int BS) {
    std::vector<int64_t> hand_shape = {BS, 2};
    return torch::zeros(hand_shape, torch::kInt);
}

torch::Tensor init_batched_flops(int BS) {
    std::vector<int64_t> flop_shape = {BS, 2};
    return torch::zeros(flop_shape, torch::kInt);
}

torch::Tensor init_batched_turns(int BS) {
    std::vector<int64_t> turn_shape = {BS, 2};
    return torch::zeros(turn_shape, torch::kInt);
}

torch::Tensor init_batched_rivers(int BS) {
    std::vector<int64_t> river_shape = {BS, 2};
    return torch::zeros(river_shape, torch::kInt);
}

torch::Tensor init_batched_fracs(int BS) {
    std::vector<int64_t> batched_fracs_shape = {BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return torch::zeros(batched_fracs_shape, torch::kFloat);
}

torch::Tensor init_batched_status(int BS) { 
    std::vector<int64_t> batched_status_shape = {BS, NUM_PLAYERS * MAX_ROUND_BETS * 4};
    return torch::zeros(batched_status_shape, torch::kFloat);
}

torch::Tensor init_batched_advs(int BS) {
    std::vector<int64_t> batched_advs_shape = {BS, NUM_ACTIONS};
    return torch::zeros(batched_advs_shape, torch::kFloat);
}

torch::Tensor init_batched_iters(int BS) {
    std::vector<int64_t> batched_iters_shape = {BS, 1};
    return torch::zeros(batched_iters_shape, torch::kFloat);
}

void get_state(
    PokerEngine& game,
    State* state,
    int player
) {
    auto history = game.construct_history();
    std::array<int, 2> hand = game.players[player].hand;
    std::array<int, 5> board = game.get_board();

    // Copy the bet_status and bet_fracs from history to state
    for (size_t i = 0; i < NUM_PLAYERS * MAX_ROUND_BETS * 4; ++i) {
        state->bet_status[i] = history.first[i];
        state->bet_fracs[i] = history.second[i];
    }
    // Assign hand, flop, turn, and river
    state->hand = hand;
    for (int i = 0; i < 3; ++i) {
        state->flop[i] = board[i];
    }
    state->turn[0] = board[3];
    state->river[0] = board[4];
}

void update_tensors(
    const State* S, 
    torch::Tensor* hand, 
    torch::Tensor* flop, 
    torch::Tensor* turn, 
    torch::Tensor* river, 
    torch::Tensor* bet_fracs, 
    torch::Tensor* bet_status,
    int batch = 0
    ) {
    // Check if all pointers are valid
    if (!S || !hand || !flop || !turn || !river || !bet_fracs || !bet_status) {
        //DEBUG_NONE("Null pointer detected in update_tensors");
        throw std::runtime_error("Null pointer in update_tensors");
    }
    //DEBUG_NONE("All pointers valid");

    // Use accessors for more efficient data access
    auto hand_accessor = hand->accessor<int32_t, 2>();
    auto flop_accessor = flop->accessor<int32_t, 2>();
    auto turn_accessor = turn->accessor<int32_t, 2>();
    auto river_accessor = river->accessor<int32_t, 2>();
    auto bet_fracs_accessor = bet_fracs->accessor<float, 2>();
    auto bet_status_accessor = bet_status->accessor<float, 2>();
    //DEBUG_NONE("Accessors created");

    // Update hand cards (first two cards)
    for (int i = 0; i < 2; ++i) {
        hand_accessor[batch][i] = S->hand[i];
    }
    //DEBUG_NONE("Hand updated");

    // Update flop cards (next three cards)
    for (int i = 0; i < 3; ++i) {
        flop_accessor[batch][i] = S->flop[i];
    }
    //DEBUG_NONE("Flop updated");

    // Update turn card
    turn_accessor[batch][0] = S->turn[0];
    //DEBUG_NONE("Turn updated");

    // Update river card
    river_accessor[batch][0] = S->river[0];
    //DEBUG_NONE("River updated");

    // Update bet fractions
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_fracs_accessor[batch][i] = S->bet_fracs[i];
    }
    //DEBUG_NONE("Bet fractions updated");

    // Update bet status
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_status_accessor[batch][i] = S->bet_status[i];
    }
    //DEBUG_NONE("Bet status updated");

    //DEBUG_NONE("update_tensors completed successfully");
}
class ObjectPool {
private:
    std::vector<PokerEngine> engine_pool;
    std::vector<bool> engine_in_use;
    
    // Store the initialization parameters
    std::array<double, NUM_PLAYERS> starting_stacks;
    std::array<double, NUM_PLAYERS> antes;
    size_t pool_size;
    size_t idx;
    int starting_actor;
    double small_bet;
    double big_bet;

public:
    ObjectPool(
        size_t pool_size,
        const std::array<double, NUM_PLAYERS>& starting_stacks,
        const std::array<double, NUM_PLAYERS>& antes,
        int starting_actor,
        double small_bet,
        double big_bet
    ) : 
        starting_stacks(starting_stacks),
        pool_size(pool_size),
        idx(0),
        antes(antes),
        starting_actor(starting_actor),
        small_bet(small_bet),
        big_bet(big_bet),
        engine_in_use(pool_size, false)
    {
        // Initialize the engine pool
        engine_pool.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            engine_pool.emplace_back(
                starting_stacks,
                antes,
                starting_actor,
                small_bet,
                big_bet,
                false  // is_limit
            );
        }

    }

    PokerEngine* get_engine() {
        if (idx < pool_size) {
            engine_pool[idx].reset(
                starting_stacks,
                antes,
                starting_actor,
                small_bet,
                big_bet,
                false  // is_limit
            );
            return &engine_pool[idx++];
        }
        // If all engines are in use, create a new one
        engine_pool.emplace_back(
            starting_stacks,
            antes,
            starting_actor,
            small_bet,
            big_bet,
            false  // is_limit
        );
        idx++;
        pool_size++;
        //engine_in_use.push_back(true);
        return &engine_pool.back();
    }

    void release_engine(PokerEngine* engine) {
        // handle logic properly later
    }
};

constexpr double NULL_VALUE = -42.0;

// Adjusted Advantage struct with smart pointers
struct Advantage {
    // Values associated with each action
    std::array<double, NUM_ACTIONS> values;
    std::array<double, NUM_ACTIONS> strat;
    std::array<bool, NUM_ACTIONS> is_illegal;

    // Weak pointer to the parent Advantage to avoid circular references
    std::weak_ptr<Advantage> parent;
    int parent_action;

    // Shared pointer to the state
    std::shared_ptr<State> state;

    int depth;
    int unprocessed_children;

    // Delete copy constructor and copy assignment operator to prevent accidental copies
    Advantage(const Advantage&) = delete;
    Advantage& operator=(const Advantage&) = delete;

    // Constructor
    Advantage(
        const std::array<double, NUM_ACTIONS>& values_,
        const std::array<double, NUM_ACTIONS>& strat_,
        const std::array<bool, NUM_ACTIONS>& is_illegal_,
        const std::shared_ptr<Advantage>& parent_adv,
        int parent_action_,
        const std::shared_ptr<State>& state_,
        int depth_,
        int num_children
    ) :
        values(values_),
        strat(strat_),
        is_illegal(is_illegal_),
        parent(parent_adv),
        parent_action(parent_action_),
        state(state_),
        depth(depth_),
        unprocessed_children(num_children)
    {}
};


// Adjusted iterative_traverse function
void iterative_traverse(
    int thread_id,
    int player,
    std::array<void*, NUM_PLAYERS>& nets,
    int t,
    int traversals_per_thread,
    const std::array<double, NUM_PLAYERS>& starting_stacks,
    const std::array<double, NUM_PLAYERS>& antes,
    int starting_actor,
    double small_bet,
    double big_bet
) {
    // Initialize the initial game state
    PokerEngine initial_engine(
        starting_stacks,
        antes,
        starting_actor,
        small_bet,
        big_bet,
        false  // is_limit
    );

    ObjectPool object_pool(
        POOL_SIZE,
        starting_stacks,
        antes,
        starting_actor,
        small_bet,
        big_bet
    );

    // Initialize tensors for neural network inputs
    torch::Tensor hands = init_batched_hands(1);
    torch::Tensor flops = init_batched_flops(1);
    torch::Tensor turns = init_batched_turns(1);
    torch::Tensor rivers = init_batched_rivers(1);
    torch::Tensor bet_fracs = init_batched_fracs(1);
    torch::Tensor bet_status = init_batched_status(1);

    for (int traversal = 0; traversal < traversals_per_thread; ++traversal) {
        int num_advs = global_index.load();
        if (num_advs >= MAX_ADVS) break;

        DEBUG_NONE("Thread=" << thread_id << " Iter=" << traversal << " advs=" << num_advs);

        // Stack for DFS traversal: depth, engine, parent_advantage, parent_action
        std::stack<std::tuple<int, PokerEngine*, std::shared_ptr<Advantage>, int>> stack;

        // Stack of terminal advantages for backpropagation
        std::stack<std::shared_ptr<Advantage>> terminal_advs;

        // Deque to store all Advantage objects
        std::deque<std::shared_ptr<Advantage>> all_advs;

        // Initialize the root engine
        PokerEngine* root_engine = object_pool.get_engine();
        *root_engine = initial_engine.copy();  // Copy the initial engine state

        // Start traversal from the root
        stack.push({0, root_engine, nullptr, -1});

        while (!stack.empty()) {
            auto [depth, engine, parent_advantage, parent_action] = stack.top();
            stack.pop();

            // Terminal state or player not in the game
            if (!engine->get_game_status() || !engine->is_playing(player)) {
                std::array<double, NUM_PLAYERS> payoff = engine->get_payoffs();
                double bb = engine->get_big_blind();
                double final_value = payoff[player] / bb;
                object_pool.release_engine(engine);

                if (parent_advantage) {
                    // Update the parent's values for the action taken
                    parent_advantage->values[parent_action] = final_value;
                    parent_advantage->unprocessed_children--;
                    if (parent_advantage->unprocessed_children == 0) {
                        terminal_advs.push(parent_advantage);
                    }
                }
            }
            // Player's turn
            else if (engine->turn() == player) {
                auto state = std::make_shared<State>();
                get_state(*engine, state.get(), player);

                std::array<double, NUM_ACTIONS> strat{0.0};
                std::array<double, NUM_ACTIONS> values;
                values.fill(NULL_VALUE);
                std::array<bool, NUM_ACTIONS> is_illegal{false};

                // Determine legal actions
                for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                    if (!verify_action(engine, player, a)) is_illegal[a] = true;
                }

                int num_children = std::count(is_illegal.begin(), is_illegal.end(), false);

                // Create a new Advantage object
                auto adv_ptr = std::make_shared<Advantage>(
                    values,
                    strat,
                    is_illegal,
                    parent_advantage,
                    parent_action,
                    state,
                    depth + 1,
                    num_children
                );

                all_advs.push_back(adv_ptr);

                // For each legal action, proceed to the next state
                for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                    if (!is_illegal[a]) {
                        PokerEngine new_engine = engine->copy();
                        take_action(&new_engine, player, a);
                        stack.push({depth + 1, &new_engine, adv_ptr, static_cast<int>(a)});
                    }
                }
            }
            // Opponent's turn
            else {
                int actor = engine->turn();
                auto state = std::make_shared<State>();
                get_state(*engine, state.get(), player);
                update_tensors(state.get(), &hands, &flops, &turns, &rivers, &bet_fracs, &bet_status);

                void* net_ptr = nets[actor];
                torch::Tensor logits = deep_cfr_model_forward(net_ptr, hands, flops, turns, rivers, bet_fracs, bet_status);
                std::array<double, NUM_ACTIONS> strat = regret_match(logits);

                int action_index = sample_action(strat);
                while (!verify_action(engine, actor, action_index)) {
                    action_index = (action_index - 1 + NUM_ACTIONS) % NUM_ACTIONS;
                }

                take_action(engine, actor, action_index);
                stack.push({depth + 1, engine, parent_advantage, parent_action});
            }
        }

        // Begin batch inference
        DEBUG_NONE("batch inference time");
        DEBUG_NONE("num_advs = " << all_advs.size());

        size_t num_repeats = (all_advs.size() + TRAIN_BS - 1) / TRAIN_BS;
        DEBUG_NONE("num_repeats = " << num_repeats);
        size_t advs_idx = 0;

        torch::Tensor batched_hands = init_batched_hands(TRAIN_BS);
        torch::Tensor batched_flops = init_batched_flops(TRAIN_BS);
        torch::Tensor batched_turns = init_batched_turns(TRAIN_BS);
        torch::Tensor batched_rivers = init_batched_rivers(TRAIN_BS);
        torch::Tensor batched_fracs = init_batched_fracs(TRAIN_BS);
        torch::Tensor batched_status = init_batched_status(TRAIN_BS);

        for (size_t r = 0; r < num_repeats; ++r) {
            size_t batch_size = std::min(TRAIN_BS, all_advs.size() - advs_idx);

            for (size_t i = 0; i < batch_size; ++i) {
                auto& adv = all_advs[advs_idx];
                update_tensors(
                    adv->state.get(),
                    &batched_hands,
                    &batched_flops,
                    &batched_turns,
                    &batched_rivers,
                    &batched_fracs,
                    &batched_status,
                    i
                );
                advs_idx++;
            }

            // Perform inference to get strategies
            torch::Tensor logits = deep_cfr_model_forward(
                nets[player],
                batched_hands.slice(0, 0, batch_size),
                batched_flops.slice(0, 0, batch_size),
                batched_turns.slice(0, 0, batch_size),
                batched_rivers.slice(0, 0, batch_size),
                batched_fracs.slice(0, 0, batch_size),
                batched_status.slice(0, 0, batch_size)
            );

            torch::Tensor regrets = regret_match_batched(logits);
            auto regrets_accessor = regrets.accessor<float, 2>();

            // Update strategies in all_advs
            for (size_t i = 0; i < batch_size; ++i) {
                auto& adv = all_advs[advs_idx - batch_size + i];
                for (size_t j = 0; j < NUM_ACTIONS; ++j) {
                    adv->strat[j] = regrets_accessor[i][j];
                }
            }
        }

        // Calculate advantages and backpropagate
        DEBUG_NONE("batch advantage calc time");
        DEBUG_NONE("num_terminal_advs = " << terminal_advs.size());

        while (!terminal_advs.empty()) {
            auto terminal_adv = terminal_advs.top();
            terminal_advs.pop();

            // Compute expected value
            double ev = 0.0;
            for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                if (!terminal_adv->is_illegal[a]) {
                    ev += terminal_adv->values[a] * terminal_adv->strat[a];
                }
            }

            // Calculate advantages
            std::array<double, NUM_ACTIONS> adv_values;
            for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                if (!terminal_adv->is_illegal[a]) {
                    double ad = terminal_adv->values[a] - ev;
                    adv_values[a] = (ad > 0.0) ? ad : (1.0 / static_cast<double>(NUM_ACTIONS));
                } else {
                    adv_values[a] = 0.0;
                }
            }

            size_t add_idx = global_index.fetch_add(1);
            if (add_idx < MAX_SIZE) {
                global_advs[add_idx] = TraverseAdvantage{terminal_adv->state, t, adv_values};
            }

            // Backpropagate to parent
            if (auto parent_adv_ptr = terminal_adv->parent.lock()) {
                parent_adv_ptr->values[terminal_adv->parent_action] = ev;
                parent_adv_ptr->unprocessed_children--;

                if (parent_adv_ptr->unprocessed_children == 0) {
                    terminal_advs.push(parent_adv_ptr);
                }
            }
        }
    }
}


// more cfr traversal steps converges faster but is sample inefficient
// 3000 cfr steps * 500 cfr iters converges same as 100,000 and 500 cfr iters
// so estimate around ~1 million total steps budget
// around 10,000 training iter steps needed per iter
// the thing is CPU is faster than M3 Max chip
int main() {
    std::vector<std::array<void*, NUM_PLAYERS>> total_nets;
    std::array<void*, NUM_PLAYERS> init_player_nets{};
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        init_player_nets[i] = create_deep_cfr_model();
    }

    
    total_nets.push_back(init_player_nets);
    global_advs.resize(MAX_SIZE);

    // init concurrency
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads=1;

    std::vector<std::array<void*, NUM_PLAYERS>> thread_nets(num_threads);
    for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id) {
        for (size_t i = 0; i < NUM_PLAYERS; ++i) {
            thread_nets[thread_id][i] = create_deep_cfr_model();
        }
    }
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

    // Modify the thread_func to use iterative_traverse
    auto thread_func = [&](int traversals_per_thread, int thread_id, int player, int cfr_iter, std::array<void*, NUM_PLAYERS> nets) {
        try {
            torch::NoGradGuard guard;
            iterative_traverse(
                thread_id,
                player, 
                nets, 
                cfr_iter, 
                traversals_per_thread,
                starting_stacks, 
                antes,
                starting_actor,
                small_bet, 
                big_bet
            );
        }
        catch(const std::exception& e) {
            DEBUG_INFO(e.what());
        }
    };

    for (int cfr_iter=1; cfr_iter<CFR_ITERS+1; ++cfr_iter) {
        std::array<void*, NUM_PLAYERS> player_nets = total_nets[total_nets.size()-1];
        std::array<void*, NUM_PLAYERS> new_player_nets{create_deep_cfr_model()};
        for (int player=0; player<NUM_PLAYERS; ++player) {
            // spawn threads
            std::vector<std::thread> threads;
            DEBUG_NONE("num_threads = " << num_threads);
            // In the main function, modify the thread creation part:
            for(unsigned int thread_id = 0; thread_id < num_threads; ++thread_id) {
                int traversals_to_run = traversals_per_thread + (thread_id < remaining_traversals ? 1 : 0);
                threads.emplace_back([&thread_func, traversals_to_run, thread_id, player, cfr_iter, &thread_nets]() {
                    thread_func(traversals_to_run, thread_id, player, cfr_iter, thread_nets[thread_id]);
                });
            }
            // Wait for all threads to finish
            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }
            void* player_model = new_player_nets[player];
            DEBUG_NONE("CFR ITER = " << cfr_iter);
            DEBUG_NONE("COLLECTED ADVS = " << global_index.load());
            // todo train
            // draw training samples
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            std::shuffle(global_advs.begin(), global_advs.begin() + global_index.load(), rng);
            std::vector<TraverseAdvantage> training_advs;
            for (size_t i = 0; i < global_index.load(); ++i) {
                training_advs.push_back(global_advs[i]);
            }

            DEBUG_NONE("TOTAL TRAINING SAMPLES = " << training_advs.size());
            int batch_repeat = training_advs.size() / TRAIN_BS;
            int advs_idx = 0;
            torch::Tensor batched_hands = init_batched_hands(TRAIN_BS);
            torch::Tensor batched_flops = init_batched_flops(TRAIN_BS);
            torch::Tensor batched_turns = init_batched_turns(TRAIN_BS);
            torch::Tensor batched_rivers = init_batched_rivers(TRAIN_BS);
            torch::Tensor batched_fracs = init_batched_fracs(TRAIN_BS);
            torch::Tensor batched_status = init_batched_status(TRAIN_BS);
            torch::Tensor batched_advs = init_batched_advs(TRAIN_BS);
            torch::Tensor batched_iters = init_batched_iters(TRAIN_BS);
            for (int _ = 0; _ < batch_repeat; ++_) {
                for (size_t i = 0; i < TRAIN_BS; ++i) {
                    if (advs_idx >= training_advs.size()) {
                        break;
                    }
                    State *S = training_advs[advs_idx].state.get();
                    update_tensors(
                        S, 
                        &batched_hands, 
                        &batched_flops, 
                        &batched_turns,
                        &batched_rivers,
                        &batched_fracs,
                        &batched_status,
                        i
                    ); 
                    auto batched_advs_accessor = batched_advs.accessor<float, 2>();
                    for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                        batched_advs_accessor[i][a] = training_advs[advs_idx].advantages[a];
                    }
                    int iteration = training_advs[advs_idx].iteration;
                    auto batched_iters_accessor = batched_iters.accessor<float, 2>();
                    batched_iters_accessor[i][0] = iteration;
                    advs_idx++;
                }

                // train
                torch::optim::Adam optimizer(get_model_parameters(player_model));

                // maximum 50 million samples
                for (size_t epoch = 0; epoch < TRAIN_EPOCHS; ++epoch) {
                    optimizer.zero_grad();
        
                    torch::Tensor pred = deep_cfr_model_forward(
                        player_model, 
                        batched_hands,
                        batched_flops,
                        batched_turns,
                        batched_rivers, 
                        batched_fracs, 
                        batched_status
                    );

                    torch::Tensor loss = torch::nn::functional::mse_loss(
                        pred, 
                        batched_advs,
                        torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)
                    );

                    loss *= batched_iters;

                    torch::Tensor batch_mean_loss = loss.mean(1).mean();
                    
                    batch_mean_loss.backward();
                    optimizer.step();

                    // Log the mean loss every epoch
                    DEBUG_NONE("Epoch " << epoch + 1 << "/" << TRAIN_EPOCHS << ", Loss: " << batch_mean_loss.item<float>());
                }
            }

            // todo eval - implement game sim in eval.cpp
            //double eval_mbb = evaluate(player_model, player);
        }
    }
}
