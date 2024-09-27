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
        DEBUG_NONE("Null pointer detected in update_tensors");
        DEBUG_NONE("S: " << (void*)S);
        DEBUG_NONE("hand: " << (void*)hand);
        DEBUG_NONE("flop: " << (void*)flop);
        DEBUG_NONE("turn: " << (void*)turn);
        DEBUG_NONE("river: " << (void*)river);
        DEBUG_NONE("bet_fracs: " << (void*)bet_fracs);
        DEBUG_NONE("bet_status: " << (void*)bet_status);
        throw std::runtime_error("Null pointer in update_tensors");
    }

    if (batch < 0 || batch >= hand->size(0)) {
        DEBUG_NONE("Invalid index passed to update_tensors");
        throw std::runtime_error("Invalid index passed to update_tensors");
    }

    // Get data pointers
    int32_t* hand_ptr = hand->data_ptr<int32_t>();
    int32_t* flop_ptr = flop->data_ptr<int32_t>();
    int32_t* turn_ptr = turn->data_ptr<int32_t>();
    int32_t* river_ptr = river->data_ptr<int32_t>();
    float* bet_fracs_ptr = bet_fracs->data_ptr<float>();
    float* bet_status_ptr = bet_status->data_ptr<float>();

    // Calculate offsets
    int64_t hand_offset = batch * hand->size(1);
    int64_t flop_offset = batch * flop->size(1);
    int64_t turn_offset = batch * turn->size(1);
    int64_t river_offset = batch * river->size(1);
    int64_t bet_fracs_offset = batch * bet_fracs->size(1);
    int64_t bet_status_offset = batch * bet_status->size(1);

    // Update hand cards (first two cards)
    for (int i = 0; i < 2; ++i) {
        hand_ptr[hand_offset + i] = S->hand[i];
    }

    // Update flop cards (next three cards)
    for (int i = 0; i < 3; ++i) {
        flop_ptr[flop_offset + i] = S->flop[i];
    }

    // Update turn card
    turn_ptr[turn_offset] = S->turn[0];

    // Update river card
    river_ptr[river_offset] = S->river[0];

    // Update bet fractions
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_fracs_ptr[bet_fracs_offset + i] = S->bet_fracs[i];
    }

    // Update bet status
    for (int i = 0; i < NUM_PLAYERS*MAX_ROUND_BETS*4; ++i) {
        bet_status_ptr[bet_status_offset + i] = S->bet_status[i];
    }
}

constexpr double NULL_VALUE = -42.0;


// Adjusted Advantage struct with smart pointers
struct Advantage {
    std::array<double, NUM_ACTIONS> values;
    std::array<double, NUM_ACTIONS> strat;
    std::array<bool, NUM_ACTIONS> is_illegal;
    std::weak_ptr<Advantage> parent;
    int parent_action;
    std::shared_ptr<State> state;
    int depth;
    int unprocessed_children;

    Advantage(const Advantage&) = delete;
    Advantage& operator=(const Advantage&) = delete;

    Advantage(
        const std::array<double, NUM_ACTIONS>& values_ = {},
        const std::array<double, NUM_ACTIONS>& strat_ = {},
        const std::array<bool, NUM_ACTIONS>& is_illegal_ = {},
        const std::shared_ptr<Advantage>& parent_adv = nullptr,
        int parent_action_ = -1,
        const std::shared_ptr<State>& state_ = nullptr,
        int depth_ = 0,
        int num_children = 0
    ) :
        values(values_),
        strat(strat_),
        is_illegal(is_illegal_),
        parent(parent_adv),
        parent_action(parent_action_),
        state(state_),
        depth(depth_),
        unprocessed_children(num_children)
    {
        // Initialize values to 0 if not provided
        if (values_.empty()) {
            values.fill(0.0);
        }

        // Initialize strat to uniform distribution if not provided
        if (strat_.empty()) {
            double uniform_prob = 1.0 / NUM_ACTIONS;
            strat.fill(uniform_prob);
        }

        // Initialize is_illegal to false if not provided
        if (is_illegal_.empty()) {
            is_illegal.fill(false);
        }
    }
};


void print_strategy(const std::array<double, NUM_ACTIONS>& strat, int chosen_act) {
    const std::array<std::string, NUM_ACTIONS> action_names = {"Fold", "Check/Call", "Bet/Raise"};
    
    std::cout << "\nStrategy probabilities:\n";
    std::cout << std::string(30, '-') << "\n";
    std::cout << std::setw(15) << "Action" << " | " << std::setw(10) << "Probability" << "\n";
    std::cout << std::string(30, '-') << "\n";
    
    for (size_t i = 0; i < NUM_ACTIONS; ++i) {
        std::cout << std::setw(15) << action_names[i] << " | " 
                  << std::setw(10) << std::fixed << std::setprecision(4) << strat[i] << "\n";
    }
    
    std::cout << std::string(30, '-') << "\n";
    std::cout << std::setw(15) << "Chosen action:" << std::to_string(chosen_act) << "\n";
    std::cout << std::string(30, '-') << "\n";
}

// Adjusted iterative_traverse function
// past: 0.3 iter / sec
// now: 

// so one possible cause for use-after-free is if in update_tensor
// instead of updating the memory location at each batch_tensor,
// im updating the batch_tensor's pointer to stack pointers in Advantage->State
void iterative_traverse(
    int thread_id,
    int player,
    std::array<DeepCFRModel, NUM_PLAYERS>& nets,
    int t,
    int traversals_per_thread,
    const std::array<double, NUM_PLAYERS>& starting_stacks,
    const std::array<double, NUM_PLAYERS>& antes,
    int starting_actor,
    double small_bet,
    double big_bet
) {
    PokerEngine initial_engine(starting_stacks, antes, starting_actor, small_bet, big_bet, false);

    // auto doesn't fix it
    auto hands = init_batched_hands(1);
    auto flops = init_batched_flops(1);
    auto turns = init_batched_turns(1);
    auto rivers = init_batched_rivers(1);
    auto bet_fracs = init_batched_fracs(1);
    auto bet_status = init_batched_status(1);

    auto batched_hands = init_batched_hands(TRAIN_BS);
    auto batched_flops = init_batched_flops(TRAIN_BS);
    auto batched_turns = init_batched_turns(TRAIN_BS);
    auto batched_rivers = init_batched_rivers(TRAIN_BS);
    auto batched_fracs = init_batched_fracs(TRAIN_BS);
    auto batched_status = init_batched_status(TRAIN_BS);

    for (int traversal = 0; traversal < traversals_per_thread; ++traversal) {
        int num_advs = global_index.load();
        if (num_advs >= MAX_ADVS) break;

        DEBUG_NONE("Thread=" << thread_id << " Iter=" << traversal << " advs=" << num_advs);

        std::stack<std::tuple<int, PokerEngine, std::shared_ptr<Advantage>, int>> stack;
        std::stack<std::shared_ptr<Advantage>> terminal_advs;
        std::deque<std::shared_ptr<Advantage>> all_advs;

        stack.push({0, initial_engine, nullptr, -1});

        while (!stack.empty()) {
            auto [depth, engine, parent_advantage, parent_action] = stack.top();
            stack.pop();

            if (!engine.get_game_status() || !engine.is_playing(player)) {
                std::array<double, NUM_PLAYERS> payoff = engine.get_payoffs();
                double bb = engine.get_big_blind();
                double final_value = payoff[player] / bb;

                if (parent_advantage != nullptr) {
                    parent_advantage->values[parent_action] = final_value;
                    parent_advantage->unprocessed_children--;
                    if (parent_advantage->unprocessed_children == 0) {
                        terminal_advs.push(parent_advantage);
                    }
                }
            }
            else if (engine.turn() == player) {
                auto state = std::make_shared<State>();
                get_state(engine, state.get(), player);

                std::array<bool, NUM_ACTIONS> is_illegal{false};
                for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                    if (!verify_action(&engine, player, a)) is_illegal[a] = true;
                }

                int num_children = std::count(is_illegal.begin(), is_illegal.end(), false);

                auto adv_ptr = std::make_shared<Advantage>(
                    std::array<double, NUM_ACTIONS>{},
                    std::array<double, NUM_ACTIONS>{}, 
                    is_illegal,
                    parent_advantage,
                    parent_action,
                    state,
                    depth + 1,
                    num_children
                );

                all_advs.push_back(adv_ptr);

                for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                    if (!is_illegal[a]) {
                        PokerEngine new_engine = engine.copy();
                        take_action(&new_engine, player, a);
                        stack.push({depth + 1, new_engine, adv_ptr, static_cast<int>(a)});
                    }
                }
            }
            else {
                int actor = engine.turn();
                auto state = std::make_shared<State>();
                get_state(engine, state.get(), player);
                update_tensors(state.get(), &hands, &flops, &turns, &rivers, &bet_fracs, &bet_status);

                DeepCFRModel net_ptr = nets[actor];
                if (net_ptr.get() == nullptr) {
                    throw std::runtime_error("net_ptr is nullptr for actor " + std::to_string(actor));
                }
                torch::Tensor logits = net_ptr->forward(hands, flops, turns, rivers, bet_fracs, bet_status);
                std::array<double, NUM_ACTIONS> strat = regret_match(logits);

                int action_index = sample_action(strat);
                while (!verify_action(&engine, actor, action_index)) {
                    action_index = (action_index - 1 + NUM_ACTIONS) % NUM_ACTIONS;
                }

                take_action(&engine, actor, action_index);
                stack.push(std::make_tuple(depth + 1, std::move(engine), parent_advantage, parent_action));
            }
        }

        // Begin batch inference
        DEBUG_NONE("batch inference time");
        DEBUG_NONE("num_advs = " << all_advs.size());

        size_t num_repeats = (all_advs.size() + TRAIN_BS - 1) / TRAIN_BS;
        DEBUG_NONE("num_repeats = " << num_repeats);
        size_t advs_idx = 0;

        for (size_t r = 0; r < num_repeats; ++r) {
            size_t batch_size = std::min(TRAIN_BS, all_advs.size()-advs_idx);

            for (size_t i = 0; i < batch_size; ++i) {
                //DEBUG_NONE("udt");
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

            // accessing freed pointer error
            // occurs in both single and multi thread
            // checking if this is libtorch problem or my code problem
            // check if model_ptr changed
            DEBUG_NONE("batch_size = " << batch_size);
            //DEBUG_NONE("model_ptr = " << nets[player]);
            torch::Tensor logits = nets[player]->forward(
                batched_hands,
                batched_flops,
                batched_turns,
                batched_rivers,
                batched_fracs,
                batched_status
            );

            torch::Tensor regrets = regret_match_batched(logits);
            float* regrets_pointer = regrets.data_ptr<float>();

            // Update strategies in all_advs
            for (size_t i = 0; i < batch_size; ++i) {
                int32_t batch_offset = i * regrets.size(1);
                auto& adv = all_advs[advs_idx-batch_size+i];

                // Check if adv is a valid pointer
                if (!adv) {
                    DEBUG_NONE("Error: Null pointer encountered at index " << advs_idx);
                    continue;  // Skip this iteration
                }
                for (size_t j = 0; j < NUM_ACTIONS; ++j) {
                    float regret = regrets_pointer[batch_offset + j];
                    adv->strat[j] = regret;
                }
            }
        }


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
                if (parent_adv_ptr == nullptr || parent_adv_ptr == terminal_adv) continue;
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
    //DeepCFRModel* model = new DeepCFRModel();
    // each thread gets a copy of latest model
    // we just need an array
    std::vector<std::array<DeepCFRModel, NUM_PLAYERS>> nets(NUM_THREADS);
    DEBUG_NONE("NUM_PLAYERS = " << NUM_PLAYERS);
    for (size_t i=0; i<NUM_THREADS; ++i) {
        for (size_t j=0; j<NUM_PLAYERS; ++j) {
            DeepCFRModel model;
            nets[i][j] = model;
            if (nets[i][j].get() == nullptr) DEBUG_NONE("null ptr at init");
            //else DEBUG_NONE("player " << j << " = " << nets[i][j]);
        }
    }
    
    global_advs.resize(MAX_SIZE);

    int traversals_per_thread = NUM_TRAVERSALS / NUM_THREADS;
    int remaining_traversals = NUM_TRAVERSALS % NUM_THREADS;
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
    auto thread_func = [&](int traversals_per_thread, int thread_id, int player, int cfr_iter, std::array<DeepCFRModel, NUM_PLAYERS>& nets) {
        try {
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
        for (int player=0; player<NUM_PLAYERS; ++player) {
            DEBUG_NONE("collecting samples for player = " << player);
            // spawn threads
            std::vector<std::thread> threads;
            // In the main function, modify the thread creation part:
            // Modify the thread creation part
            for (unsigned int thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
                int traversals_to_run = traversals_per_thread + (thread_id < remaining_traversals ? 1 : 0);
                
                // Create a reference to nets[thread_id]
                std::array<DeepCFRModel, NUM_PLAYERS>& thread_nets = nets[thread_id];
                
                // Capture thread_nets by reference
                threads.emplace_back(
                    [&thread_func, traversals_to_run, thread_id, player, cfr_iter, &thread_nets]() {
                        thread_func(traversals_to_run, thread_id, player, cfr_iter, thread_nets);
                    }
                );
            } 
            // Wait for all threads to finish
            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }

            // define fresh net to train
            DeepCFRModel train_net;
            DEBUG_NONE("CFR ITER = " << cfr_iter);
            DEBUG_NONE("COLLECTED ADVS = " << global_index.load());
            // todo train
            // draw training samples
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            std::shuffle(global_advs.begin(), global_advs.begin() + global_index.load(), rng);
            std::vector<TraverseAdvantage> training_advs;
            size_t train_iters = std::min(TRAIN_ITERS, static_cast<size_t>(global_index.load()));
            for (size_t i = 0; i < train_iters; ++i) {
                training_advs.push_back(global_advs[i]);
            }

            DEBUG_NONE("TOTAL TRAINING SAMPLES = " << training_advs.size());
            int batch_repeat = train_iters / TRAIN_BS;
            int advs_idx = 0;

            auto batched_hands = init_batched_hands(TRAIN_BS);
            auto batched_flops = init_batched_flops(TRAIN_BS);
            auto batched_turns = init_batched_turns(TRAIN_BS);
            auto batched_rivers = init_batched_rivers(TRAIN_BS);
            auto batched_fracs = init_batched_fracs(TRAIN_BS);
            auto batched_status = init_batched_status(TRAIN_BS);
            auto batched_advs = init_batched_advs(TRAIN_BS);
            auto batched_iters = init_batched_iters(TRAIN_BS);

            auto batched_advs_ptr = batched_advs.data_ptr<float>();
            auto batched_iters_ptr = batched_iters.data_ptr<float>();

            for (int _ = 0; _ < batch_repeat; ++_) {
                for (size_t i = 0; i < TRAIN_BS; ++i) {
                    if (advs_idx >= train_iters) {
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

                    for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                        batched_advs_ptr[i * NUM_ACTIONS + a] = training_advs[advs_idx].advantages[a];
                    }
                    int iteration = training_advs[advs_idx].iteration;
                    batched_iters_ptr[i] = static_cast<float>(iteration);
                    advs_idx++;
                }

                // train
                //torch::autograd::GradMode::set_enabled(true);
                torch::optim::Adam optimizer(train_net->parameters());

                // maximum 50 million samples
                for (size_t epoch = 0; epoch < TRAIN_EPOCHS; ++epoch) {
                    train_net->zero_grad();
        
                    auto pred = train_net->forward(
                        //train_net, 
                        batched_hands,
                        batched_flops,
                        batched_turns,
                        batched_rivers, 
                        batched_fracs, 
                        batched_status
                    );

                    auto loss = torch::nn::functional::mse_loss(
                        pred, 
                        batched_advs,
                        torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)
                    );

                    loss *= batched_iters;

                    auto batch_mean_loss = loss.mean(1).mean();
                    
                    batch_mean_loss.backward();
                    optimizer.step();

                    // Log the mean loss every epoch
                    DEBUG_NONE("Epoch " << epoch + 1 << "/" << TRAIN_EPOCHS << ", Loss: " << batch_mean_loss.item<float>());
                }
            }

            //double eval_mbb = evaluate(player_model, player);

            // todo save nets
            DEBUG_NONE("saving nets..");

            std::filesystem::path current_path = std::filesystem::current_path();
            std::string save_path = (current_path / "models" / std::to_string(cfr_iter) / std::to_string(player) / "model.pt").string();
            std::filesystem::create_directories(std::filesystem::path(save_path).parent_path());
            torch::save(train_net, save_path);
            DEBUG_NONE("successfully saved nets");

            if (std::filesystem::exists(save_path)) {
                DEBUG_NONE("File successfully created at " << save_path);
            } else {
                DEBUG_NONE("File was not created at " << save_path);
                throw std::runtime_error("");
            }

            // todo replace nets with trained nets
            for (size_t i=0; i<NUM_THREADS; ++i) {
                DeepCFRModel model;
                torch::load(model, save_path);
                nets[i][player] = model;
            }
            DEBUG_NONE("loaded trained nets into nets");
        }
    }
}

