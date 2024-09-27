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
            auto logits = nets[player]->forward(
                batched_hands,
                batched_flops,
                batched_turns,
                batched_rivers,
                batched_fracs,
                batched_status
            );

            auto regrets = regret_match_batched(logits);
            auto regrets_a = regrets.accessor<float, 2>();

            // Then update the loop that uses regrets:
            for (size_t i = 0; i < batch_size; ++i) {
                auto& adv = all_advs[advs_idx-batch_size+i];

                if (!adv) {
                    DEBUG_NONE("Error: Null pointer encountered at index " << advs_idx);
                    continue;
                }
                for (size_t j = 0; j < NUM_ACTIONS; ++j) {
                    float regret = regrets_a[i][j];
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

void profile_model(DeepCFRModel& model, int batch_size = 1000, int num_iterations = 100) {
    // Prepare input tensors
    auto hands = init_batched_hands(batch_size);
    auto flops = init_batched_flops(batch_size);
    auto turns = init_batched_turns(batch_size);
    auto rivers = init_batched_rivers(batch_size);
    auto bet_fracs = init_batched_fracs(batch_size);
    auto bet_status = init_batched_status(batch_size);

    // Function to run inference and measure time
    auto run_inference = [&](torch::Device device) {
        model->to(device);
        hands = hands.to(device);
        flops = flops.to(device);
        turns = turns.to(device);
        rivers = rivers.to(device);
        bet_fracs = bet_fracs.to(device);
        bet_status = bet_status.to(device);

        torch::NoGradGuard no_grad;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            auto output = model->forward(hands, flops, turns, rivers, bet_fracs, bet_status);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / num_iterations;
    };

    // Profile CPU
    double cpu_time = run_inference(torch::kCPU);
    std::cout << "CPU average inference time: " << cpu_time << " ms" << std::endl;

    // Profile CUDA if available
    if (torch::cuda::is_available()) {
        double cuda_time = run_inference(torch::kCUDA);
        std::cout << "CUDA average inference time: " << cuda_time << " ms" << std::endl;
        
        // Calculate speedup
        double speedup = cpu_time / cuda_time;
        std::cout << "CUDA speedup: " << speedup << "x" << std::endl;
    } else {
        std::cout << "CUDA is not available on this system." << std::endl;
    }
}

int main() {
    //DeepCFRModel model;
    //profile_model(model, 1000, 100);
    //exit(-1);
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

            auto batched_advs_a = batched_advs.accessor<float, 2>();
            auto batched_iters_a = batched_iters.accessor<int, 2>();

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
                        batched_advs_a[i][a] = training_advs[advs_idx].advantages[a];
                    }
                    int iteration = training_advs[advs_idx].iteration;
                    batched_iters_a[i][0] = static_cast<int>(iteration);
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

            double eval_mbb = evaluate(train_net, player);

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