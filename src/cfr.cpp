#include "cfr.h"
#include "debug.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include <tuple>
#include <thread>
#include <atomic>
#include <set>
#include <algorithm>
#include <cstdlib> 
#include <ctime>   
#include <chrono>
#include <iomanip> 
#include <cmath>
#include <sstream> 
#include <memory>

struct RandInit {
    RandInit() { std::srand(static_cast<unsigned int>(std::time(nullptr))); }
} rand_init;

std::array<std::mutex, NUM_PLAYERS> player_advs_mutex;
std::array<std::vector<TraverseAdvantage>, NUM_PLAYERS> global_player_advs{};
std::array<std::atomic<size_t>, NUM_PLAYERS> total_advs{};
std::atomic<size_t> cfr_iter_advs(0);
torch::Device cpu_device(torch::kCPU);
torch::Device gpu_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, 0);
constexpr double NULL_VALUE = -42.0;

// Replace direct access with this function:
void safe_add_advantage(int player, const TraverseAdvantage& adv) {
    std::lock_guard<std::mutex> lock(player_advs_mutex[player]);
    size_t current_total = total_advs[player].load();
    
    if (current_total < MAX_SIZE) {
        global_player_advs[player][current_total] = adv;
        total_advs[player].fetch_add(1);
    } else {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, current_total - 1);
        size_t r = dist(rng); // Use thread-safe random number
        global_player_advs[player][r] = adv;
    }
}

struct Advantage {
    std::array<double, NUM_ACTIONS> values;
    std::array<double, NUM_ACTIONS> strat;
    std::array<bool, NUM_ACTIONS> is_illegal;
    std::weak_ptr<Advantage> parent;
    int parent_action;
    State state;
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
        const State state_ = {},
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
        if (values_.empty()) {
            values.fill(0.0);
        }

        if (strat_.empty()) {
            double uniform_prob = 1.0 / NUM_ACTIONS;
            strat.fill(uniform_prob);
        }

        if (is_illegal_.empty()) {
            is_illegal.fill(false);
        }
    }
};

auto format_scientific = [](int number) -> std::string {
    if (number == 0) {
        return "0.0e0";
    }
    double mantissa = number;
    int exponent = 0;

    exponent = static_cast<int>(std::floor(std::log10(std::abs(mantissa))));
    mantissa /= std::pow(10, exponent);
    mantissa = std::round(mantissa * 10.0) / 10.0;  // Round to one decimal place

    // Handle cases where rounding affects the mantissa and exponent
    if (mantissa >= 10.0) {
        mantissa /= 10.0;
        exponent += 1;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << mantissa << "e" << exponent;
    return oss.str();
};

int get_opp(int player) {
    if (player == 1) return 0;
    return 1;
}

void iterative_traverse(
    int thread_id,
    int player,
    std::array<std::string, NUM_PLAYERS> model_paths,
    int t,
    int traversals_per_thread,
    const std::array<double, NUM_PLAYERS>& starting_stacks,
    const std::array<double, NUM_PLAYERS>& antes,
    int starting_actor,
    double small_bet,
    double big_bet
) {
    //torch::NoGradGuard no_grad_guard;
    PokerEngine initial_engine(starting_stacks, antes, starting_actor, small_bet, big_bet, false);

    auto hands = init_batched_hands(1);
    auto flops = init_batched_flops(1);
    auto turns = init_batched_turns(1);
    auto rivers = init_batched_rivers(1);
    auto bet_fracs = init_batched_fracs(1);
    auto bet_status = init_batched_status(1);

    auto batched_hands = init_batched_hands(TRAVERSAL_BS);
    auto batched_flops = init_batched_flops(TRAVERSAL_BS);
    auto batched_turns = init_batched_turns(TRAVERSAL_BS);
    auto batched_rivers = init_batched_rivers(TRAVERSAL_BS);
    auto batched_fracs = init_batched_fracs(TRAVERSAL_BS);
    auto batched_status = init_batched_status(TRAVERSAL_BS);

    DEBUG_NONE("loading");
    DeepCFRModel opp_net;
    if (t>1) {
        torch::load(opp_net, model_paths[get_opp(player)]);
    }
    opp_net->to(cpu_device);
    opp_net->eval();
    DEBUG_NONE("loaded");
    DEBUG_NONE("moved to eval");

    for (int traversal = 0; traversal < traversals_per_thread; ++traversal) {
        int n_total_advs = total_advs[player].load();
        int n_cfr_advs = cfr_iter_advs.load();
        DEBUG_NONE("Thread=" << thread_id
               << " Iter=" << traversal
               << " Cfr_iter_advs=" << format_scientific(n_cfr_advs)
               << " Total_advs=" << format_scientific(n_total_advs));
        // Reset manipulators to default formatting if needed
        std::stack<std::tuple<int, PokerEngine, std::shared_ptr<Advantage>, int>> stack;
        std::stack<std::shared_ptr<Advantage>> terminal_advs;
        std::deque<std::shared_ptr<Advantage>> all_advs;

        stack.push({0, initial_engine, nullptr, -1});

        while (!stack.empty()) {
            //if (cfr_iter_advs.load() >= CFR_MAX_SIZE) {
            if (all_advs.size() >= CFR_MAX_SIZE) {
                DEBUG_NONE("exceeds_max = True");
                //return;
                break;
            }
            
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
                    *state,
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
                // opponent case
                int actor = engine.turn();
                auto state = std::make_shared<State>();
                get_state(engine, state.get(), player);
                update_tensors(*(state.get()), hands, flops, turns, rivers, bet_fracs, bet_status);

                if (opp_net.get() == nullptr) {
                    throw std::runtime_error("net_ptr is nullptr for actor " + std::to_string(actor));
                }
                torch::Tensor logits = opp_net->forward(
                    hands, 
                    flops, 
                    turns, 
                    rivers, 
                    bet_fracs, 
                    bet_status
                );
                std::array<double, NUM_ACTIONS> strat = regret_match(logits);

                int action_index = sample_action(strat);
                while (!verify_action(&engine, actor, action_index)) {
                    action_index = (action_index - 1 + NUM_ACTIONS) % NUM_ACTIONS;
                }

                take_action(&engine, actor, action_index);
                stack.push(std::make_tuple(depth + 1, std::move(engine), parent_advantage, parent_action));
            }
        }
        DeepCFRModel gpu_net;
        if (t > 1) {
            torch::load(gpu_net, model_paths[player]);
        }
        gpu_net->to(gpu_device);
        gpu_net->eval();
        // Begin batch inference
        DEBUG_NONE("batch inference time");
        DEBUG_NONE("num_advs = " << all_advs.size());
        DEBUG_NONE("cfr_iter_advs = " << cfr_iter_advs.load());
        if (cfr_iter_advs.load() >= CFR_MAX_SIZE) {
            DEBUG_NONE("exceeds_max = True");
            return;
        } else {
            cfr_iter_advs.fetch_add(all_advs.size());
        }

        size_t num_repeats = (all_advs.size() + TRAVERSAL_BS - 1) / TRAVERSAL_BS;
        DEBUG_NONE("num_repeats = " << num_repeats);
        size_t advs_idx = 0;

        for (size_t r = 0; r < num_repeats; ++r) {
            size_t batch_size = std::min(TRAVERSAL_BS, all_advs.size()-advs_idx);

            for (size_t i = 0; i < batch_size; ++i) {
                //DEBUG_NONE("udt");
                auto& adv = all_advs[advs_idx];
                update_tensors(
                    adv->state,
                    batched_hands,
                    batched_flops,
                    batched_turns,
                    batched_rivers,
                    batched_fracs,
                    batched_status,
                    i 
                );
                advs_idx++;
            }

            auto logits = gpu_net->forward(
                batched_hands.slice(0,0,batch_size).to(gpu_device),
                batched_flops.slice(0,0,batch_size).to(gpu_device),
                batched_turns.slice(0,0,batch_size).to(gpu_device),
                batched_rivers.slice(0,0,batch_size).to(gpu_device),
                batched_fracs.slice(0,0,batch_size).to(gpu_device),
                batched_status.slice(0,0,batch_size).to(gpu_device))
            .to(cpu_device);

            auto regrets = regret_match_batched(logits);
            auto regrets_a = regrets.accessor<float, 2>();

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

            safe_add_advantage(player, TraverseAdvantage{terminal_adv->state, t, adv_values});

            if (terminal_adv->parent.expired()) {
                // Handle case where parent no longer exists
                continue;
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

void init_constants_log(std::string constant_log_file) {
    std::ifstream constants_file("../src/constants.h");
    std::ofstream const_log_file(constant_log_file);
    if (constants_file.is_open() && const_log_file.is_open()) {
        const_log_file << constants_file.rdbuf();
    } else {
        std::cerr << "Failed to open constants.h or constants log file." << std::endl;
    }
}

int main() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *ltm = std::localtime(&now_time);

    std::stringstream ss;
    ss << std::put_time(ltm, "%Y%m%d%H%M%S");
    std::string train_start_datetime = ss.str();

    std::string run_dir = "../out/" + train_start_datetime;
    std::filesystem::create_directories(run_dir);
    std::string logfile = run_dir + "/train.log";
    std::string const_log_filename = run_dir + "/const.log";
    std::filesystem::path current_path = run_dir;
    init_constants_log(const_log_filename);
    
    int traversals_per_thread = NUM_TRAVERSALS / NUM_THREADS;
    int remaining_traversals = NUM_TRAVERSALS % NUM_THREADS;
    DEBUG_WRITE(logfile, "Traversals per thread: " << traversals_per_thread);
    DEBUG_NONE("Remaining traversals: " << remaining_traversals);
    std::array<double, NUM_PLAYERS> starting_stacks{};
    std::array<double, NUM_PLAYERS> antes{};
    for (size_t i=0; i<NUM_PLAYERS; ++i) {
        starting_stacks[i] = 100.0;
        antes[i] = 0.0;
        global_player_advs[i].resize(MAX_SIZE);
    } 
    double small_bet = 0.5;
    double big_bet = 1.0;
    int starting_actor = 0;

    auto thread_func = [&](int traversals_per_thread, int thread_id, int player, int cfr_iter, std::array<std::string, NUM_PLAYERS> model_paths) {
        try {
            iterative_traverse(
                thread_id,
                player, 
                model_paths, 
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

    for (int cfr_iter=1; cfr_iter<CFR_ITERS+1; ++cfr_iter) {
        // init models
        std::vector<std::array<DeepCFRModel, NUM_PLAYERS>> nets(NUM_THREADS);
        std::array<std::string, NUM_PLAYERS> model_paths{"null"};
        // load and assign models
        for (int player=0; player<NUM_PLAYERS; ++player) {
            if (cfr_iter > 1) {
                // Then load the new model and set the references
                int sampled_iter = sample_iter(static_cast<size_t>(cfr_iter-1));
                std::filesystem::path iter_model_path = current_path;
                iter_model_path /= std::to_string(sampled_iter);
                iter_model_path /= std::to_string(player);
                iter_model_path /= "model.pt";
                std::string path_str = iter_model_path.string(); 
                model_paths[player] = path_str; 
            } 
        }

        for (int player=0; player<NUM_PLAYERS; ++player) {
            // init threads
            std::vector<std::thread> threads;
            for (unsigned int thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
                int traversals_to_run = traversals_per_thread + (thread_id < remaining_traversals ? 1 : 0);
                
                DEBUG_NONE("spawning thread");
                // Capture thread_nets by reference
                threads.emplace_back(
                    [&thread_func, traversals_to_run, thread_id, player, cfr_iter, model_paths]() {
                        thread_func(traversals_to_run, thread_id, player, cfr_iter, model_paths);
                    }
                );
            } 

            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }

            // fresh net to train
            DeepCFRModel train_net;
            train_net->to(gpu_device);
            train_net->train();
            DEBUG_NONE("CFR ITER = " << cfr_iter);
            DEBUG_WRITE(logfile, "CFR ITER = " << cfr_iter);
            DEBUG_NONE("COLLECTED ADVS = " << cfr_iter_advs.load());
            DEBUG_WRITE(logfile, "COLLECTED ADVS = " << total_advs[player].load());
            DEBUG_NONE("PLAYER = " << player);
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            size_t train_bs = std::min(TRAIN_BS, cfr_iter_advs.load());
            DEBUG_NONE("TRAIN_BS = " << train_bs);
            if (train_bs == 0) continue;
            cfr_iter_advs.store(0, std::memory_order_seq_cst);
            size_t total_advs_player = total_advs[player].load();
            const size_t POOL_SIZE = std::min(train_bs * 4, total_advs_player);
            std::vector<size_t> pool(POOL_SIZE);
            std::iota(pool.begin(), pool.end(), 0);
            std::shuffle(pool.begin(), pool.end(), rng);
            size_t window_start = 0;
            torch::optim::Adam optimizer(train_net->parameters(), torch::optim::AdamOptions(0.001));
            // Inside the training loop
            for (size_t train_iter = 0; train_iter < TRAIN_ITERS; ++train_iter) {
                // Rotate and shuffle window
                std::rotate(pool.begin(), 
                        pool.begin() + window_start, 
                        pool.begin() + window_start + train_bs);
                std::shuffle(pool.begin(), pool.begin() + train_bs, rng);
                window_start = (window_start + train_bs) % POOL_SIZE;

                for (size_t i = 0; i < train_bs; ++i) {
                    State S = global_player_advs[player][pool[i]].state;
                    update_tensors(
                        S, 
                        batched_hands, 
                        batched_flops, 
                        batched_turns,
                        batched_rivers,
                        batched_fracs,
                        batched_status,
                        i
                    ); 

                    for (size_t a = 0; a < NUM_ACTIONS; ++a) {
                        batched_advs_a[i][a] = global_player_advs[player][pool[i]].advantages[a];
                    }
                    int iteration = global_player_advs[player][pool[i]].iteration;
                    batched_iters_a[i][0] = static_cast<int>(iteration);
                }

                auto pred = train_net->forward(
                    batched_hands.slice(0, 0, train_bs).to(gpu_device),
                    batched_flops.slice(0, 0, train_bs).to(gpu_device),
                    batched_turns.slice(0, 0, train_bs).to(gpu_device),
                    batched_rivers.slice(0, 0, train_bs).to(gpu_device), 
                    batched_fracs.slice(0, 0, train_bs).to(gpu_device), 
                    batched_status.slice(0, 0, train_bs).to(gpu_device)
                );

                auto loss = torch::nn::functional::mse_loss(
                    pred, 
                    batched_advs.slice(0, 0, train_bs).to(gpu_device),
                    torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)
                );

                loss *= batched_iters.slice(0, 0, train_bs).to(torch::kFloat32).to(gpu_device) / static_cast<float>(CFR_ITERS);
                auto batch_mean_loss = loss.mean(1).mean();
                batch_mean_loss.backward();
                torch::nn::utils::clip_grad_norm_(train_net->parameters(), 1.0);
                optimizer.step();
                DEBUG_NONE("ITER: " << train_iter << "/" << TRAIN_ITERS << " LOSS: " << batch_mean_loss.to(cpu_device).item());            
            }

            std::string save_path = (current_path / std::to_string(cfr_iter) / std::to_string(player) / "model.pt").string();

            std::filesystem::create_directories(std::filesystem::path(save_path).parent_path());
            torch::save(train_net, save_path);
            DEBUG_NONE("successfully saved nets");
            DEBUG_WRITE(logfile, "successfully saved at: " << save_path);

            if (std::filesystem::exists(save_path)) {
                DEBUG_NONE("File successfully created at " << save_path);
            } else {
                DEBUG_NONE("File was not created at " << save_path);
                throw std::runtime_error("");
            }
        }
    }
}
