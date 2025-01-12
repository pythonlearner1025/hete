#include "cfr.h"
#include "debug.h"
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
#include "model/model.h"

struct RandInit {
    RandInit() { std::srand(static_cast<unsigned int>(std::time(nullptr))); }
} rand_init;

std::mutex eval_mutex;
std::array<std::mutex, NUM_PLAYERS> player_advs_mutex;
std::array<std::vector<TraverseAdvantage>, NUM_PLAYERS> global_player_advs{};
std::array<std::atomic<size_t>, NUM_PLAYERS> total_advs{};
std::atomic<size_t> cfr_iter_advs(0);
constexpr double NULL_VALUE = -42.0;

void safe_add_advantage(int player, const TraverseAdvantage& adv, std::mt19937& rng) {
    std::lock_guard<std::mutex> lock(player_advs_mutex[player]);
    size_t current_total = total_advs[player].load();
    cfr_iter_advs.fetch_add(1);
    
    if (current_total < MAX_SIZE) {
        global_player_advs[player][current_total] = adv;
        total_advs[player].fetch_add(1);
    } else {
        std::uniform_int_distribution<size_t> dist(0, current_total - 1);
        size_t r = dist(rng);
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
    double sampling_prob;

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
        int unprocessed_children_ = 0,
        double sampling_prob_ = 0
    ) :
        values(values_),
        strat(strat_),
        is_illegal(is_illegal_),
        parent(parent_adv),
        parent_action(parent_action_),
        state(state_),
        depth(depth),
        unprocessed_children(unprocessed_children_),
        sampling_prob(sampling_prob_)
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

std::string format_array(const std::array<double, NUM_ACTIONS>& arr) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        ss << arr[i];
        if (i < arr.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}


double logarithmic_smoothing(double x_start, double x_end, int t, int max_steps) {
    // Prevent division by zero and log(0)
    t = std::max(1, std::min(t, max_steps));
    double decay = 1.0 - (std::log(static_cast<double>(t)) / std::log(static_cast<double>(max_steps)));
    double value = x_end + (x_start - x_end) * decay;
    return value;
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

double get_fold_exploration(double epsilon, int cfr_iter, bool is_preflop) {
    // Higher epsilon for folds preflop to encourage more calls/raises
    double base_gain = is_preflop ? 2 : 1.5;  // Increased preflop epsilon
    return base_gain*epsilon;
}

void outcome_sampling(
    int thread_id,
    int player, 
    std::filesystem::path model_path, 
    int cfr_iter, 
    int traversals,
    std::array<double, NUM_PLAYERS> starting_stacks, 
    std::array<double, NUM_PLAYERS> antes,
    int starting_actor,
    double small_bet, 
    double big_bet,
    std::string logfile
) {
    mlx::core::set_default_device(Device::gpu);

    // Initialize RNG
    thread_local std::mt19937 rng(std::random_device{}());
    std::map<std::tuple<int, int>, std::shared_ptr<PokerGPT>> cache;


    // For each traversal
    for (int t = 0; t < traversals; ++t) {
        DEBUG_NONE("cfr_iter="<<cfr_iter<<" thread="<<thread_id<<" traversal= "<<t<<"/"<<traversals<<" cfr_iter_advs="<<format_scientific(cfr_iter_advs.load()));
        // for now, just read from path
        std::array<std::shared_ptr<PokerGPT>, NUM_PLAYERS> models;

        DEBUG_NONE("cfr_iter="<<cfr_iter<<" thread="<<thread_id<<" traversal= "<<t<<"/"<<traversals<<" cfr_iter_advs="<<format_scientific(cfr_iter_advs.load()));
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (cfr_iter>1) {
                int sampled_iter = sample_iter(static_cast<size_t>(cfr_iter-1));
                auto key = std::tuple(p, sampled_iter);
                if (cache.find(key) == cache.end()) {
                    // make shared ptr first, then load into it
                    cache[key] = std::make_shared<PokerGPT>();
                    auto path = model_path / std::to_string(sampled_iter) / std::to_string(p) / "model.safetensors";
                    auto params = load_model(path);
                    cache[key]->update(params);
                    models[p] = cache[key];
                    models[p]->to_bfloat16();
                } else {
                    models[p] = cache[key];
                }
            } else {
                models[p] = std::make_shared<PokerGPT>();
                models[p]->to_bfloat16();
            }
        }

        // Initialize the game
        PokerEngine engine(
            starting_stacks,
            antes,
            starting_actor,
            small_bet,
            big_bet,
            false // manual mode
        );

        // Recursive CFR function
        std::function<double(PokerEngine&, int, int, double, double, double, int)> cfr;
        cfr = [&](PokerEngine& engine, int traversing_player, int current_player, double player_reach, double opp_reach, double total_sampling_prob, int depth) -> double {
            // Base case: if terminal node
            if (!engine.get_game_status()) {
                // Game is over
                auto payoffs = engine.get_payoffs();
                double u_i = payoffs[traversing_player];
                return u_i;
            }

            // If current player is the traversing player
            if (current_player == traversing_player) {
                try {
                    // Get the information set (state)
                    State S;
                    get_state(engine, &S, traversing_player);
                    DEBUG_WRITE(logfile, "PLAYER TURN\n");
                    DEBUG_WRITE(logfile, "hands: \n" << S.hands);
                    DEBUG_WRITE(logfile, "bets: \n" << S.bets);

                    auto& gpt = *models[traversing_player];  
                    array hands = array(S.hands.data(), {1, static_cast<int>(S.hands.size())}, float32);
                    array bets = array(S.bets.data(), {1, static_cast<int>( S.bets.size())}, float32);

                    array preds = gpt.forward(hands, bets); 
                    //DEBUG_NONE(preds);
                    //eval(preds);

                    std::array<double, NUM_ACTIONS> strategy = regret_match(preds);

                    DEBUG_WRITE(logfile, "strat=" << format_array(strategy));
                    // first check if we have any legal actions with non-zero strategy
                    bool all_legal_actions_zero = true;
                    int num_legal_actions = 0;
                    for (int a = 0; a < NUM_ACTIONS; ++a) {
                        if (verify_action(&engine, current_player, a, logfile)) {
                            num_legal_actions++;
                            if (strategy[a] > 0) {
                                all_legal_actions_zero = false;
                                break;
                            }
                        }
                    }

                    auto epsilon = std::max(EPSILON, 1.0 / std::sqrt(cfr_iter));  // Decay exploration over time

                    std::array<double, NUM_ACTIONS> sampling_strategy{0.0};
                    if (all_legal_actions_zero && num_legal_actions > 0) {
                        // corner case: uniform over legal actions
                        for (int a = 0; a < NUM_ACTIONS; ++a) {
                            if (verify_action(&engine, current_player, a, logfile)) {
                                sampling_strategy[a] = 1.0 / num_legal_actions;
                                strategy[a] = sampling_strategy[a];  // keep consistent for regret calc
                            }
                        }
                    } else if (num_legal_actions > 0) {
                        // normal case: epsilon exploration over non-zero strat actions
                        for (int a = 0; a < NUM_ACTIONS; ++a) {
                            if (a == 0) {
                                sampling_strategy[a] = strategy[a] * (1.0 - get_fold_exploration(epsilon, cfr_iter, engine.get_round() == 0));
                            } else {
                                sampling_strategy[a] = (1.0 - epsilon) * strategy[a] + epsilon / NUM_ACTIONS;
                            }
                            if (strategy[a] == 0) sampling_strategy[a] = 0;
                        }
                        sampling_strategy = normalize_to_prob_dist(sampling_strategy);
                    } else {
                        DEBUG_NONE("EDGE CASE: no legal actions that aren't zero");
                    }

                    int action = sample_action(sampling_strategy);

                    while (!verify_action(&engine, current_player, action, logfile)) {
                        DEBUG_WRITE(logfile, "invalid action=" << action << "trying next " << (action - 1 + NUM_ACTIONS) % NUM_ACTIONS);
                        action = (action - 1 + NUM_ACTIONS) % NUM_ACTIONS;
                    }

                    DEBUG_WRITE(logfile, "action=" << action);

                    DEBUG_WRITE(logfile, "pi_sigma_prime=" << total_sampling_prob);

                    // Compute the sampling probability for the action
                    double sampling_prob = sampling_strategy[action];
                    if (sampling_prob == 0)  {
                        DEBUG_NONE("this shouldn't happen");
                        sampling_prob = 0.01;
                    }
                    DEBUG_WRITE(logfile, "sigma_prime=" << sampling_prob);
                    DEBUG_WRITE(logfile, "pi_sigma_i=" << player_reach);

                    // update the sampling probability
                    double new_total_sampling_prob = total_sampling_prob * sampling_prob;

                    // Compute traversing player's reach probability under sigma
                    double sigma_i = strategy[action];
                    if (sigma_i == 0) {
                        DEBUG_NONE("this shouldn't happen");
                        sigma_i = 0.01;
                    }
                    //double sigma_i = strategy[action];
                    DEBUG_WRITE(logfile, "sigma_i=" << sigma_i);
                    double new_player_reach = player_reach * sigma_i;
                    DEBUG_WRITE(logfile, "new_pi_sigma_prime=" << new_total_sampling_prob);

                    // Take the action in the engine
                    take_action(&engine, current_player, action);
                    //DEBUG_NONE("next player");

                    // Get next player
                    int next_player = engine.turn();

                    double u = cfr(engine, traversing_player, next_player, new_player_reach, opp_reach, new_total_sampling_prob, depth + 1);
                    DEBUG_WRITE(logfile, "u=" << u);

                    double w_I = u * (player_reach / total_sampling_prob);

                    // Compute the regrets for each action
                    auto cum_regrets = 0.0;
                    std::array<double, NUM_ACTIONS> regrets;
                    for (int a = 0; a < NUM_ACTIONS; ++a) {
                        if (a == action) {
                            regrets[a] = w_I * (1.0 - strategy[a]);
                        } else {
                            regrets[a] = -w_I * strategy[a];
                        }
                        cum_regrets += std::abs(regrets[a]);
                    }
                    DEBUG_WRITE(logfile, "regrets=" << format_array(regrets));

                    if (cum_regrets >= 1e-3) {
                        // Store the regrets (advantages) in the buffer
                        TraverseAdvantage adv;
                        adv.state = S;
                        adv.regrets = regrets;
                        adv.iteration = cfr_iter;
                        if (t % 100 == 0) {
                            DEBUG_WRITE(logfile, "advantages="<<format_array(regrets));
                        }
                        safe_add_advantage(traversing_player, adv, rng);
                    }
                    return u;
                } catch (...) {
                    DEBUG_NONE("unknown error in forward pass");
                    throw;
                }
            } else {
                try {
                    // Opponent's turn
                    // Get the model for opponent
                    auto& gpt = *models[current_player];  

                    State S;
                    get_state(engine, &S, current_player);

                    array hands = array(S.hands.data(), {1, static_cast<int>(S.hands.size())}, float32);
                    array bets = array(S.bets.data(), {1, static_cast<int>(S.bets.size())}, float32);

                    array preds = gpt.forward(hands, bets); 
                    //eval(preds);
                    //eval(preds);

                    std::array<double, NUM_ACTIONS> strategy = regret_match(preds);

                    int action = sample_action(strategy);

                    while (!verify_action(&engine, current_player, action, logfile)) {
                        DEBUG_WRITE(logfile, "invalid action=" << action << "trying next " << (action - 1 + NUM_ACTIONS) % NUM_ACTIONS);
                        action = (action - 1 + NUM_ACTIONS) % NUM_ACTIONS;
                    }

                    double sigma = strategy[action];
                    if (sigma == 0) sigma = 0.1;

                    double new_opp_reach = opp_reach * sigma;

                    take_action(&engine, current_player, action);

                    int next_player = engine.turn();

                    // Recurse
                    double u = cfr(engine, traversing_player, next_player, player_reach, new_opp_reach, new_opp_reach, depth + 1);

                    return u;
                } catch (const std::exception& e) {
                    DEBUG_NONE("error in forward pass: " << e.what());
                    throw;
                } catch (...) {
                    DEBUG_NONE("unknown error in forward pass");
                    throw;
                }
            }
        };

        // Start traversal
        double u = cfr(engine, player, engine.turn(), 1.0, 1.0, 1.0, 0);
    }
}

template<typename F>
auto time_section(
    const std::string& name, 
    std::unordered_map<std::string, double>& timers, 
    std::unordered_map<std::string, size_t>& counters, 
    F&& func
    ) -> decltype(func()) {
    auto start = std::chrono::steady_clock::now();
    if constexpr (std::is_void_v<decltype(func())>) {
        func();
        auto end = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        DEBUG_NONE(name << " took " << us << " us");
        timers[name] += us;
        counters[name]++;
    } else {
        auto result = func();
        auto end = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        DEBUG_NONE(name << " took " << us << " us");
        timers[name] += us;
        counters[name]++;
        return result;
    }
}


int test() {
    test_mlx();
    return 0;
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

    {
        PokerGPT gpt;
        size_t n_params = 0; 
        for (auto& [name, param_opt] : gpt.parameters()) {
            DEBUG_NONE("name: " << name);
            DEBUG_NONE("n_params: " <<param_opt.value().size());
            if (param_opt.has_value()) {
                n_params += param_opt.value().size();
            }
        }
        DEBUG_NONE("n params: " << n_params);
        DEBUG_WRITE(logfile, "n params: " << n_params);
    }
    
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

    auto thread_func = [&](int traversals_per_thread, int thread_id, int player, int cfr_iter) {
        try {
            outcome_sampling(
                thread_id,
                player, 
                current_path, 
                cfr_iter, 
                traversals_per_thread,
                starting_stacks, 
                antes,
                starting_actor,
                small_bet, 
                big_bet,
                logfile
            );
        }
        catch(const std::exception& e) {
            DEBUG_INFO(e.what());
        }
    };


    for (int cfr_iter=1; cfr_iter<CFR_ITERS+1; ++cfr_iter) {

        for (int player=0; player<NUM_PLAYERS; ++player) {
            // init threads
            auto start = std::chrono::steady_clock::now();
            std::vector<std::thread> threads;
            for (unsigned int thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
                int traversals_to_run = traversals_per_thread + (thread_id < remaining_traversals ? 1 : 0);
                
                DEBUG_NONE("spawning thread");
                // Capture thread_nets by reference
                threads.emplace_back(
                    [&thread_func, traversals_to_run, thread_id, player, cfr_iter]() {
                        thread_func(traversals_to_run, thread_id, player, cfr_iter);
                    }
                );
            } 

            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }

            auto end = std::chrono::steady_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
            DEBUG_NONE("took " << us << "s");
            DEBUG_NONE("CFR ITER = " << cfr_iter);
            DEBUG_WRITE(logfile, "CFR ITER = " << cfr_iter);
            DEBUG_NONE("COLLECTED ADVS = " << cfr_iter_advs.load());
            DEBUG_WRITE(logfile, "COLLECTED ADVS = " << total_advs[player].load());
            DEBUG_NONE("PLAYER = " << player);
            std::random_device rd;
            std::mt19937 rng(rd());
            size_t train_bs = std::min(TRAIN_BS, cfr_iter_advs.load());
            if (train_bs == 0) continue;
            cfr_iter_advs.store(0, std::memory_order_seq_cst);
            size_t total_advs_player = total_advs[player].load();
            if (total_advs_player == 0) continue;

            // Build weights vector
            std::vector<double> weights(total_advs_player);
            for (size_t i = 0; i < total_advs_player; ++i) {
                weights[i] = static_cast<double>(global_player_advs[player][i].iteration);
            }

            // Create discrete distribution for weighted sampling
            std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
            size_t train_iters = std::min(TRAIN_ITERS, static_cast<size_t>(total_advs_player / TRAIN_BS * TRAIN_EPOCHS));

            mlx::core::set_default_device(Device::gpu);

            PokerGPT gpt;
            AdamOptimizer opt(0.001f);
            opt.init(gpt.parameters()); //fresh params
            gpt.to_bfloat16();

            DEBUG_NONE("TRAIN_BS = " << train_bs);
            DEBUG_NONE("TRAIN_ITERS = " << train_iters);
            DEBUG_WRITE(logfile, "TRAIN_ITERS = " << train_iters);
            long total_time = 0; 

            auto loss_fn = [&](const std::vector<array>& arg_arrays) {
                // construct parameter map from vector
                auto param_map = gpt.parameters();
                int i = 3;
                for (auto& [name, param_opt] : param_map) {
                    if (param_opt.has_value()) {
                        param_opt = arg_arrays[i++];
                    }
                }

                gpt.update(param_map);
                
                array preds = gpt.forward(arg_arrays[0], arg_arrays[1]);
                return smooth_l1_loss(squeeze(preds), arg_arrays[2]);
            };

            for (size_t train_iter = 0; train_iter < train_iters; ++train_iter) {
                std::vector<array> hands_vec;
                std::vector<array> bets_vec;
                std::vector<std::vector<double>> target_regrets;
                for (size_t i = 0; i < train_bs; ++i) {
                    size_t idx = dist(rng);

                    State S = global_player_advs[player][idx].state;
                    array hands = array(S.hands.data(), {static_cast<int>(S.hands.size())}, float32);
                    array bets = array(S.bets.data(), {static_cast<int>(S.bets.size())}, float32);
                    hands_vec.push_back(hands);
                    bets_vec.push_back(bets);

                    std::vector<double> regrets; 
                    for  (size_t a = 0; a < NUM_ACTIONS; ++a) {
                        regrets.push_back(global_player_advs[player][idx].regrets[a]);
                    }
                    target_regrets.push_back(regrets);
                }

                array batched_hands = reshape(concatenate(hands_vec, 0), {TRAIN_BS, -1});
                array batched_bets = reshape(concatenate(bets_vec, 0), {TRAIN_BS, -1});
                std::vector<double> flattened_regrets;
                flattened_regrets.reserve(target_regrets.size() * NUM_ACTIONS);
                for (const auto& regret_vec : target_regrets) {
                    flattened_regrets.insert(flattened_regrets.end(), regret_vec.begin(), regret_vec.end());
                }
                array batched_regrets = array(flattened_regrets.data(), 
                                            {static_cast<int>(target_regrets.size()), 
                                                static_cast<int>(NUM_ACTIONS)}, 
                                            float32);
                auto start = std::chrono::steady_clock::now();
                
                std::vector<array> param_arrays = {batched_hands, batched_bets, batched_regrets};
                std::vector<int> argnums;
                int param_idx = 3;
                for (const auto& [name, param_opt] : gpt.parameters()) {
                    if (param_opt.has_value()) {
                        param_arrays.push_back(param_opt.value());
                        argnums.push_back(param_idx);
                        param_idx++;
                    }
                }

                auto grad_fn = mlx::core::value_and_grad(loss_fn, argnums);

                auto [losses, grads] = grad_fn(param_arrays); 

                std::map<std::string, std::optional<array>> grad_map;
                int i = 0;
                for (const auto& [name, param_opt] : gpt.parameters()) {
                    if (param_opt.has_value()) {
                        grad_map[name] = grads[i++];
                    }
                }

                opt.update(gpt, grad_map);

                eval(losses);

                auto end = std::chrono::steady_clock::now();
                auto step_tm = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                total_time += static_cast<long>(step_tm);
                if (train_iter % 10 == 0) {
                    DEBUG_NONE("ITER: " << train_iter << "/" << train_iters << " took " << total_time / train_iter  << "ms" << " LOSS: " << losses.item<float>());            
                }
            }

            std::string save_path = (current_path / std::to_string(cfr_iter) / std::to_string(player) / "model.safetensors").string();
            std::filesystem::create_directories(std::filesystem::path(save_path).parent_path());
            save_model(gpt.parameters(), save_path);
            DEBUG_NONE("successfully saved nets");
            DEBUG_WRITE(logfile, "successfully saved at: " << save_path);

            if (std::filesystem::exists(save_path)) {
                DEBUG_NONE("File successfully created at " << save_path);
            } else {
                DEBUG_NONE("File was not created at " << save_path);
                throw std::runtime_error("file was not created!");
            }
        }
    }
}
