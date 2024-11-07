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

std::string format_game_state(const State& state) {
    std::stringstream ss;
    ss << "Hand: [" << state.hand[0] << "," << state.hand[1] << "] "
       << "Board: [" << state.flop[0] << "," << state.flop[1] << "," 
       << state.flop[2] << "," << state.turn[0] << "," << state.river[0] << "]";
    return ss.str();
}

std::string format_tensor(const torch::Tensor& tensor) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);
    
    // Handle 0-dim tensor
    if (tensor.dim() == 0) {
        ss << tensor.item<float>();
        return ss.str();
    }
    
    ss << "[";
    
    // Handle 1-dim tensor
    if (tensor.dim() == 1) {
        for (int64_t i = 0; i < tensor.size(0); i++) {
            ss << tensor[i].item<float>();
            if (i < tensor.size(0) - 1) ss << ", ";
        }
    }
    // Handle 2-dim tensor 
    else if (tensor.dim() == 2) {
        for (int64_t i = 0; i < tensor.size(0); i++) {
            ss << "[";
            for (int64_t j = 0; j < tensor.size(1); j++) {
                ss << tensor[i][j].item<float>();
                if (j < tensor.size(1) - 1) ss << ", ";
            }
            ss << "]";
            if (i < tensor.size(0) - 1) ss << ", ";
        }
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
    // Initialize RNG
    std::mt19937 rng(std::random_device{}());
    torch::Tensor hand = init_batched_hands(1).to(gpu_device);
    torch::Tensor flop = init_batched_flops(1).to(gpu_device);
    torch::Tensor turn = init_batched_turns(1).to(gpu_device);
    torch::Tensor river = init_batched_rivers(1).to(gpu_device);
    torch::Tensor bet_fracs = init_batched_fracs(1).to(gpu_device);
    torch::Tensor bet_status = init_batched_status(1).to(gpu_device);

    std::map<std::tuple<int, int>, DeepCFRModel> cache;
    // For each traversal
    for (int t = 0; t < traversals; ++t) {
        if (t % 1 == 0) {
            DEBUG_NONE("cfr_iter="<<cfr_iter<<" thread="<<thread_id<<" traversal= "<<t<<"/"<<traversals<<" cfr_iter_advs="<<format_scientific(cfr_iter_advs.load()));
        }
        // Initialize models for each player
        std::array<DeepCFRModel, NUM_PLAYERS> models;
        int sampled_iter = sample_iter(static_cast<size_t>(cfr_iter-1));
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (cfr_iter>1) {
                auto key = std::tuple(p, sampled_iter);
                if (cache.find(key) == cache.end()) {
                    // Load the model from the path
                    auto path = model_path / std::to_string(sampled_iter) / std::to_string(p) / "model.pt";
                    models[p] = DeepCFRModel();
                    torch::load(models[p], path);
                    models[p]->to(gpu_device);
                    models[p]->eval();
                } else {
                    models[p] = cache[key]; 
                }
            } else{
                models[p]->to(gpu_device);
                models[p]->eval();
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
        cfr = [&](PokerEngine& engine, int traversing_player, int current_player, double pi_sigma_i, double pi_sigma_minus_i, double pi_sigma_prime, int depth) -> double {
            // Base case: if terminal node
            if (!engine.get_game_status()) {
                // Game is over
                auto payoffs = engine.get_payoffs();
                double u_i = payoffs[traversing_player];
                return u_i;
            }

            // If current player is the traversing player
            if (current_player == traversing_player) {
                // Get the information set (state)
                State state;
                get_state(engine, &state, traversing_player);
                if (t % 100 == 0) {
                    DEBUG_WRITE(logfile, "PLAYER TURN\n" << format_game_state(state));
                }

                // Get the model for traversing player
                auto model = models[traversing_player];

                // Prepare tensors for input
                update_tensors(state, hand, flop, turn, river, bet_fracs, bet_status, 0);

                // Get the model output
                auto logits = model->forward(hand, flop, turn, river, bet_fracs, bet_status);
                if (t % 100 == 0) {
                    DEBUG_WRITE(logfile, "logits=" << format_tensor(logits));
                }

                // Convert logits to strategy using regret matching
                std::array<double, NUM_ACTIONS> strategy = regret_match(logits[0]);

                if (t % 100 == 0) {
                    DEBUG_WRITE(logfile,"strategy="<<format_array(strategy));
                }

                // Ensure minimum exploration probability epsilon
                std::array<double, NUM_ACTIONS> sampling_strategy;
                for (int a = 0; a < NUM_ACTIONS; ++a) {
                    sampling_strategy[a] = (1.0 - EPSILON) * strategy[a] + EPSILON / NUM_ACTIONS;
                }

                // Normalize sampling strategy
                sampling_strategy = normalize_to_prob_dist(sampling_strategy);

                // Sample an action according to the sampling strategy
                int action = sample_action(sampling_strategy);

                // Compute the sampling probability for the action
                double sigma_prime = sampling_strategy[action];

                // Update the sampling probability
                double new_pi_sigma_prime = pi_sigma_prime * sigma_prime;

                // Compute traversing player's reach probability under sigma
                double sigma_i = strategy[action];
                double new_pi_sigma_i = pi_sigma_i * sigma_i;

                // Take the action in the engine
                take_action(&engine, current_player, action);

                // Get next player
                int next_player = engine.turn();

                // Recurse
                double u = cfr(engine, traversing_player, next_player, new_pi_sigma_i, pi_sigma_minus_i, new_pi_sigma_prime, depth + 1);

                // Compute the weighting factor w_I
                double w_I = u * (pi_sigma_i / pi_sigma_prime);

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

                if (cum_regrets >= 1e-3) {
                    // Store the regrets (advantages) in the buffer
                    TraverseAdvantage adv;
                    adv.state = state;
                    adv.advantages = regrets;
                    adv.iteration = cfr_iter;
                    if (t % 100 == 0) {
                        DEBUG_WRITE(logfile, "advantages="<<format_array(regrets));
                    }
                    safe_add_advantage(traversing_player, adv, rng);
                }
                return u;
            } else {
                // Opponent's turn
                // Get the model for opponent
                auto model = models[current_player];

                // Get the information set (state)
                State state;
                get_state(engine, &state, current_player);

                update_tensors(state, hand, flop, turn, river, bet_fracs, bet_status, 0);

                // Get the model output
                auto logits = model->forward(hand, flop, turn, river, bet_fracs, bet_status);

                // Convert logits to strategy using regret matching
                std::array<double, NUM_ACTIONS> strategy = regret_match(logits[0]);

                // Sample an action according to the strategy
                int action = sample_action(strategy);

                // Compute sigma(a|I)
                double sigma = strategy[action];

                // Update the reach probabilities
                double new_pi_sigma_minus_i = pi_sigma_minus_i * sigma;

                // Take the action in the engine
                take_action(&engine, current_player, action);

                // Get next player
                int next_player = engine.turn();

                // Recurse
                double u = cfr(engine, traversing_player, next_player, pi_sigma_i, new_pi_sigma_minus_i, pi_sigma_prime, depth + 1);

                return u;
            }
        };

        // Start traversal
        double u = cfr(engine, player, engine.turn(), 1.0, 1.0, 1.0, 0);
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

        for (int player=0; player<NUM_PLAYERS; ++player) {
            // init threads
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

            torch::optim::Adam optimizer(train_net->parameters(), torch::optim::AdamOptions(0.001));

            size_t train_iters = std::min(TRAIN_ITERS, static_cast<size_t>(total_advs_player / TRAIN_BS * TRAIN_EPOCHS));

            DEBUG_NONE("TRAIN_BS = " << train_bs);
            DEBUG_NONE("TRAIN_ITERS = " << train_iters);
            DEBUG_WRITE(logfile, "TRAIN_ITERS = " << train_iters);

            for (size_t train_iter = 0; train_iter < train_iters; ++train_iter) {
                // Rotate and shuffle window
                for (size_t i = 0; i < train_bs; ++i) {
                    size_t idx = dist(rng);

                    State S = global_player_advs[player][idx].state;
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
                        batched_advs_a[i][a] = global_player_advs[player][idx].advantages[a];
                    }
                    int iteration = global_player_advs[player][idx].iteration;
                    batched_iters_a[i][0] = static_cast<int>(iteration);
                }

                auto batch_advantages = batched_advs.slice(0, 0, train_bs).to(gpu_device);

                //auto transformed_advantages = batch_advantages.sign() * torch::log1p(batch_advantages.abs());

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
                    batch_advantages,
                    torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean)
                );

                loss.backward();
                torch::nn::utils::clip_grad_norm_(train_net->parameters(), 1.0);
                optimizer.step();
                optimizer.zero_grad();
                if (train_iter % 100 == 0) {
                    DEBUG_NONE("ITER: " << train_iter << "/" << train_iters << " LOSS: " << loss.to(cpu_device).item());            
                }
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
