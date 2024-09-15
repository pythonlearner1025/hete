// cfr.cpp

#include "cfr.h"
#include <torch/torch.h>
#include <tuple>
#include <thread>
#include <atomic>
#include <set>
#include <algorithm>
#include <cstdlib> // For rand()
#include <ctime>   // For seeding rand()

// for debugging
std::atomic<int> total_traversals(0);
std::atomic<int> total_advantages(0);

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

    //std::cout << "Player hand: " << game.players[player].hand[0] << ", " << game.players[player].hand[1] << std::endl;
    I.cards[0] = torch::tensor({static_cast<int64_t>(game.players[player].hand[0]), 
                                static_cast<int64_t>(game.players[player].hand[1])}).view({1, 2});

    //std::cout << "Board size: " << board_size << std::endl;
    if (board_size >= 3) {
        //std::cout << "Flop: " << board[0] << ", " << board[1] << ", " << board[2] << std::endl;
        I.cards[1] = torch::tensor({static_cast<int64_t>(board[0]), 
                                    static_cast<int64_t>(board[1]), 
                                    static_cast<int64_t>(board[2])}).view({1, 3});
    } else {
        I.cards[1] = torch::tensor({-1, -1, -1}).view({1, 3});
    }

    if (board_size >= 4) {
        //std::cout << "Turn: " << board[3] << std::endl;
        I.cards[2] = torch::tensor({static_cast<int64_t>(board[3])}).view({1, 1});
    } else {
        I.cards[2] = torch::tensor({-1}).view({1, 1});
    }

    if (board_size >= 5) {
        //std::cout << "River: " << board[4] << std::endl;
        I.cards[3] = torch::tensor({static_cast<int64_t>(board[4])}).view({1, 1});
    } else {
        I.cards[3] = torch::tensor({-1}).view({1, 1});
    }
}

// Prepare the information set for the neural network input
Infoset prepare_infoset(
    PokerEngine& game,
    int player,
    int max_bets_per_player
) {
    Infoset I;
    auto history = game.construct_history();
    // Convert history to tensors
    get_cards(game, player, I);
    // Assuming history.first is bet_status and history.second is bet_fracs
    std::vector<double> bet_fracs_vec(history.second.begin(), history.second.end());
    std::vector<float> bet_fracs_float(bet_fracs_vec.begin(), bet_fracs_vec.end());
    I.bet_fracs = torch::tensor(bet_fracs_float, torch::kFloat32).view({1, static_cast<long long>(bet_fracs_vec.size())});

    std::vector<int> bet_status_vec;
    for(auto status : history.first) {
        bet_status_vec.push_back(status ? 1 : 0);
    }
    std::vector<float> bet_status_float(bet_status_vec.begin(), bet_status_vec.end());
    I.bet_status = torch::tensor(bet_status_float, torch::kFloat32).view({1, static_cast<long long>(bet_status_float.size())});

    return I;
}

std::array<double, MAX_ACTIONS> regret_match(const torch::Tensor& logits, int n_acts) {
    auto relu_logits = torch::relu(logits);
    
    double logits_sum = relu_logits.sum().item<double>();
    
    std::array<double, MAX_ACTIONS> strat;
    
    // If the sum is positive, calculate the strategy
    if (logits_sum > 0) {
        auto strategy_tensor = relu_logits / (logits_sum - relu_logits);
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
    for (int i = 0; i < n_acts; ++i) {
        cumulative += strat[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return n_acts - 1; // Return last valid action if none selected
}

void take_action(PokerEngine& engine, int player, int act, int n_acts) {
    std::cout << "chosen act: " + std::to_string(act) << std::endl;
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
    double inc = 1.0 / static_cast<double>(n_acts);
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
    std::vector<std::vector<void*>>& nets,
    int t,
    std::vector<TraverseAdvantage>& traverse_advs,
    std::mutex& advs_mutex
) {
    std::cout << get_timestamp() << " Entering traverse()" << std::endl;
    std::cout << get_timestamp() << " Game status: " << engine.get_game_status() << ", Is player playing: " << engine.is_playing(player) << std::endl;

    total_traversals++;
    // Check if game is done for this player
    if (!engine.get_game_status() || !engine.is_playing(player)) {
        std::cout << get_timestamp() << " Game over" << std::endl;
        std::array<double, PokerEngine::MAX_PLAYERS> payoff = engine.get_payoffs();
        double bb = engine.get_big_blind();
        return payoff[player] / bb;
    }
    else if (engine.turn() == player) {
        std::cout << get_timestamp() << " My turn" << std::endl;

        void* net_ptr = nets[player][nets[player].size()-1];

        Infoset I = prepare_infoset(engine, player, PokerEngine::MAX_ROUND_BETS);

        torch::Tensor logits = deep_cfr_model_forward(net_ptr, I.cards, I.bet_fracs, I.bet_status);

        std::cout << get_timestamp() << " Neural network logits: " << logits << std::endl;

        int n_acts = get_action_head_dim(net_ptr);
        // Regret matching
        std::array<double, MAX_ACTIONS> strat = regret_match(logits, n_acts);

        // Define possible actions
        std::array<double, MAX_ACTIONS> values{};  // Initialize all elements to 0.0
        std::array<double, MAX_ACTIONS> advs{};    // Initialize all elements to 0.0
        std::array<bool, MAX_ACTIONS> is_illegal{};  // Initialize all elements to false
        double ev = 0.0;

        for (size_t a = 0; a < n_acts; ++a) {

            std::cout << "modeling action: " + std::to_string(a) << std::endl;
            // Verify action
            // ensure game state doesn't change
            if (!verify_action(engine, player, a, n_acts)) {
                is_illegal[a] = true;
                continue;
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
                traverse_advs,
                advs_mutex
            );

            values[a] = value;
            ev += value * strat[a];
        }

        std::cout << get_timestamp() << " Calculated advantages: ";
        for (size_t a = 0; a < n_acts; ++a) {
            if (!is_illegal[a]) {
                double adv = values[a] - ev;
                advs[a] = (adv > 0.0) ? adv : (1.0 / static_cast<double>(n_acts));
            } else {
                // TODO what should illegal act adv be? 
                advs[a] = 0.0;
            }
            std::cout << advs[a] << " ";
        }
        std::cout << std::endl;


        // Add to traverse_advs with synchronization
        TraverseAdvantage ta;
        ta.infoset = I;
        ta.iteration = t;
        ta.advantages = advs;

        {
            std::lock_guard<std::mutex> lock(advs_mutex);
            traverse_advs.emplace_back(ta);
        }

        // Return expected value
        std::cout << get_timestamp() << "ev: " + std::to_string(ev) << std::endl;
        return ev;
    }
    else {
        std::cout << get_timestamp() << " Opponent's turn" << std::endl;
        // Opponent's turn
        int actor = engine.turn();
        void* net_ptr = nets[actor][nets[actor].size()-1];
        int n_acts = get_action_head_dim(net_ptr);
        // Prepare infoset
        Infoset I = prepare_infoset(engine, actor, PokerEngine::MAX_ROUND_BETS);
        // Forward pass
        torch::Tensor logits = deep_cfr_model_forward(net_ptr, I.cards, I.bet_fracs, I.bet_status);
        // Regret matching
        std::array<double, MAX_ACTIONS> strat = regret_match(logits, n_acts);

        // Sample action according to strat
        int action_index = sample_action(strat, n_acts);

        // Verify and adjust action if necessary
        while (!verify_action(engine, player, action_index, n_acts)) {
            action_index = (action_index + 1) % n_acts;
        }
        // Take action
        take_action(engine, actor, action_index, n_acts);

        std::cout << get_timestamp() << " Opponent's turn. Selected action: " << action_index << std::endl;

        // Recursive call
        return traverse(
            engine,
            player,
            nets,
            t,
            traverse_advs,
            advs_mutex
        );
    }
}

// Function to run multiple traversals in parallel
int main() {
    int num_traversals = 10;
    int player = 0;
    int n_players = 2;
    double small_bet = 0.5;
    double big_bet = 1;
    std::vector<double> antes = {0.0, 0.0};
    void* init_model = create_deep_cfr_model(n_players, PokerEngine::MAX_ROUND_BETS, MAX_ACTIONS);
    std::vector<std::vector<void*>> nets = {{init_model}, {init_model}};
    int t = 0;
    std::vector<TraverseAdvantage> all_traverse_advs;
    std::mutex advs_mutex;
    // Determine the number of available hardware threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback to 4 if unable to detect
    num_threads = 1;

    std::cout << "total threads: " + std::to_string(num_threads) << std::endl;

    // Function for each thread to execute
    auto thread_func = [&](int traversals_per_thread, int thread_id) {
        std::cout << get_timestamp() << " Thread " << thread_id << " starting with " << traversals_per_thread << " traversals" << std::endl;
        int local_traversals = 0;
        int local_advantages = 0;

        for(int k = 0; k < traversals_per_thread; ++k) {
            // Extract parameters
            std::vector<double> starting_stacks = {100.0, 100.0};

            int starting_actor = 0;

            // Initialize PokerEngine
            PokerEngine engine(
                starting_stacks, 
                antes,
                starting_actor,
                n_players, 
                small_bet, 
                big_bet, 
                false
            ); // Assuming max_round_bets = 4 and manual_mode = true

            std::vector<TraverseAdvantage> local_traverse_advs;

            try {
                traverse(engine, player, nets, t, local_traverse_advs, advs_mutex);
                local_traversals++;
                local_advantages += local_traverse_advs.size();
            }
            catch(const std::exception& e) {
                // Handle exceptions (e.g., log and continue)
                std::cout << e.what() << std::endl;
                continue;
            }
            // Add local_traverse_advs to the shared all_traverse_advs with synchronization
            {
                std::lock_guard<std::mutex> lock(advs_mutex);
                std::cout << get_timestamp() << " Thread " << thread_id << " adding " << local_traverse_advs.size() << " advantages" << std::endl;
                all_traverse_advs.insert(all_traverse_advs.end(), local_traverse_advs.begin(), local_traverse_advs.end());
            }
        }
        total_advantages += local_advantages;
    };

    // Calculate traversals per thread
    int traversals_per_thread = num_traversals / num_threads;
    std::cout << "traversals per thread: " << std::to_string(traversals_per_thread) << std::endl;
    int remaining_traversals = num_traversals % num_threads;
    std::cout << "remaining traversals: " << std::to_string(remaining_traversals) << std::endl;

    // Create threads
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

    std::cout << "all jobs done" << std::endl;
    std::cout << get_timestamp() << " Total traversals: " << total_traversals.load() << std::endl;
    std::cout << get_timestamp() << " Total advantages collected: " << total_advantages.load() << std::endl;
    std::cout << get_timestamp() << " Size of all_traverse_advs: " << all_traverse_advs.size() << std::endl;

    return 0;
}
