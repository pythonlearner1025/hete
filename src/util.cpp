#include "util.h"

using namespace mlx::core;

std::vector<float> get_bets(PokerEngine& engine) {
   std::vector<float> amounts;
   amounts.reserve(4 * NUM_PLAYERS * MAX_ROUND_BETS);
   double pot = engine.small_blind + engine.big_blind;
   
   for(int r = 0; r < 4; r++) {
       for(int p = 0; p < NUM_PLAYERS; p++) {
           for(int b = 0; b < MAX_ROUND_BETS; b++) {
               float val = 0.0f;
               if(engine.players[p].bets_per_round[r][b] >= 0 && pot > 0) {
                   val = engine.players[p].bets_per_round[r][b] / pot;
                   pot += engine.players[p].bets_per_round[r][b];
               }
               amounts.push_back(val);
           }
       }
   }
   return amounts;
}

float sample_uniform() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng); // Thread-safe random float between 0 and 1
}

float sample_uniform() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng); // Thread-safe random float between 0 and 1
}

int sample_iter(size_t max_iter) {
    if (max_iter == 0) {
        return 0;
    }
    
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, static_cast<int>(max_iter));
    
    try {
        return dist(rng);
    } catch (const std::exception& e) {
        std::cerr << "Error in sample_iter: " << e.what() << " max_iter: " << max_iter << std::endl;
        return 1;  // Return safe default
    }
}
// Sample an action according to the strategy probabilities
int sample_action(const std::array<double, NUM_ACTIONS>& strat) {
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng); // Thread-safe random number between 0 and 1
    double cumulative = 0.0;
    for (size_t i = 0; i < strat.size(); ++i) {
        cumulative += strat[i];
        if (r <= cumulative) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(strat.size() - 1);
}

void take_action(PokerEngine* engine, int player, int act) {
    DEBUG_INFO("Chosen act: " << act);
    if (act == 0) {
        engine->fold(player);
        return;
    }
    if (act == 1) {
        engine->check_or_call(player); 
        return;
    }

    if (act == NUM_ACTIONS-1) {
        engine->all_in(player);
        return;
    }

    double inc = engine->get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            engine->bet_or_raise(player, bet_amt);
            return;
        }
        bet_amt += inc;
    }
}

bool verify_action(PokerEngine* engine, int player, int act, std::string logfile = "") {
    if (act == 0) {
        return engine->can_fold(player);
    }
    if (act == 1) {
        return engine->can_check_or_call(player); 
    }
    double inc = engine->get_pot() * 1.0 / static_cast<double>(NUM_ACTIONS);
    double bet_amt = inc;
    for (int a = 2; a < NUM_ACTIONS; ++a) {
        if (a == act) {
            return engine->can_bet_or_raise(player, bet_amt, logfile);
        }
        bet_amt += inc;
    }
    return false;
}

void get_state(
    PokerEngine& game,
    State* state,
    int player
) {
    state->bets = get_bets(game); 
    std::vector<float> hands_vec;
    std::array<int, 2> hand = game.players[player].hand;
    for (int i = 0; i < 2; ++i) {
        hands_vec.push_back(game.players[player].hand[i]);
    }

    std::array<int, 5> board = game.get_board();
    for (int i = 0; i < 3; ++i) {
        hands_vec.push_back(board[i]);
    }

    hands_vec.push_back(board[3]);
    hands_vec.push_back(board[4]);
    state->hands = hands_vec;
}

std::array<double, NUM_ACTIONS> regret_match(const mlx::core::array logits) {
  auto relu_logits = maximum(logits, zeros_like(logits));
  auto logits_sum = sum(relu_logits);
  
  mlx::core::array probs = (logits_sum.item<float>() > 0) 
      ? relu_logits / logits_sum 
      : full(logits.shape(), static_cast<double>(1.0f/NUM_ACTIONS));

  std::array<double, NUM_ACTIONS> out{0};
  for(int i = 0; i < NUM_ACTIONS; i++) {
      //array idx = array({i}); 
      out[i] = static_cast<double>(take(probs, i).item<float>());
  }
  return out;
}

array regret_match_batched(const mlx::core::array logits) {
  auto relu_logits = maximum(logits, zeros_like(logits));
  auto logits_sum = sum(relu_logits);
  
  mlx::core::array probs = (logits_sum.item<float>() > 0) 
      ? relu_logits / logits_sum 
      : full(logits.shape(), static_cast<double>(1.0f/NUM_ACTIONS));

  return probs;
}
// Must return a probability distribution
std::array<double, NUM_ACTIONS> sample_prob(const torch::Tensor& logits, float beta) {
    double logits_sum = logits.sum().item<double>() + beta;
    std::array<double, NUM_ACTIONS> strat{};
    auto strategy_tensor = (logits+beta) / logits_sum;
    auto strat_data = strategy_tensor.data_ptr<float>();
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        strat[i] = strat_data[i];
    }
    return strat;
}

void save_model(std::map<std::string, std::optional<mlx::core::array>> params, const std::string& filepath) {
    std::unordered_map<std::string, array> save_map;
    
    for (const auto& [name, param_opt] : params) {
        if (param_opt.has_value()) {
            save_map.emplace(name, param_opt.value());
        }

    }
    save_safetensors(filepath, save_map);
}

std::map<std::string, std::optional<mlx::core::array>> load_model(const std::string& filepath) {
    auto [loaded_arrays, metadata] = load_safetensors(filepath);
    
    std::map<std::string, std::optional<array>> param_map;
    for (const auto& [name, arr] : loaded_arrays) {
        param_map[name] = std::optional<array>(arr);
    }
    
    return param_map;
}