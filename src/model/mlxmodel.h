// Copyright © 2023 Apple Inc.

#include <chrono>
#include <cmath>
#include <iostream>
#include "../debug.h"
#include <mlx/mlx.h>

/**
 * An example of linear regression with MLX.
 */

using namespace mlx::core;

std::string str_array(const std::array<double, NUM_ACTIONS>& arr) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        ss << arr[i];
        if (i < arr.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

using namespace mlx::core;


// TODO maybe change to std::shared_ptr for safety 
class MyModule {
protected:
   struct RegisteredArray {
       array* arr;
       std::string name;
   };

   struct RegisteredModule {
       MyModule* ptr; 
       std::string name;
   };

   MyModule* parent = nullptr;
   std::vector<RegisteredArray> arrays;
   std::vector<RegisteredModule> modules;
   std::string name;

   void register_array(array& arr, const std::string& name) {
        arrays.push_back({&arr, name});  // take ownership
   }

    void register_module(MyModule& module, const std::string& name) {
       if (name.empty()) {
           throw std::runtime_error("Cannot register module with empty name");
       }
       module.name = name;  // <-- SET THE NAME HERE 
       module.parent = this;
       modules.push_back({&module, name}); 
   }

public:
   MyModule(const std::string& name_) : name(name_) {}

   std::string get_full_path() const {
       if (!parent) {
        return name;
       }
       auto path = parent->get_full_path() + "." + name;
       return path;
   }

    std::map<std::string, std::optional<array>> parameters() {
        std::map<std::string, std::optional<array>> params;
        
        visit([&params](const std::string& path, array& arr) {
            params[path] = std::optional<array>(arr);
        });

        return params;
    }

    void visit(const std::function<void(const std::string&, array&)>& visitor, 
            const std::string& parent_path = "") {
        std::string base_path = parent_path.empty() ? name 
                                              : parent_path + "." + name;

        // visit arrays
        for (auto& reg : arrays) {
            visitor(base_path + "." + reg.name, *reg.arr);
        }

        // visit nested modules 
        for (auto& reg : modules) {
            reg.ptr->visit(visitor, base_path);
        }
    }

   void update(const std::map<std::string, std::optional<array>>& params) {
       std::string base_path = get_full_path();
       
       // update arrays
       for (auto& reg : arrays) {
           std::string full_path = base_path + "." + reg.name;
           if (params.count(full_path)) {
               *reg.arr = params.at(full_path).value();
           }
       }
       // update nested modules
       for (auto& reg : modules) {
           reg.ptr->update(params);
       }
   }
};

class AdamOptimizer {
private:
    float lr;
    float beta1;
    float beta2; 
    float eps;
    int step_count = 0;
    
    struct State {
        std::optional<array> m; // first moment 
        std::optional<array> v; // second moment
        
        // now we can have a default constructor
        State() = default;
        
        // and our normal constructor 
        State(const array& m_, const array& v_) : m(m_), v(v_) {}
    };
    // optimizer state storage
    std::map<std::string, State> state;

public:
    AdamOptimizer(
        float learning_rate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f
    ) : lr(learning_rate), beta1(beta1), beta2(beta2), eps(eps) {}

    // initialize optimizer state for parameters
    void init(const std::map<std::string, std::optional<array>>& params) {
        for (const auto& [name, param_opt] : params) {
            if (!param_opt.has_value()) continue;
            
            const array& param = param_opt.value();
            state[name] = State{
                zeros_like(param), // m
                zeros_like(param)  // v
            };
        }
    }

   // update parameters using gradients
    void update(MyModule& model, const std::map<std::string, std::optional<array>>& grads, float max_norm = -1.0f) {
        step_count++;
        
        // compute bias correction terms
        float bc1 = 1.0f - std::pow(beta1, step_count);
        float bc2 = 1.0f - std::pow(beta2, step_count);
        float lr_t = lr * std::sqrt(bc2) / bc1;

        // get current parameters
        auto params = model.parameters();

        // updated parameters to send back to model
        std::map<std::string, std::optional<array>> updated_params;

        for (const auto& [name, param_opt] : params) {
            if (!param_opt.has_value() || grads.count(name) == 0) {
                updated_params[name] = param_opt;
                continue;
            }

            const array& param = param_opt.value();
            const array& grad = grads.at(name).value();
            auto& s = state[name];

            // clip gradients if needed
            array clipped_grad = grad;
            if (max_norm > 0) {
                array grad_norm = sqrt(sum(square(grad)));
                array scale = minimum(array(max_norm), grad_norm) / grad_norm;
                clipped_grad = multiply(grad, scale);
            }
            
            // update momentum
            s.m = add(
                multiply(array(beta1), s.m.value()),
                multiply(array(1.0f - beta1), clipped_grad)
            );
            
            // update velocity 
            s.v = add(
                multiply(array(beta2), s.v.value()),
                multiply(array(1.0f - beta2), square(clipped_grad))
            );
            
            // compute update
            array update = multiply(
                array(lr_t),
                divide(
                    s.m.value(),
                    add(sqrt(s.v.value()), array(eps))
                )
            );

            // store updated parameter
            updated_params[name] = subtract(param, update);
        }

        // update model parameters
        model.update(updated_params);
    }

    // basic scheduler that reduces learning rate by factor after n steps
    void schedule_lr(float factor, int after_steps) {
        if (step_count > after_steps) {
            lr *= factor;
        }
    }
};

constexpr size_t N_HEADS = 4;
constexpr size_t N_LAYERS = 3;
constexpr int HEAD_DIM = MODEL_DIM / N_HEADS; 

class HeadLayerNorm : public MyModule {
  public:
    double eps = 1e-5; 
    array gamma = ones({HEAD_DIM});
    array beta = zeros({HEAD_DIM});

    HeadLayerNorm() : MyModule("LayerNorm") {
        register_array(gamma, "gamma");
        register_array(beta, "beta");
        DEBUG_NONE("layernorm reg complete");
    }

    array forward(array x) {
      array u = mean(x, -1, true);
      array v = var(x, -1, true);
      return gamma * (x - u) / mlx::core::sqrt(v + eps) + beta;
    }
};

class Decoder : public MyModule  {
   public:
     int attn_dim = MODEL_DIM / NUM_HEADS;
     array attn_wq = random::normal({MODEL_DIM, HEAD_DIM});
     array attn_wk = random::normal({MODEL_DIM, HEAD_DIM});
     array attn_wv = random::normal({MODEL_DIM, HEAD_DIM});
     array attn_out = random::normal({MODEL_DIM, HEAD_DIM});
     array ffn_1 = random::normal({HEAD_DIM, 4*MODEL_DIM});
     array ffn_2 = random::normal({4*MODEL_DIM, HEAD_DIM});
     array layer_norm = random::normal({HEAD_DIM});
     HeadLayerNorm norm;

     Decoder() : MyModule("Decoder") {
       register_array(attn_wq, "attn_wq");
       register_array(attn_wk, "attn_wk"); 
       register_array(attn_wv, "attn_wv");
       register_array(attn_out, "attn_out");
       register_array(ffn_1, "ffn_1");
       register_array(ffn_2, "ffn_2");
       register_array(layer_norm, "layer_norm");
       register_module(norm, "norm");
       DEBUG_NONE("decoder reg complete");
     }

   array forward(array x) {
      array q = matmul(x, attn_wq);
      array k = matmul(x, attn_wk);
      array v = matmul(x, attn_wv);
      
      array attn = reshape(matmul(q, reshape(k, {k.shape()[1], k.shape()[2], -1})) * sqrt(HEAD_DIM), {1, x.shape()[1], x.shape()[1]});
      attn = tril(attn);
      x = softmax(attn, -1);
      x = matmul(x, v);
      x = matmul(x, ffn_1);
      x = maximum(x, zeros_like(x));
      x = matmul(x, ffn_2);
      x = norm.forward(x);
      return x;
   }
};

class MHA : public MyModule {
    public: 
    std::array<Decoder, N_HEADS> heads;

    MHA() : MyModule("MHA"), heads{} {
        for(size_t i = 0; i < N_HEADS; i++) {
            register_module(heads[i], "head_" + std::to_string(i));
        }
    }

    array forward(array x) {
        std::vector<array> outs{};
        for (size_t i=0; i<N_HEADS; ++i) {
        array head_out = heads[i].forward(x);
        outs.push_back(head_out);
        }

        array out = concatenate(outs, -1);
        return out;
    }
};

array test_card_embed(array x, array card_emb_w, array rank_emb_w, array suit_emb_w) {
  auto B = x.shape()[0];  // 1 
  auto num_cards = x.shape()[1];  // 2
  auto MODEL_DIM = card_emb_w.shape()[1];
  x = reshape(x, {B * num_cards});
  
  auto valid = astype((x >= 0), float32);
  x = maximum(x, zeros_like(x));
  x = astype(x, int32);

  auto rank_indices = floor_divide(x, array(4));
  auto suit_indices = remainder(x, array(4));

  // gather full vectors along first axis
  auto card_embs = squeeze(gather(card_emb_w, {x}, {0}, {1,MODEL_DIM})); // should give [2, MODEL_DIM]
  
  auto rank_embs = squeeze(gather(rank_emb_w, {rank_indices}, {0}, {1,MODEL_DIM}));
  
  auto suit_embs = squeeze(gather(suit_emb_w, {suit_indices}, {0}, {1,MODEL_DIM}));

  auto embs = add(add(card_embs, rank_embs), suit_embs);

  embs = multiply(embs, expand_dims(valid, 1));
  embs = reshape(embs, {B, num_cards, MODEL_DIM});
  return embs; 
}

class PokerGPT : public MyModule {
 public: 
   array card_emb_w = random::normal({52, MODEL_DIM});
   array rank_emb_w = random::normal({13, MODEL_DIM}); 
   array suit_emb_w = random::normal({4, MODEL_DIM});
   array bet_proj_w = random::normal({1, MODEL_DIM});
   array action_head = random::normal({MODEL_DIM, NUM_ACTIONS});
   std::array<MHA, N_LAYERS> layers; 

   PokerGPT() : MyModule("PokerGPT") {
     //DEBUG_NONE("initializing PokerGPT");
     register_array(card_emb_w, "card_emb_w");
     register_array(rank_emb_w, "rank_emb_w");
     register_array(suit_emb_w, "suit_emb_w");
     register_array(bet_proj_w, "bet_proj_w");
     register_array(action_head, "action_head");
     for(size_t i = 0; i < N_LAYERS; i++) {
        std::string layer_name = "layer_" + std::to_string(i);
        register_module(layers[i], layer_name);  // register each head directly
     }
    DEBUG_NONE("pokergpt reg complete");
   }

   array forward(array cards, array bets, array round_ids, std::vector<int>& vec_pos_ids) {
      auto BS = cards.shape()[0];
  
      auto card_emb = test_card_embed(cards, card_emb_w, rank_emb_w, suit_emb_w);
      bets = matmul(reshape(bets, {BS,-1, 1}), bet_proj_w);
      array pos_ids = make_pos_ids(vec_pos_ids);

      array round_pe = get_round_encoding();
      array action_pe = get_action_encoding();

      array preflop = reshape(repeat(take(round_pe, 0, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM});
      array flop = reshape(repeat(take(round_pe, 1, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM}); 
      array turn = reshape(repeat(take(round_pe, 2, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM});
      array river = reshape(repeat(take(round_pe, 3, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM});
      array concat_round_pos = concatenate({preflop, flop, turn, river}, 0); 

      bets = add(add(bets, concat_round_pos), repeat(action_pe, 4, 0));

      array x = concatenate({card_emb, bets}, 1);

      for (size_t i=0; i<NUM_LAYERS; ++i) {
        x = layers[i].forward(x);
      }

      array last_tokens = slice(
        x,
        {0, x.shape(1)-1, 0},
        {x.shape(0), x.shape(1), x.shape(2)}
      );

      array out = matmul(last_tokens, action_head);
      return out;
   }

   array make_pos_ids(const std::vector<int>& pos) {
       array arr = array(pos.data(), {static_cast<int>(pos.size())});
       return arr;
   }

    array get_round_encoding() {
        // Generate base positional encoding for 4 rounds
        return get_sinusoidal_pos_encoding(4, MODEL_DIM);
    }

    array get_action_encoding() {
        // Generate base positional encoding for max actions
        return get_sinusoidal_pos_encoding(NUM_PLAYERS * MAX_ROUND_BETS, MODEL_DIM); 
    }

    array get_sinusoidal_pos_encoding(int length, int dim) {
      // Generate position and dimension indices
      array pos = reshape(arange(length), {length, 1});  // [length, 1]
      array dim_indices = arange(0, dim/2);              // [dim/2]
      
      // Calculate frequencies: 10000^(-2i/d_model)
      array freqs = exp(multiply(dim_indices, array(-log(10000.0) / dim))); // [dim/2]
      
      // Calculate arguments for sin/cos: pos * freqs
      array args = matmul(pos, reshape(freqs, {1, dim/2}));  // [length, dim/2]
      
      // Calculate sin and cos values
      array sin_vals = sin(args);  // [length, dim/2]
      array cos_vals = cos(args);  // [length, dim/2]
      
      // Initialize output array
      array pe = zeros({length, dim}); 
      
      // Create indices for scattering
      array row_indices = repeat(arange(length), dim/2);
      
      // Create column indices
      array sin_col_indices = reshape(multiply(arange(dim/2), array(2)), {-1});  // [0,2,4...]
      array sin_cols = repeat(sin_col_indices, length);
      
      array sin_updates = reshape(sin_vals, {length * dim/2, 1, 1});
      array cos_updates = reshape(cos_vals, {length * dim/2, 1, 1});
      
      // Scatter sin values into even columns
      pe = scatter(pe, 
                  {row_indices, sin_cols},
                  sin_updates,
                  {0, 1});
                  
      // Scatter cos values into odd columns
      pe = scatter(pe,
                  {row_indices, add(sin_cols, array(1))},
                  cos_updates, 
                  {0, 1});
                  
      return pe;
  }
};

enum class Reduction {
    NONE,
    MEAN,
    SUM
};

array reduce(const array& x, Reduction reduction) {
    switch (reduction) {
        case Reduction::NONE:
            return x;
        case Reduction::MEAN:
            return mean(x);
        case Reduction::SUM:
            return sum(x);
        default:
            throw std::runtime_error("Unknown reduction");
    }
}

array smooth_l1_loss(
    const array& predictions,
    const array& targets,
    float beta = 1.0f,
    Reduction reduction = Reduction::MEAN
) {
    if (predictions.shape() != targets.shape()) {
        std::ostringstream msg;
        msg << "Predictions shape " << predictions.shape() 
            << " does not match targets shape " << targets.shape();
        throw std::invalid_argument(msg.str());
    }

    array diff = subtract(predictions, targets);
    array squared_loss = multiply(array(0.5f / beta), square(diff));
    array abs_loss = subtract(abs(diff), array(0.5f * beta));
    
    array loss = where(
        less(abs(diff), array(beta)),
        squared_loss,
        abs_loss
    );

    return reduce(loss, reduction);
}

array mse_loss(
    const array& predictions,
    const array& targets,
    Reduction reduction = Reduction::MEAN
) {
    if (predictions.shape() != targets.shape()) {
        std::ostringstream msg;
        msg << "Predictions shape " << predictions.shape() 
            << " does not match targets shape " << targets.shape();
        throw std::invalid_argument(msg.str());
    }

    array loss = square(subtract(predictions, targets));
    return reduce(loss, reduction);
}

void test_mlx() {
    int n_players = 2;
    int max_round_bets = 6;
    mlx::core::set_default_device(Device::gpu);

    array card_emb_w = random::normal({52, MODEL_DIM});
    array rank_emb_w = random::normal({13, MODEL_DIM});
    array suit_emb_w = random::normal({4, MODEL_DIM});
    
    // define test inputs
    array hand = array({10, 10}, {1,2});
    std::vector<int> pos_vec(NUM_PLAYERS * MAX_ROUND_BETS);
    std::iota(pos_vec.begin(), pos_vec.end(), 0);
    array round_ids = array({0,1,2,3});
    array bets = zeros({1, NUM_PLAYERS*MAX_ROUND_BETS*4});

    PokerGPT gpt;
    AdamOptimizer opt(0.001f);
    opt.init(gpt.parameters());

    size_t n_params = 0; 
    for (auto& [name, param_opt] : gpt.parameters()) {
        DEBUG_NONE("name: " << name);
        DEBUG_NONE("n_params: " <<param_opt.value().size());
        if (param_opt.has_value()) {
            n_params += param_opt.value().size();
        }
    }
    DEBUG_NONE("n params: " << n_params);

    DEBUG_NONE("init opt");
    array targets = zeros(gpt.forward(hand, bets, round_ids, pos_vec).shape());
      // define loss function that takes vector of arrays instead of map
    auto loss_fn = [&](const std::vector<array>& param_arrays) {
        // construct parameter map from vector
        auto param_map = gpt.parameters();
        int i = 0;
        for (auto& [name, param_opt] : param_map) {
            if (param_opt.has_value()) {
                param_opt = param_arrays[i++];
            }
        }
        
        gpt.update(param_map);
        
        array preds = gpt.forward(hand, bets, round_ids, pos_vec);
        // compute loss
        return smooth_l1_loss(preds, targets);
    };

    DEBUG_NONE("got targets");
    // training loop
    for (int step = 0; step < 1000; step++) {
        std::vector<array> param_arrays;
        std::vector<int> argnums;
        int param_idx = 0;
        for (const auto& [name, param_opt] : gpt.parameters()) {
            if (param_opt.has_value()) {
                param_arrays.push_back(param_opt.value());
                argnums.push_back(param_idx);
                param_idx++;
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto grad_fn = mlx::core::value_and_grad(loss_fn, argnums);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        DEBUG_NONE("step took " << duration.count() << " us");
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
        
        DEBUG_NONE("loss val: " << losses.item<float>());

        if (step % 10 == 0) {
            std::cout << "Step " << step << " Loss: " << losses << std::endl;
        }
    }
}

// assumes csv files are in a folder called 'mnist_data' in the current directory
std::string train_file = "/Users/minjunes/Downloads/mnist/mnist_train.csv"; 
std::string test_file = "/Users/minjunes/Downloads/mnist/mnist_test.csv";

class MLPNet : public MyModule {
  public:
   array l1 = random::normal({784, 256}) * sqrt(2.0f/784); // he initialization
   array l2 = random::normal({256, 128}) * sqrt(2.0f/256);
   array l3 = random::normal({128, 64}) * sqrt(2.0f/128);
   array head = random::normal({64, 10}) * sqrt(2.0f/64);
   float dropout_rate = 0.15f;

   MLPNet() : MyModule("MLPNet") {
     register_array(l1, "l1");  
     register_array(l2, "l2");
     register_array(l3, "l3");
     register_array(head, "head");
   }
   
   array forward(array x, bool training = true) {
    x = reshape(x, {-1, 784});
     
    x = matmul(x, l1);
    x = maximum(multiply(x, array(0.1f)), x);
    if(training) {
        array mask = random::uniform(x.shape()) > dropout_rate;
        x = multiply(x, mask) / (1.0f - dropout_rate);
    }
     
    x = matmul(x, l2);
    x = maximum(multiply(x, array(0.1f)), x);
    if(training) {
        array mask = random::uniform(x.shape()) > dropout_rate;
        x = multiply(x, mask) / (1.0f - dropout_rate);
    }
     
    x = matmul(x, l3);
    x = maximum(multiply(x, array(0.1f)), x);
    if(training) {
        array mask = random::uniform(x.shape()) > dropout_rate;
        x = multiply(x, mask) / (1.0f - dropout_rate);
    }
     
    x = matmul(x, head);
    return softmax(x);
}
};

array one_hot(const array& indices, int num_classes) {
    // get batch size from indices shape
    auto bs = indices.shape()[0];
    
    // create output array of zeros [bs, num_classes]
    array out = zeros({bs, num_classes});
    
    // create row indices for scatter
    array row_indices = arange(bs);
    
    // scatter 1s into the right positions
    return scatter(out, 
                  {row_indices, astype(indices, int32)},
                  ones({bs, 1, 1}),
                  {0, 1});
}
 
std::pair<array,array> load_batch(const std::string& csv_file, int batch_size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + csv_file);
    }

    // count total lines first time (could cache this)
    int total_lines = 0;
    std::string line;
    while (std::getline(file, line)) total_lines++;
    
    // subtract 1 if there's a header
    try {
        std::stof(line.substr(0, line.find(',')));
    } catch (...) {
        total_lines--;
    }

    // randomly seek to a position
    std::uniform_int_distribution<> dis(0, total_lines - batch_size);
    int start_pos = dis(gen);
    
    file.clear();
    file.seekg(0);
    
    // skip header if present
    std::getline(file, line);
    
    // skip to random position
    for (int i = 0; i < start_pos; i++) {
        std::getline(file, line);
    }

    // rest of the function same as before
    std::vector<float> images;
    std::vector<float> labels;
    
    int curr_bs = 0;
    while(std::getline(file, line) && curr_bs < batch_size) {
        std::stringstream lineStream(line);
        std::string cell;
        
        // first entry in each row is the label
        if (!std::getline(lineStream, cell, ',')) {
        throw std::runtime_error("missing label value at row " + std::to_string(curr_bs+1)); 
        }
        try {
        labels.push_back(std::stof(cell));
        } catch (const std::exception& e) {
        throw std::runtime_error("invalid label value '" + cell + "' at row " + std::to_string(curr_bs+1)); 
        }
        
        // remaining 784 entries are the pixel values
        int pixel_idx = 0;  
        while(std::getline(lineStream, cell, ',')) {
        try {
            images.push_back(std::stof(cell) / 255.0); // scale to [0,1]
        } catch (const std::exception& e) {
            throw std::runtime_error("invalid pixel value '" + cell + "' at row " + std::to_string(curr_bs+1) + ", column " + std::to_string(pixel_idx+1));
        }
        pixel_idx++;
    }

    if (pixel_idx != 784) {
      throw std::runtime_error("expected 784 pixel values but got " + std::to_string(pixel_idx) + " at row " + std::to_string(curr_bs+1));
    }
    
    curr_bs++;  
  }

    array x = array(images.data(), {curr_bs, 784});
    array y = array(labels.data(), {curr_bs});
    return {x,y};
}

void test_mnist() {
    mlx::core::set_default_device(Device::gpu);
    MLPNet mlp;
    AdamOptimizer opt(0.001f);
    opt.init(mlp.parameters());

    int batch_size = 1024;  // smaller batch to start
    int num_train_iters = 10000;
    int log_interval = 1;  // more frequent logging

    auto loss_fn = [&](const std::vector<array>& param_arrays) {
        auto param_map = mlp.parameters();
        int i = 0;
        for (auto& [name, param_opt] : param_map) {
            if (param_opt.has_value()) {
                param_opt = param_arrays[i++];
            }
        }
        
        mlp.update(param_map);
        
        auto [x, y_indices] = load_batch(train_file, batch_size);
        array y = one_hot(y_indices, 10);
        array logits = mlp.forward(x);
        array log_probs = log(logits + 1e-10);
        array prod = multiply(y, log_probs);
        array summed = sum(prod, -1);
        array loss = -mean(summed);
        return std::vector<array>{loss};
    };

    for (int step = 1; step <= num_train_iters; step++) {
        std::vector<array> param_arrays;
        std::vector<int> argnums;
        int param_idx = 0;
        
        for (const auto& [name, param_opt] : mlp.parameters()) {
            if (param_opt.has_value()) {
                param_arrays.push_back(param_opt.value());
                argnums.push_back(param_idx);
                param_idx++;
            }
        }

        auto grad_fn = mlx::core::value_and_grad(loss_fn, argnums);
        auto [losses, grads] = grad_fn(param_arrays);
        array loss_val = losses[0];

        std::map<std::string, std::optional<array>> grad_map;
        int i = 0;
        for (const auto& [name, param_opt] : mlp.parameters()) {
            if (param_opt.has_value()) {
                grad_map[name] = grads[i++];
            }
        }

        opt.update(mlp, grad_map);
        eval(loss_val);  // make sure we eval loss

        if (step % log_interval == 0) {
            auto [test_x, test_y_indices] = load_batch(test_file, batch_size);
            array test_preds = mlp.forward(test_x);
            array pred_indices = argmax(test_preds, -1);
            array accuracy = mean(astype(equal(pred_indices, test_y_indices), float32));
            
            eval(accuracy);
            
            std::cout << "step " << step 
                << " loss: " << loss_val.item<float>()
                << " test accuracy: " << accuracy.item<float>() 
                << std::endl;
        }
    }
}