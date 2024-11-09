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

// key functions to implement:
// model.update(new_weight_dict) -> apply new weights to all model weights, see 
// https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/base.py
// optimizer apply_gradients
// i'd have to impl apply_single 
/*

 def apply_gradients(self, gradients: dict, parameters: dict):
        """Apply the gradients to the parameters and return the updated parameters.

        Can be used to update a model via
        ``model.update(opt.apply_gradients(grads, model))`` which is precisely
        how :meth:`Optimizer.update` is implemented.

        Args:
            gradients (dict): A Python tree of gradients.
            parameters (dict): A Python tree of parameters. It can be a
              superset of the gradients. In that case the returned python
              tree will be of the same structure as the gradients.
        """
        if not self._initialized:
            self.init(gradients)

        # Update any scheduled variables
        for param, scheduler in self._schedulers.items():
            self.state[param] = scheduler(self.step)

        # Increment the step
        self.state["step"] = self.step + 1

        # Apply the update
        return tree_map(self.apply_single, gradients, parameters, self.state)

*/
class AdamOptimizer {
private:
    float lr;
    float beta1;
    float beta2; 
    float eps;
    int step_count = 0;
    
    // store refs to model parameters for faster access
    std::map<std::string, array*> param_ptrs;
    
    // optimizer state
    std::map<std::string, array> m; // first moment
    std::map<std::string, array> v; // second moment

    // helper for gradient clipping
    array clip_gradients(const array& grad, float max_norm) {
        if (max_norm <= 0) return grad;
        
        array grad_norm = sqrt(sum(square(grad)));
        array scale = minimum(array(max_norm), grad_norm) / grad_norm;
        return multiply(grad, scale);
    }

public:
    AdamOptimizer(
        MyModule& model,
        float learning_rate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f
    ) : lr(learning_rate), beta1(beta1), beta2(beta2), eps(eps) {
        // get parameter map from model
        auto params = model.parameters();
        
        // initialize state
        for (auto& [name, param_opt] : params) {
            if (!param_opt.has_value()) continue;
            
            // get non-const pointer to parameter
            array* param = const_cast<array*>(&param_opt.value());
            param_ptrs[name] = param;
            
            // init momentum and velocity to zeros
            m[name] = zeros_like(*param);
            v[name] = zeros_like(*param);
        }
    }

    void step(const std::map<std::string, array>& grads, float max_norm = -1.0f) {
        step_count++;
        
        // compute bias correction terms
        float bc1 = 1.0f - std::pow(beta1, step_count);
        float bc2 = 1.0f - std::pow(beta2, step_count);
        float lr_t = lr * std::sqrt(bc2) / bc1;

        for (const auto& [name, param_ptr] : param_ptrs) {
            if (grads.count(name) == 0) continue;
            
            const array& grad = grads.at(name);
            array& param = *param_ptr;
            
            // clip gradients if max_norm specified
            array clipped_grad = clip_gradients(grad, max_norm);
            
            // update momentum
            m[name] = add(
                multiply(array(beta1), m[name]),
                multiply(array(1.0f - beta1), clipped_grad)
            );
            
            // update velocity 
            v[name] = add(
                multiply(array(beta2), v[name]),
                multiply(array(1.0f - beta2), square(clipped_grad))
            );
            
            // update parameters
            array update = multiply(
                array(lr_t),
                divide(
                    m[name],
                    add(sqrt(v[name]), array(eps))
                )
            );
            
            param = subtract(param, update);
        }
    }

    // basic scheduler that reduces learning rate by factor after n steps
    void schedule_lr(float factor, int after_steps) {
        if (step_count > after_steps) {
            lr *= factor;
        }
    }
};

class MyModule {
protected:
   struct RegisteredArray {
       array* ptr;
       std::string name;
   };

   struct RegisteredModule {
       MyModule* ptr; 
       std::string name;
   };

   MyModule* parent = nullptr;
   std::vector<RegisteredArray> arrays;
   std::vector<RegisteredModule> modules;
   std::string class_name;

   void register_array(array& arr, const std::string& name) {
       arrays.push_back({&arr, name});
   }

   void register_module(MyModule& module, const std::string& name) {
       module.parent = this;
       modules.push_back({&module, name}); 
   }

   template<typename T, size_t N>
   void register_module_array(std::array<T,N>& arr, const std::string& array_name) {
       for(size_t i = 0; i < N; i++) {
           arr[i].parent = this;
           modules.push_back({&arr[i], array_name + "." + std::to_string(i)});
       }
   }

public:
   MyModule(const std::string& class_name) : class_name(class_name) {}

   std::string get_full_path() const {
       if (!parent) return class_name;
       return parent->get_full_path() + "." + class_name;
   }

    std::map<std::string, std::optional<array>> parameters() {
        std::map<std::string, std::optional<array>> params;
        
        visit([&params](const std::string& path, array& arr) {
            params[path] = std::optional<array>(arr);
        });

        return params;
    }

   void visit(const std::function<void(const std::string&, array&)>& visitor) {
       std::string base_path = get_full_path();

       // visit arrays
       for (auto& reg : arrays) {
           visitor(base_path + "." + reg.name, *reg.ptr);
       }

       // visit nested modules
       for (auto& reg : modules) {
           reg.ptr->visit(visitor);
       }
   }

   void update(const std::map<std::string, std::optional<array>>& params) {
       std::string base_path = get_full_path();
       
       // update arrays
       for (auto& reg : arrays) {
           std::string full_path = base_path + "." + reg.name;
           if (params.count(full_path)) {
               *reg.ptr = params.at(full_path).value();
           }
       }

       // update nested modules
       for (auto& reg : modules) {
           reg.ptr->update(params);
       }
   }
};

constexpr size_t N_HEADS = 2;
constexpr size_t N_LAYERS = 1;
constexpr int HEAD_DIM = MODEL_DIM / N_HEADS; 

class HeadLayerNorm : public MyModule {
  public:
    double eps = 1e-5; 
    array gamma = ones({HEAD_DIM});
    array beta = zeros({HEAD_DIM});

    HeadLayerNorm() : MyModule("LayerNorm") {
        register_array(gamma, "gamma");
        register_array(beta, "beta");
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
       DEBUG_NONE("initializing Decoder");
       register_array(attn_wq, "attn_wq");
       register_array(attn_wk, "attn_wk"); 
       register_array(attn_wv, "attn_wv");
       register_array(attn_out, "attn_out");
       register_array(ffn_1, "ffn_1");
       register_array(ffn_2, "ffn_2");
       register_array(layer_norm, "layer_norm");
       register_module(norm, "norm");
       DEBUG_NONE("Decoder initialization complete");
     }

   array forward(array x) {
      DEBUG_NONE("Decoder forward pass start");
      DEBUG_NONE("input shape: " << x.shape() << " values: " << x);

      array q = matmul(x, attn_wq);
      eval(q);
      DEBUG_NONE("query shape: " << q.shape() << " values: " << q);

      array k = matmul(x, attn_wk);
      eval(k);
      DEBUG_NONE("key shape: " << k.shape() << " values: " << k);

      array v = matmul(x, attn_wv);
      eval(v);
      DEBUG_NONE("value shape: " << v.shape() << " values: " << v);

      array attn = reshape(matmul(q, reshape(k, {k.shape()[1], k.shape()[2], -1})) * sqrt(HEAD_DIM), {1, x.shape()[1], x.shape()[1]});
      eval(attn);
      DEBUG_NONE("attention scores shape: " << attn.shape() << " values: " << attn);

      attn = tril(attn);
      eval(attn);

      x = softmax(attn, -1);
      eval(x);
      DEBUG_NONE("softmax attention shape: " << x.shape() << " values: " << x);

      x = matmul(x, v);
      DEBUG_NONE("attention output shape: " << x.shape() << " values: " << x);

      x = matmul(x, ffn_1);
      eval(x);
      DEBUG_NONE("ffn1 output shape: " << x.shape() << " values: " << x);

      x = maximum(x, zeros_like(x));
      eval(x);
      DEBUG_NONE("relu output shape: " << x.shape() << " values: " << x);

      x = matmul(x, ffn_2);
      eval(x);
      DEBUG_NONE("ffn2 output shape: " << x.shape() << " values: " << x);

      x = norm.forward(x);
      eval(x);
      DEBUG_NONE("layer norm output shape: " << x.shape() << " values: " << x);

      DEBUG_NONE("Decoder forward pass complete");
      return x;
   }
};

class MHA : public MyModule {
 public: 
   std::array<Decoder, N_HEADS> heads;
   array proj = random::normal({MODEL_DIM, MODEL_DIM});

   MHA() : MyModule("MHA") {
     DEBUG_NONE("initializing MHA");
     register_module_array(heads, "head");
     register_array(proj, "proj");
     DEBUG_NONE("MHA initialization complete");
   }

   array forward(array x) {
     DEBUG_NONE("MHA forward pass start");
     DEBUG_NONE("input shape: " << x.shape() << " values: " << x);
     
     std::vector<array> outs{};
     for (size_t i=0; i<N_HEADS; ++i) {
       DEBUG_NONE("processing head " << i);
       array head_out = heads[i].forward(x);
       eval(head_out);
       DEBUG_NONE("head " << i << " output shape: " << head_out.shape() << " values: " << head_out);
       outs.push_back(head_out);
     }

     array out = concatenate(outs, -1);
     eval(out);
     DEBUG_NONE("concatenated output shape: " << out.shape() << " values: " << out);
     DEBUG_NONE("MHA forward pass complete");
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

  DEBUG_NONE("card_embs shape after gather:" << card_embs.shape());

  // Now each embs should be [2, MODEL_DIM]
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
   std::array<MHA, NUM_LAYERS> layers; 

   PokerGPT() : MyModule("PokerGPT") {
     DEBUG_NONE("initializing PokerGPT");
     register_array(card_emb_w, "card_emb_w");
     register_array(rank_emb_w, "rank_emb_w");
     register_array(suit_emb_w, "suit_emb_w");
     register_array(bet_proj_w, "bet_proj_w");
     register_array(action_head, "action_head");
     register_module_array(layers, "layers");
     DEBUG_NONE("PokerGPT initialization complete");
   }

   array forward(array cards, array bets, array round_ids, std::vector<int>& vec_pos_ids) {
      auto BS = cards.shape()[0];
      DEBUG_NONE("PokerGPT forward pass start");
      DEBUG_NONE("input cards shape: " << cards.shape() << " values: " << cards);
      DEBUG_NONE("input bets shape: " << bets.shape() << " values: " << bets);
      DEBUG_NONE("input round_ids shape: " << round_ids.shape() << " values: " << round_ids);
      
      auto card_emb = test_card_embed(cards, card_emb_w, rank_emb_w, suit_emb_w);
      eval(card_emb);
      DEBUG_NONE("card embeddings shape: " << card_emb.shape() << " values: " << card_emb);
      bets = matmul(reshape(bets, {BS,-1, 1}), bet_proj_w);
      eval(bets);
      DEBUG_NONE("projected bets shape: " << bets.shape() << " values: " << bets);

      array pos_ids = make_pos_ids(vec_pos_ids);
      eval(pos_ids);
      DEBUG_NONE("position ids shape: " << pos_ids.shape() << " values: " << pos_ids);

      array round_pe = get_round_encoding();
      array action_pe = get_action_encoding();

      array preflop = reshape(repeat(take(round_pe, 0, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM});
      array flop = reshape(repeat(take(round_pe, 1, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM}); 
      array turn = reshape(repeat(take(round_pe, 2, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM});
      array river = reshape(repeat(take(round_pe, 3, 0), MAX_ROUND_BETS*NUM_PLAYERS), {-1, MODEL_DIM});
      array concat_round_pos = concatenate({preflop, flop, turn, river}, 0); 

      bets = add(add(bets, concat_round_pos), repeat(action_pe, 4, 0));
      eval(bets);
      DEBUG_NONE("final bets shape: " << bets.shape() << " values: " << bets);

      array x = concatenate({card_emb, bets}, 1);
      eval(x);
      DEBUG_NONE("concatenated input shape: " << x.shape() << " values: " << x);

      for (size_t i=0; i<NUM_LAYERS; ++i) {
        DEBUG_NONE("processing layer " << i);
        x = layers[i].forward(x);
        eval(x);
        DEBUG_NONE("layer " << i << " output shape: " << x.shape() << " values: " << x);
      }

      array last_tokens = slice(
        x,
        {0, x.shape(1)-1, 0},
        {x.shape(0), x.shape(1), x.shape(2)}
      );

      DEBUG_NONE("last_tokens: " << last_tokens.shape());
      array out = matmul(last_tokens, action_head);
      eval(out);
      DEBUG_NONE("final output shape: " << out.shape() << " values: " << out);
      DEBUG_NONE("PokerGPT forward pass complete");
      return out;
   }

   array make_pos_ids(const std::vector<int>& pos) {
       DEBUG_NONE("making position ids from vector of size: " << pos.size());
       array arr = array(pos.data(), {static_cast<int>(pos.size())});
       eval(arr);
       DEBUG_NONE("created position ids shape: " << arr.shape() << " values: " << arr);
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
      
      // Reshape updates to match ndim requirements (indices.ndim() + a.ndim())
      // indices has shape [N], a has shape [length, dim]
      // so updates needs shape [N, 1, 1]
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
void test_parameter_update() {
   PokerGPT gpt;
   
   // get params and print
   auto params = gpt.parameters();
   for (const auto& [path, arr] : params) {
       DEBUG_NONE("param " << path << " shape: " << arr.value().shape());
   }

   // create zero params map 
   std::map<std::string, std::optional<array>> zero_params;
   for (const auto& [path, arr] : params) {
       zero_params[path] = zeros_like(arr.value());
   }

   DEBUG_NONE("created zero params, updating model...");
   
   // update model
   gpt.update(zero_params);
   
   // verify all params are zero
   auto new_params = gpt.parameters();
   for (const auto& [path, arr] : new_params) {
       // sum should be 0 for all
       array sum = mlx::core::sum(arr.value());
       eval(sum);
       DEBUG_NONE("param " << path << " sum: " << sum.item<float>());
   }
}

void test_mlx() {
    int n_players = 2;
    int max_round_bets = 6;
    mlx::core::set_default_device(Device::gpu);

    array card_emb_w = random::normal({52, MODEL_DIM});
    array rank_emb_w = random::normal({13, MODEL_DIM});
    array suit_emb_w = random::normal({4, MODEL_DIM});
    // define test hand, flops, turns, rivers, bet_fracs, bet_status
    array hand = array({10, 10}, {1,2});
    std::vector<int> pos_vec(NUM_PLAYERS * MAX_ROUND_BETS);
    std::iota(pos_vec.begin(), pos_vec.end(), 0);
    array round_ids = array({0,1,2,3});
    array bets = zeros({1, NUM_PLAYERS*MAX_ROUND_BETS*4});

    PokerGPT gpt;

    test_parameter_update();
    
    array preds = gpt.forward(hand, bets, round_ids, pos_vec);
    array y = random::normal(preds.shape());

    array loss = smooth_l1_loss(preds, y);
    // much simpler - just compute loss directly
    // get value and all gradients 
    auto value_and_grad_fn = mlx::core::value_and_grad([&](const std::vector<array>& params) {
        return loss;
    });

    // extract current params
    std::vector<array> current_params;
    auto param_map = gpt.parameters();
    for (const auto& [_, param] : param_map) {
        current_params.push_back(param.value());
    }

    // compute loss value and grads
    auto [loss_val, grads] = value_and_grad_fn(current_params);

    // map grads to param names
    std::map<std::string, array> grad_map;
    int i = 0;
    for (const auto& [name, _] : param_map) {
        grad_map[name] = grads[i++];
    }

    AdamOptimizer opt(gpt);
    opt.step(grad_map);

  return;
}
