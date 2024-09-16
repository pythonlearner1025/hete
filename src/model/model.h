// model.h

#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <vector>
#include "../constants.h"
#include "../debug.h"

// Forward declaration of the model struct
//struct DeepCFRModelImpl;
//using DeepCFRModel = torch::nn::ModuleHolder<DeepCFRModelImpl>;

// Factory functions using void* for opaqueness
void* create_deep_cfr_model(int64_t n_players, int64_t n_bets, int64_t n_actions, int64_t dim = 256);
torch::Tensor deep_cfr_model_forward(void* model_ptr, std::array<torch::Tensor, 4> cards, torch::Tensor bet_fracs, torch::Tensor bet_status);
void delete_deep_cfr_model(void* model_ptr);
void set_model_eval_mode(void* model_ptr); // If implemented
int64_t get_action_head_dim(void* model_ptr);

#endif // MODEL_H
