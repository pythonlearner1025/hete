// model.h

#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <vector>

// Forward declaration of the model struct
struct DeepCFRModelImpl;
using DeepCFRModel = torch::nn::ModuleHolder<DeepCFRModelImpl>;

// Factory functions using void* for opaqueness
extern void* create_deep_cfr_model(int64_t n_card_types, int64_t n_players, int64_t n_bets, int64_t n_actions, int64_t dim = 256);
extern torch::Tensor deep_cfr_model_forward(void* model_ptr, std::vector<torch::Tensor> cards, torch::Tensor bet_fracs, torch::Tensor bet_status);
extern void delete_deep_cfr_model(void* model_ptr);
extern void set_model_eval_mode(void* model_ptr); // If implemented

#endif // MODEL_H
