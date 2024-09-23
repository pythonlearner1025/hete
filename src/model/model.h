// model.h

#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include "../constants.h"
#include "../debug.h"

// Forward declaration of the model struct
//struct DeepCFRModelImpl;
//using DeepCFRModel = torch::nn::ModuleHolder<DeepCFRModelImpl>;

// Factory functions using void* for opaqueness
void* create_deep_cfr_model();
torch::Tensor deep_cfr_model_forward(
    void* model_ptr, 
    torch::Tensor hands, 
    torch::Tensor flops, 
    torch::Tensor turns, 
    torch::Tensor rivers, 
    torch::Tensor bet_fracs, 
    torch::Tensor bet_status
);
void delete_deep_cfr_model(void* model_ptr);
void set_model_eval_mode(void* model_ptr); // If implemented
std::vector<torch::Tensor> get_model_parameters(void* model_ptr);
void save_model(void* model_ptr, const std::string& path);
void* load_model(const std::string& path);

// test fn
void profile_net();

#endif // MODEL_H
