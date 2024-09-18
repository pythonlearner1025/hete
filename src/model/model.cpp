// model.cpp

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cassert>
#include "model.h"

// ========================
// CardEmbedding Module
// ========================

struct CardEmbeddingImpl : torch::nn::Module {
    torch::nn::Embedding rank{nullptr}, suit{nullptr}, card{nullptr};

    // Constructor
    CardEmbeddingImpl() {
        // Initialize Embedding layers
        rank = register_module("rank", torch::nn::Embedding(13,MODEL_DIM));
        suit = register_module("suit", torch::nn::Embedding(4,MODEL_DIM));
        card = register_module("card", torch::nn::Embedding(52,MODEL_DIM));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor input) {
        auto B = input.size(0);
        auto num_cards = input.size(1);
        
        // Flatten the input
        auto x = input.view({-1});
        
        // Create a mask for valid cards (input >= 0)
        auto valid = (x >= 0).to(torch::kFloat); // -1 indicates 'no card'
        
        // Clamp negative indices to 0
        x = torch::clamp(x, /*min=*/0);
        
        // Ensure x is of integer type
        if (x.dtype() != torch::kInt64 && x.dtype() != torch::kInt32) {
            x = x.to(torch::kInt64);
        }
        
        // Compute rank and suit indices using integer division and modulo
        auto rank_indices = torch::floor_divide(x, 4).to(torch::kInt64);      // [B*num_cards]
        auto suit_indices = torch::remainder(x, 4).to(torch::kInt64);        // [B*num_cards]
        
        // Ensure that rank_indices and suit_indices are within valid ranges
        rank_indices = torch::clamp(rank_indices, 0, 12); // Ranks: 0-12
        suit_indices = torch::clamp(suit_indices, 0, 3);  // Suits: 0-3
        
        // Compute embeddings
        auto card_embs = card->forward(x);             // [B*num_cards,MODEL_DIM]
        auto rank_embs = rank->forward(rank_indices);  // [B*num_cards,MODEL_DIM]
        auto suit_embs = suit->forward(suit_indices);  // [B*num_cards,MODEL_DIM]
        
        // Sum the embeddings
        auto embs = card_embs + rank_embs + suit_embs; // [B*num_cards,MODEL_DIM]
        
        // Zero out embeddings for 'no card'
        embs = embs * valid.unsqueeze(1); // [B*num_cards,MODEL_DIM]
        
        // Reshape and sum across cards
        embs = embs.view({B, num_cards, -1}).sum(1); // [B,MODEL_DIM]
        return embs;
    }
};
TORCH_MODULE(CardEmbedding); // Creates CardEmbedding as a ModuleHolder<CardEmbeddingImpl>

// ========================
// DeepCFRModel Module
// ========================

struct DeepCFRModelImpl : torch::nn::Module {
    torch::nn::ModuleList card_embeddings{nullptr};
    torch::nn::Linear card1{nullptr}, card2{nullptr}, card3{nullptr};
    torch::nn::Linear bet1{nullptr}, bet2{nullptr};
    torch::nn::Linear comb1{nullptr}, comb2{nullptr}, comb3{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear action_head{nullptr};

    // Constructor
    DeepCFRModelImpl(){
        
        int64_t n_card_types = 4;
        // Initialize card_embeddings ModuleList
        card_embeddings = register_module("card_embeddings", torch::nn::ModuleList());

        for(int64_t i = 0; i < n_card_types; ++i){
            // Create and add CardEmbedding modules to the list
            CardEmbedding embedding;
            auto card_embedding = embedding;
            card_embeddings->push_back(card_embedding);
        }

        // Initialize card linear layers
        card1 = register_module("card1", torch::nn::Linear(MODEL_DIM* n_card_types,MODEL_DIM));
        card2 = register_module("card2", torch::nn::Linear(MODEL_DIM,MODEL_DIM));
        card3 = register_module("card3", torch::nn::Linear(MODEL_DIM,MODEL_DIM));

        // Initialize bet linear layers
        // Calculate input size based on the formula: (MAX_ROUND_BETS * NUM_PLAYERS * nrounds - nrounds) * 2
        int64_t bet_input_size = (MAX_ROUND_BETS * NUM_PLAYERS * 4) * 2;
        //std::cout << "expected bet_input_size: " + std::to_string(bet_input_size) << std::endl; 
        bet1 = register_module("bet1", torch::nn::Linear(bet_input_size,MODEL_DIM));
        bet2 = register_module("bet2", torch::nn::Linear(MODEL_DIM,MODEL_DIM));

        // Initialize combined trunk layers
        comb1 = register_module("comb1", torch::nn::Linear(2 *MODEL_DIM,MODEL_DIM));
        comb2 = register_module("comb2", torch::nn::Linear(MODEL_DIM,MODEL_DIM));
        comb3 = register_module("comb3", torch::nn::Linear(MODEL_DIM,MODEL_DIM));
        
        // Correct LayerNorm initialization with a vector
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({MODEL_DIM})));
        
        action_head = register_module("action_head", torch::nn::Linear(MODEL_DIM, NUM_ACTIONS));
    }

    // Forward pass
    torch::Tensor forward(std::array<torch::Tensor, 4> cards, torch::Tensor bet_fracs, torch::Tensor bet_status) {
        /*
        Parameters:
            cards: std::vector<torch::Tensor> of size n_card_types
                   Each tensor shape: [N, num_cards_in_group] (e.g., [N, 2], [N, 3], etc.)
            bet_fracs: torch::Tensor of shape [N, bet_input_size / 2]
            bet_status: torch::Tensor of shape [N, bet_input_size / 2]
        */

        DEBUG_INFO("Entering DeepCFRModelImpl::forward");
        // Add checks for tensor shapes and types
        for (int i = 0; i < 4; ++i) {
            if (!cards[i].defined()) {
                throw std::runtime_error("Card tensor " + std::to_string(i) + " is not defined");
            }
            DEBUG_INFO("Card " << i << " shape: " << cards[i].sizes());
        }
        if (!bet_fracs.defined() || !bet_status.defined()) {
            throw std::runtime_error("bet_fracs or bet_status is not defined");
        }
        DEBUG_INFO("bet_fracs shape: " << bet_fracs.sizes());
        DEBUG_INFO("bet_status shape: " << bet_status.sizes());

        // 1. Card Branch
        std::vector<torch::Tensor> card_embs;
        int64_t n_card_types = card_embeddings->size();
        for(int64_t i = 0; i < n_card_types; ++i){
            // Retrieve the i-th CardEmbedding module
            auto embedding_module = std::dynamic_pointer_cast<CardEmbeddingImpl>(card_embeddings->ptr(i));
            if (!embedding_module){
                DEBUG_INFO("moduel != cardembeddingimpl");
                throw std::runtime_error("Module is not of type CardEmbeddingImpl");
            }
            // Forward pass through CardEmbedding
            auto card_emb = embedding_module->forward(cards[i]); // [N,MODEL_DIM]
            card_embs.push_back(card_emb);
            DEBUG_INFO("cardembed done");
        }
        // Concatenate embeddings from all card groups
        auto card_embs_cat = torch::cat(card_embs, /*dim=*/1); // [N,MODEL_MODEL_DIM* n_card_types]
        DEBUG_INFO("cardconcat");
        
        // Pass through card linear layers with ReLU activations
        auto x = torch::relu(card1->forward(card_embs_cat)); // [N,MODEL_DIM]
        x = torch::relu(card2->forward(x));                  // [N,MODEL_DIM]
        x = torch::relu(card3->forward(x));                  // [N,MODEL_DIM]
        DEBUG_INFO("card forwqrds");

        // 2. Bet Branch
        auto bet_size = bet_fracs.clamp(/*min=*/0, /*max=*/1e6); // Clamp between 0 and 1e6
        auto bet_occurred = bet_status.to(torch::kFloat);        // [N, bet_input_size / 2]
        auto bet_feats = torch::cat({bet_size, bet_occurred}, /*dim=*/1); // [N, bet_input_size]
        DEBUG_INFO("bet embed complete");
        
        // Pass through bet linear layers with ReLU and residual connection
        // Inside the forward function, just before the Bet Branch
        DEBUG_INFO("bet_feats shape: " << bet_feats.sizes());
        DEBUG_INFO("bet1 weight shape: " << bet1->weight.sizes());
        DEBUG_INFO("bet1 bias shape: " << bet1->bias.sizes());
        auto y = torch::relu(bet1->forward(bet_feats));           // [N,MODEL_DIM]
        y = torch::relu(bet2->forward(y) + y);                    // [N,MODEL_DIM]
        DEBUG_INFO("bet12 complete");

        // 3. Combined Trunk
        auto z = torch::cat({x, y}, /*dim=*/1);                    // [N, 2 *MODEL_DIM]
        z = torch::relu(comb1->forward(z));                       // [N,MODEL_DIM]
        z = torch::relu(comb2->forward(z) + z);                    // [N,MODEL_DIM] (Residual)
        z = torch::relu(comb3->forward(z) + z);                    // [N,MODEL_DIM] (Residual)
        z = norm->forward(z);                                      // LayerNorm
        
        // Action Head
        auto output = action_head->forward(z);                      // [N, NUM_ACTIONS]
        DEBUG_INFO("act head complete");
        //std::cout << "forward complete" << std::endl;
        return output;
    }
};
TORCH_MODULE(DeepCFRModel); // Creates DeepCFRModel as a ModuleHolder<DeepCFRModelImpl>

// ========================
// Example Usage
// ========================

// Factory functions
void* create_deep_cfr_model() {
    return new DeepCFRModel;
}

torch::Tensor deep_cfr_model_forward(void* model_ptr, std::array<torch::Tensor, 4> cards, torch::Tensor bet_fracs, torch::Tensor bet_status) {
    if (model_ptr == nullptr) {
        throw std::invalid_argument("Model pointer is null.");
    }
    DeepCFRModel* model = static_cast<DeepCFRModel*>(model_ptr);
    return (*model)->forward(cards, bet_fracs, bet_status); // Corrected line
}

// Factory function to delete a DeepCFRModel instance
void delete_deep_cfr_model(void* model_ptr) {
    if (model_ptr != nullptr) {
        DeepCFRModel* model = static_cast<DeepCFRModel*>(model_ptr);
        delete model;
    }
}

void set_model_eval_mode(void* model_ptr) {
    if (model_ptr == nullptr) {
        throw std::invalid_argument("Model pointer is null.");
    }
    DeepCFRModel* model = static_cast<DeepCFRModel*>(model_ptr);
    (*model)->eval(); // Correctly call eval()
}

// Add this to model.cpp
std::vector<torch::Tensor> get_model_parameters(void* model_ptr) {
    if (model_ptr == nullptr) {
        throw std::invalid_argument("Model pointer is null.");
    }
    DeepCFRModel* model = static_cast<DeepCFRModel*>(model_ptr);
    std::vector<torch::Tensor> params;
    for (auto& p : (*model)->parameters()) {
        if (p.requires_grad()) {
            params.push_back(p);
        }
    }
    return params;
}

/*
    BATCH IS FASTER

    Total forward call time over 1326 calls: 161239 microseconds
    Average batched forward call time over 30 calls @ BS=1326: 7790.6 microseconds
    ~20x speedup
*/

void create_batch(const std::array<torch::Tensor, 4>& cards_template, const torch::Tensor& bet_fracs_template, const torch::Tensor& bet_status_template, int64_t batch_size, std::array<torch::Tensor, 4>& batched_cards, torch::Tensor& batched_bet_fracs, torch::Tensor& batched_bet_status) {
    // Since input values don't matter, we'll create tensors with the correct shapes
    for (int i = 0; i < 4; ++i) {
        if (cards_template[i].defined()) {
            auto card_shape = cards_template[i].sizes(); // Original shape
            std::vector<int64_t> new_shape(card_shape.begin(), card_shape.end());
            new_shape[0] = batch_size; // Update batch size
            batched_cards[i] = torch::zeros(new_shape, cards_template[i].options());
        } else {
            batched_cards[i] = torch::Tensor();
        }
    }

    auto bet_fracs_shape = bet_fracs_template.sizes();
    std::vector<int64_t> new_bet_fracs_shape(bet_fracs_shape.begin(), bet_fracs_shape.end());
    new_bet_fracs_shape[0] = batch_size;
    batched_bet_fracs = torch::zeros(new_bet_fracs_shape, bet_fracs_template.options());

    auto bet_status_shape = bet_status_template.sizes();
    std::vector<int64_t> new_bet_status_shape(bet_status_shape.begin(), bet_status_shape.end());
    new_bet_status_shape[0] = batch_size;
    batched_bet_status = torch::zeros(new_bet_status_shape, bet_status_template.options());
}

void profile_net() {
    // Define model parameters
    int64_t n_card_types = 4;

    // Instantiate the model
    DeepCFRModel model;

    // Set the model to evaluation mode
    model->eval();

    // Create dummy input data for a single batch
    int64_t N = 1; // Original batch size
    std::array<torch::Tensor, 4> cards;
    cards[0] = torch::zeros({N,2}, torch::kInt64);
    cards[1] = torch::zeros({N,3}, torch::kInt64);
    cards[2] = torch::zeros({N,1}, torch::kInt64);
    cards[3] = torch::zeros({N,1}, torch::kInt64);

    int64_t nrounds = 4;
    int64_t bet_input_size = (MAX_ROUND_BETS * NUM_PLAYERS * 4);

    torch::Tensor bet_fracs = torch::zeros({N,bet_input_size}, torch::kFloat);
    torch::Tensor bet_status = torch::zeros({N,bet_input_size}, torch::kInt64);

    // Measure average model forward call time over 1000 calls with original batch size
    int num_calls = 13260;
    std::cout << "calling" << std::endl;    
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_calls; ++i) {
        torch::Tensor output = model->forward(cards, bet_fracs, bet_status);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Total forward call time over " << num_calls << " calls: "
                << duration.count() << " microseconds" << std::endl;

    // Now measure the average time of batched inference over 30 calls
    int batch_num_calls = 30;
    int64_t batch_size = 13260; // New batch size for batched inference
    double batch_total_time = 0.0;

    // Create batched inputs
    std::array<torch::Tensor, 4> batched_cards;
    torch::Tensor batched_bet_fracs;
    torch::Tensor batched_bet_status;

    create_batch(cards, bet_fracs, bet_status, batch_size, batched_cards, batched_bet_fracs, batched_bet_status);

    // Warm-up call to ensure any lazy initialization is done
    model->forward(batched_cards, batched_bet_fracs, batched_bet_status);

    for (int i = 0; i < batch_num_calls; ++i) {
        auto batch_start = std::chrono::high_resolution_clock::now();

        torch::Tensor output = model->forward(batched_cards, batched_bet_fracs, batched_bet_status);

        auto batch_end = std::chrono::high_resolution_clock::now();

        auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);

        batch_total_time += batch_duration.count();
    }

    double batch_avg_time = batch_total_time / static_cast<double>(batch_num_calls);

    std::cout << "Average batched forward call time over " << batch_num_calls << " calls: "
                << batch_avg_time << " microseconds" << std::endl;

}
