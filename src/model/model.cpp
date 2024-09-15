// model.cpp

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cassert>

// ========================
// CardEmbedding Module
// ========================

struct CardEmbeddingImpl : torch::nn::Module {
    torch::nn::Embedding rank{nullptr}, suit{nullptr}, card{nullptr};
    int64_t dim;

    // Constructor
    CardEmbeddingImpl(int64_t dim_) : dim(dim_) {
        // Initialize Embedding layers
        rank = register_module("rank", torch::nn::Embedding(13, dim));
        suit = register_module("suit", torch::nn::Embedding(4, dim));
        card = register_module("card", torch::nn::Embedding(52, dim));
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
        auto card_embs = card->forward(x);             // [B*num_cards, dim]
        auto rank_embs = rank->forward(rank_indices);  // [B*num_cards, dim]
        auto suit_embs = suit->forward(suit_indices);  // [B*num_cards, dim]
        
        // Sum the embeddings
        auto embs = card_embs + rank_embs + suit_embs; // [B*num_cards, dim]
        
        // Zero out embeddings for 'no card'
        embs = embs * valid.unsqueeze(1); // [B*num_cards, dim]
        
        // Reshape and sum across cards
        embs = embs.view({B, num_cards, -1}).sum(1); // [B, dim]
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
    int64_t dim;

    // Constructor
    DeepCFRModelImpl(int64_t n_players, int64_t n_bets, int64_t n_actions, int64_t dim_ = 256)
        : dim(dim_) {
        
        int64_t n_card_types = 4;
        // Initialize card_embeddings ModuleList
        card_embeddings = register_module("card_embeddings", torch::nn::ModuleList());

        for(int64_t i = 0; i < n_card_types; ++i){
            // Create and add CardEmbedding modules to the list
            auto card_embedding = CardEmbedding(dim);
            card_embeddings->push_back(card_embedding);
        }

        // Initialize card linear layers
        card1 = register_module("card1", torch::nn::Linear(dim * n_card_types, dim));
        card2 = register_module("card2", torch::nn::Linear(dim, dim));
        card3 = register_module("card3", torch::nn::Linear(dim, dim));

        // Initialize bet linear layers
        int64_t nrounds = 4;
        // Calculate input size based on the formula: (n_bets * n_players * nrounds - nrounds) * 2
        int64_t bet_input_size = (n_bets * n_players * nrounds) * 2;
        //std::cout << "expected bet_input_size: " + std::to_string(bet_input_size) << std::endl; 
        bet1 = register_module("bet1", torch::nn::Linear(bet_input_size, dim));
        bet2 = register_module("bet2", torch::nn::Linear(dim, dim));

        // Initialize combined trunk layers
        comb1 = register_module("comb1", torch::nn::Linear(2 * dim, dim));
        comb2 = register_module("comb2", torch::nn::Linear(dim, dim));
        comb3 = register_module("comb3", torch::nn::Linear(dim, dim));
        
        // Correct LayerNorm initialization with a vector
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        
        action_head = register_module("action_head", torch::nn::Linear(dim, n_actions));
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

        // 1. Card Branch
        std::vector<torch::Tensor> card_embs;
        int64_t n_card_types = card_embeddings->size();
        for(int64_t i = 0; i < n_card_types; ++i){
            // Retrieve the i-th CardEmbedding module
            auto embedding_module = std::dynamic_pointer_cast<CardEmbeddingImpl>(card_embeddings->ptr(i));
            if (!embedding_module){
                throw std::runtime_error("Module is not of type CardEmbeddingImpl");
            }
            // Forward pass through CardEmbedding
            auto card_emb = embedding_module->forward(cards[i]); // [N, dim]
            card_embs.push_back(card_emb);
        }
        // Concatenate embeddings from all card groups
        auto card_embs_cat = torch::cat(card_embs, /*dim=*/1); // [N, dim * n_card_types]
        
        // Pass through card linear layers with ReLU activations
        auto x = torch::relu(card1->forward(card_embs_cat)); // [N, dim]
        x = torch::relu(card2->forward(x));                  // [N, dim]
        x = torch::relu(card3->forward(x));                  // [N, dim]

        // 2. Bet Branch
        auto bet_size = bet_fracs.clamp(/*min=*/0, /*max=*/1e6); // Clamp between 0 and 1e6
        auto bet_occurred = bet_status.to(torch::kFloat);        // [N, bet_input_size / 2]
        auto bet_feats = torch::cat({bet_size, bet_occurred}, /*dim=*/1); // [N, bet_input_size]
        
        // Pass through bet linear layers with ReLU and residual connection
        auto y = torch::relu(bet1->forward(bet_feats));           // [N, dim]
        y = torch::relu(bet2->forward(y) + y);                    // [N, dim]

        // 3. Combined Trunk
        auto z = torch::cat({x, y}, /*dim=*/1);                    // [N, 2 * dim]
        z = torch::relu(comb1->forward(z));                       // [N, dim]
        z = torch::relu(comb2->forward(z) + z);                    // [N, dim] (Residual)
        z = torch::relu(comb3->forward(z) + z);                    // [N, dim] (Residual)
        z = norm->forward(z);                                      // LayerNorm
        
        // Action Head
        auto output = action_head->forward(z);                      // [N, n_actions]
        //std::cout << "forward complete" << std::endl;
        return output;
    }
};
TORCH_MODULE(DeepCFRModel); // Creates DeepCFRModel as a ModuleHolder<DeepCFRModelImpl>

// ========================
// Example Usage
// ========================

// Factory functions
void* create_deep_cfr_model(int64_t n_players, int64_t n_bets, int64_t n_actions, int64_t dim = 256) {
    return new DeepCFRModel(n_players, n_bets, n_actions, dim);
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

int64_t get_action_head_dim(void* model_ptr) {
    if (model_ptr == nullptr) {
        throw std::invalid_argument("Model pointer is null.");
    }
    DeepCFRModel* model = static_cast<DeepCFRModel*>(model_ptr);
    return (*model)->action_head->weight.size(0);
}

int profile_net() {
    try {
        // Define model parameters
        int64_t n_card_types = 2; // e.g., hole and board
        int64_t n_players = 2;
        int64_t n_bets = 10;
        int64_t n_actions = 10;
        int64_t dim = 256;

        // Instantiate the model
        DeepCFRModel model(n_players, n_bets, n_actions, dim);

        // Set the model to evaluation mode
        model->eval();

        // Create dummy input data
        int64_t N = 5; // Batch size
        std::array<torch::Tensor, 4> cards;
        
        // Example card groups: [N, 2] and [N, 3]
        // Using -1 to indicate 'no card'
        cards[0] = torch::randint(-1, 52, {N, 2}, torch::kInt64);
        cards[1] = torch::randint(-1, 52, {N, 3}, torch::kInt64);
        
        // Calculate bet_input_size based on the constructor formula
        int64_t nrounds = 4;
        int64_t bet_input_size = (n_bets * n_players * nrounds - nrounds) * 2;
        
        // Create dummy bet_fracs and bet_status tensors
        torch::Tensor bet_fracs = torch::rand({N, (n_bets * n_players * nrounds - nrounds)}, torch::kFloat);
        torch::Tensor bet_status = torch::randint(0, 2, {N, (n_bets * n_players * nrounds - nrounds)}, torch::kInt64);

        // Measure average model forward call time over 1000 calls
        int num_calls = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_calls; ++i) {
            torch::Tensor output = model->forward(cards, bet_fracs, bet_status);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_time = duration.count() / static_cast<double>(num_calls);
        
        std::cout << "Average forward call time over " << num_calls << " calls: " 
                  << avg_time << " microseconds" << std::endl;
        
        // avg time is 203.878 microseconds

    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.msg() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception occurred!" << std::endl;
        return -1;
    }

    return 0;
}
