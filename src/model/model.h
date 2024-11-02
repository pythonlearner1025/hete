// model.h

#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iterator>
#include "../constants.h"
#include "../debug.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cassert>
#include "model.h"
#include <unordered_map>

// ========================
// CardEmbedding Module
// ========================

struct CardEmbeddingImpl :  torch::nn::Module {
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
        auto x = input.reshape({-1});
        
        // Create a mask for valid cards (input >= 0)
                                                      // set copy to true
        auto valid = (x >= 0).to(torch::kFloat, false, true); // -1 indicates 'no card'
        
        // Clamp negative indices to 0
        x = torch::clamp(x, /*min=*/0);
        
        // Ensure x is of integer type
        if (x.dtype() != torch::kInt) {
            x = x.to(torch::kInt);
        }
        
        // Compute rank and suit indices using integer division and modulo
        auto rank_indices = torch::floor_divide(x, 4).to(torch::kInt);      // [B*num_cards]
        auto suit_indices = torch::remainder(x, 4).to(torch::kInt);        // [B*num_cards]
        
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
        embs = embs.reshape({B, num_cards, -1}).sum(1); // [B,MODEL_DIM]
        //DEBUG_NONE("reshaped");
        return embs;
    }
};
TORCH_MODULE(CardEmbedding); // Creates CardEmbedding as a ModuleHolder<CardEmbeddingImpl>

// ========================
// DeepCFRModel Module
// ========================

struct DeepCFRModelImpl :  torch::nn::Module {
    torch::nn::ModuleList card_embeddings{nullptr};
    torch::nn::Linear card1{nullptr}, card2{nullptr}, card3{nullptr};
    torch::nn::Linear bet1{nullptr}, bet2{nullptr};
    torch::nn::Linear comb1{nullptr}, comb2{nullptr}, comb3{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear action_head{nullptr};
    CardEmbedding hand_embed{nullptr};
    CardEmbedding flop_embed{nullptr};
    CardEmbedding turn_embed{nullptr};
    CardEmbedding river_embed{nullptr};

    // Constructor
    DeepCFRModelImpl(){
        
        int64_t n_card_types = 4;

        hand_embed = register_module("hand_embed", CardEmbedding());
        flop_embed = register_module("flop_embed", CardEmbedding());
        turn_embed = register_module("turn_embed", CardEmbedding());
        river_embed = register_module("river_embed", CardEmbedding());

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
    torch::Tensor forward(
        torch::Tensor hand, 
        torch::Tensor flop, 
        torch::Tensor turn, 
        torch::Tensor river, 
        torch::Tensor bet_fracs, 
        torch::Tensor bet_status
    ) {
        /*
        Parameters:
            cards: std::vector<torch::Tensor> of size n_card_types
                   Each tensor shape: [N, num_cards_in_group] (e.g., [N, 2], [N, 3], etc.)
            bet_fracs: torch::Tensor of shape [N, bet_input_size / 2]
            bet_status: torch::Tensor of shape [N, bet_input_size / 2]
        */

        // 1. Card Branch
        torch::Tensor card_embs_cat;
        torch::Tensor hand_emb = hand_embed->forward(hand);
        torch::Tensor flop_emb = flop_embed->forward(flop);
        torch::Tensor turn_emb = turn_embed->forward(turn);
        torch::Tensor river_emb = river_embed->forward(river);

        // Concatenate embeddings from all card groups
        card_embs_cat = torch::cat({hand_emb, flop_emb, turn_emb, river_emb}, /*dim=*/1); // [N,MODEL_MODEL_DIM * n_card_types]

        // Pass through card linear layers with ReLU activations
        auto x = torch::relu(card1->forward(card_embs_cat)); // [N,MODEL_DIM]
        x = torch::relu(card2->forward(x));                  // [N,MODEL_DIM]
        x = torch::relu(card3->forward(x));                  // [N,MODEL_DIM]

        // 2. Bet Branch
        auto bet_size = bet_fracs.clamp(/*min=*/0, /*max=*/1e6); // Clamp between 0 and 1e6
        auto bet_occurred = bet_status.to(torch::kFloat);        // [N, bet_input_size / 2]
        auto bet_feats = torch::cat({bet_size, bet_occurred}, /*dim=*/1); // [N, bet_input_size]
        
        // Pass through bet linear layers with ReLU and residual connection
        // Inside the forward function, just before the Bet Branch
        /*
        DEBUG_NONE("bet_feats shape: " << bet_feats.sizes());
        DEBUG_NONE("bet1 weight shape: " << bet1->weight.sizes());
        DEBUG_NONE("bet1 bias shape: " << bet1->bias.sizes());
        */
        auto y = torch::relu(bet1->forward(bet_feats));           // [N,MODEL_DIM]
        y = torch::relu(bet2->forward(y) + y);                    // [N,MODEL_DIM]

        // 3. Combined Trunk
        auto z = torch::cat({x, y}, /*dim=*/1);                    // [N, 2 *MODEL_DIM]
        z = torch::relu(comb1->forward(z));                       // [N,MODEL_DIM]
        z = torch::relu(comb2->forward(z) + z);                    // [N,MODEL_DIM] (Residual)
        z = torch::relu(comb3->forward(z) + z);                    // [N,MODEL_DIM] (Residual)
        z = norm->forward(z);                                      // LayerNorm
        
        // Action Head
        auto output = action_head->forward(z);                      // [N, NUM_ACTIONS]
        //std::cout << "forward complete" << std::endl;
        return output;
    }
};
TORCH_MODULE(DeepCFRModel); // Creates DeepCFRModel as a ModuleHolder<DeepCFRModelImpl>
#endif // MODEL_H
