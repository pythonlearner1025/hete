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

struct CardEmbeddingImpl : torch::nn::Module {
    torch::nn::Embedding rank{nullptr}, suit{nullptr}, card{nullptr};

    // Constructor
    CardEmbeddingImpl() {
        // Initialize Embedding layers
        rank = register_module("rank", torch::nn::Embedding(13, MODEL_DIM));
        suit = register_module("suit", torch::nn::Embedding(4, MODEL_DIM));
        card = register_module("card", torch::nn::Embedding(52, MODEL_DIM));
    }

    // Forward pass with mask output
    std::tuple<torch::Tensor, torch::Tensor> forward_with_mask(torch::Tensor input) {
        auto B = input.size(0);
        auto num_cards = input.size(1);

        // Flatten the input
        auto x = input.reshape({-1});

        // Create a mask for valid cards (input >= 0)
        auto valid = (x >= 0).to(torch::kFloat, false, true); // -1 indicates 'no card'

        // Clamp negative indices to 0
        x = torch::clamp(x, /*min=*/0);

        // Ensure x is of integer type
        if (x.dtype() != torch::kInt) {
            x = x.to(torch::kInt);
        }

        // Compute rank and suit indices
        auto rank_indices = torch::floor_divide(x, 4).to(torch::kInt); // [B*num_cards]
        auto suit_indices = torch::remainder(x, 4).to(torch::kInt);     // [B*num_cards]

        // Compute embeddings
        auto card_embs = card->forward(x);             // [B*num_cards, MODEL_DIM]
        auto rank_embs = rank->forward(rank_indices);  // [B*num_cards, MODEL_DIM]
        auto suit_embs = suit->forward(suit_indices);  // [B*num_cards, MODEL_DIM]

        // Sum the embeddings
        auto embs = card_embs + rank_embs + suit_embs; // [B*num_cards, MODEL_DIM]

        // Zero out embeddings for 'no card'
        embs = embs * valid.unsqueeze(1); // [B*num_cards, MODEL_DIM]

        // Reshape to [B, num_cards, MODEL_DIM]
        embs = embs.reshape({B, num_cards, MODEL_DIM}); // [B, num_cards, MODEL_DIM]

        // Reshape valid mask
        valid = valid.reshape({B, num_cards}); // [B, num_cards]

        return std::make_tuple(embs, valid);
    }
};
TORCH_MODULE(CardEmbedding); // Creates CardEmbedding as a ModuleHolder<CardEmbeddingImpl>

// ========================
// DeepCFRModel Module
// ========================

struct DeepCFRModelImpl : torch::nn::Module {
    // Embeddings
    CardEmbedding hand_embed{nullptr};
    CardEmbedding flop_embed{nullptr};
    CardEmbedding turn_embed{nullptr};
    CardEmbedding river_embed{nullptr};

    // Transformer encoders
    torch::nn::TransformerEncoder card_encoder{nullptr};
    torch::nn::TransformerEncoder bet_encoder{nullptr};

    // Bet embedding
    torch::nn::Linear bet_embedding{nullptr};

    // Combined trunk layers
    torch::nn::Linear comb1{nullptr}, comb2{nullptr}, comb3{nullptr};

    // Normalization
    torch::nn::LayerNorm norm{nullptr};

    // Action head
    torch::nn::Linear action_head{nullptr};

    // Constructor
    DeepCFRModelImpl() {
        int64_t n_card_types = 4; // Hand, flop, turn, river

        // Initialize card embeddings
        hand_embed = register_module("hand_embed", CardEmbedding());
        flop_embed = register_module("flop_embed", CardEmbedding());
        turn_embed = register_module("turn_embed", CardEmbedding());
        river_embed = register_module("river_embed", CardEmbedding());

        // Initialize card TransformerEncoder
        auto card_encoder_layer_options = torch::nn::TransformerEncoderLayerOptions(MODEL_DIM, NUM_HEADS);
        card_encoder_layer_options.activation(torch::kGELU);
        auto card_encoder_layer = torch::nn::TransformerEncoderLayer(card_encoder_layer_options);
        card_encoder = register_module("card_encoder", torch::nn::TransformerEncoder(card_encoder_layer, NUM_LAYERS));

        // Initialize bet embedding
        bet_embedding = register_module("bet_embedding", torch::nn::Linear(2, MODEL_DIM));

        // Initialize bet TransformerEncoder
        auto bet_encoder_layer_options = torch::nn::TransformerEncoderLayerOptions(MODEL_DIM, NUM_HEADS);
        bet_encoder_layer_options.activation(torch::kGELU);
        auto bet_encoder_layer = torch::nn::TransformerEncoderLayer(bet_encoder_layer_options);
        bet_encoder = register_module("bet_encoder", torch::nn::TransformerEncoder(bet_encoder_layer, NUM_LAYERS));

        // Initialize combined trunk layers
        comb1 = register_module("comb1", torch::nn::Linear(2 * MODEL_DIM, MODEL_DIM));
        comb2 = register_module("comb2", torch::nn::Linear(MODEL_DIM, MODEL_DIM));
        comb3 = register_module("comb3", torch::nn::Linear(MODEL_DIM, MODEL_DIM));

        // LayerNorm
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({MODEL_DIM})));

        // Action head
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
        // Card Embeddings with masks
        torch::Tensor hand_emb, hand_mask;
        std::tie(hand_emb, hand_mask) = hand_embed->forward_with_mask(hand);

        torch::Tensor flop_emb, flop_mask;
        std::tie(flop_emb, flop_mask) = flop_embed->forward_with_mask(flop);

        torch::Tensor turn_emb, turn_mask;
        std::tie(turn_emb, turn_mask) = turn_embed->forward_with_mask(turn);

        torch::Tensor river_emb, river_mask;
        std::tie(river_emb, river_mask) = river_embed->forward_with_mask(river);

        // Concatenate embeddings and masks
        auto card_embs = torch::cat({hand_emb, flop_emb, turn_emb, river_emb}, /*dim=*/1);   // [B, total_num_cards, MODEL_DIM]
        auto card_masks = torch::cat({hand_mask, flop_mask, turn_mask, river_mask}, /*dim=*/1); // [B, total_num_cards]

        // Create attention mask for cards
        auto src_key_padding_mask = ~card_masks.to(torch::kBool); // [B, total_num_cards]

        // Transpose embeddings for Transformer
        auto card_embs_transposed = card_embs.transpose(0, 1); // [sequence_length, B, MODEL_DIM]

        // Process through TransformerEncoder
        auto card_encodings = card_encoder->forward(card_embs_transposed, /*src_mask=*/{}, /*src_key_padding_mask=*/src_key_padding_mask); // [sequence_length, B, MODEL_DIM]

        // Transpose back
        card_encodings = card_encodings.transpose(0, 1); // [B, sequence_length, MODEL_DIM]

        // Aggregate the card encodings
        auto card_masks_expanded = card_masks.unsqueeze(2); // [B, sequence_length, 1]
        auto x = (card_encodings * card_masks_expanded).sum(1) / card_masks_expanded.sum(1).clamp_min(1e-9); // [B, MODEL_DIM]

        // Bet Embeddings
        // Combine bet_fracs and bet_status
        auto bet_feats = torch::cat({bet_fracs.unsqueeze(2), bet_status.unsqueeze(2)}, /*dim=*/2); // [B, num_bets, 2]

        // Create bet masks (assuming -1 indicates padding)
        auto bet_masks = (bet_status > 0).to(torch::kFloat); // [B, num_bets]

        // Project to MODEL_DIM
        auto bet_embs = bet_embedding->forward(bet_feats); // [B, num_bets, MODEL_DIM]

        // Transpose embeddings
        auto bet_embs_transposed = bet_embs.transpose(0, 1); // [num_bets, B, MODEL_DIM]

        // Create bet attention mask
        auto bet_key_padding_mask = ~bet_masks.to(torch::kBool); // [B, num_bets]

        // Process through bet TransformerEncoder
        auto bet_encodings = bet_encoder->forward(bet_embs_transposed, /*src_mask=*/{}, /*src_key_padding_mask=*/bet_key_padding_mask); // [num_bets, B, MODEL_DIM]

        // Transpose back
        bet_encodings = bet_encodings.transpose(0, 1); // [B, num_bets, MODEL_DIM]

        // Aggregate bet encodings
        auto bet_masks_expanded = bet_masks.unsqueeze(2); // [B, num_bets, 1]
        auto y = (bet_encodings * bet_masks_expanded).sum(1) / bet_masks_expanded.sum(1).clamp_min(1e-9); // [B, MODEL_DIM]

        // Combined Trunk
        auto z = torch::cat({x, y}, /*dim=*/1); // [B, 2 * MODEL_DIM]
        z = torch::gelu(comb1->forward(z));
        z = torch::gelu(comb2->forward(z) + z); // Residual connection
        z = torch::gelu(comb3->forward(z) + z); // Residual connection
        z = norm->forward(z);                   // LayerNorm

        // Action Head
        auto output = action_head->forward(z); // [B, NUM_ACTIONS]
        return output;
    }
};
TORCH_MODULE(DeepCFRModel); // Creates DeepCFRModel as a ModuleHolder<DeepCFRModelImpl>

#endif // MODEL_H
