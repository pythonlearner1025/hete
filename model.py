import mlx.core as mx
import mlx.python.mlx.nn as nn
import mlx.python.mlx.optimizers as optim
from typing import Optional
from dataclasses import dataclass


import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    model_dim: int = 128
    num_layers: int = 1
    num_heads: int = 4
    num_players: int = 2
    num_actions: int = 6
    max_round_bets: int = 6
    head_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.model_dim // self.num_heads
    
    def update_from_dict(self, config_dict) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class PokerGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # embeddings
        setattr(self, "PokerGPT.card_emb_w", mx.random.normal((52, config.model_dim)))
        setattr(self, "PokerGPT.rank_emb_w", mx.random.normal((13, config.model_dim)))
        setattr(self, "PokerGPT.suit_emb_w", mx.random.normal((4, config.model_dim)))
        setattr(self, "PokerGPT.bet_proj_w", mx.random.normal((1, config.model_dim)))
        setattr(self, "PokerGPT.action_head", mx.random.normal((config.model_dim, config.num_actions)))
        
        # transformer layers
        for i in range(config.num_layers):
            # layernorm
            setattr(self, f"PokerGPT.layer_{i}.layernorm.gamma", mx.ones((config.model_dim,)))
            setattr(self, f"PokerGPT.layer_{i}.layernorm.beta", mx.zeros((config.model_dim,)))
            
            # attention heads
            for j in range(config.num_heads):
                prefix = f"PokerGPT.layer_{i}.head_{j}"
                setattr(self, f"{prefix}.attn_wq", mx.random.normal((config.model_dim, config.head_dim)))
                setattr(self, f"{prefix}.attn_wk", mx.random.normal((config.model_dim, config.head_dim)))
                setattr(self, f"{prefix}.attn_wv", mx.random.normal((config.model_dim, config.head_dim)))
                setattr(self, f"{prefix}.attn_out", mx.random.normal((config.model_dim, config.head_dim)))
                setattr(self, f"{prefix}.ffn_1", mx.random.normal((config.head_dim, 4*config.model_dim)))
                setattr(self, f"{prefix}.ffn_2", mx.random.normal((4*config.model_dim, config.head_dim)))

    def _apply_attention(self, x, layer_idx, head_idx):
        prefix = f"PokerGPT.layer_{layer_idx}.head_{head_idx}"
        
        q = mx.matmul(x, getattr(self, f"{prefix}.attn_wq"))
        k = mx.matmul(x, getattr(self, f"{prefix}.attn_wk"))
        v = mx.matmul(x, getattr(self, f"{prefix}.attn_wv"))

        attn = mx.matmul(q, k.transpose((0,2,1))) / mx.sqrt(self.config.head_dim)
        attn = mx.tril(attn)
        attn = mx.softmax(attn, axis=-1)

        x = mx.matmul(attn, v)
        x = mx.matmul(x, getattr(self, f"{prefix}.ffn_1"))
        x = mx.maximum(x, 0)
        x = mx.matmul(x, getattr(self, f"{prefix}.ffn_2"))
        
        return x

    def _apply_layer_norm(self, x, layer_idx):
        gamma = getattr(self, f"PokerGPT.layer_{layer_idx}.layernorm.gamma")
        beta = getattr(self, f"PokerGPT.layer_{layer_idx}.layernorm.beta")
        
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / mx.sqrt(var + 1e-5) + beta

    def __call__(self, cards, bets):
        BS = cards.shape[0]
        
        # embedding and positional stuff stays the same except for bet_proj access
        card_emb = self._embed_cards(cards)
        bets = mx.matmul(mx.reshape(bets, (BS, -1, 1)), getattr(self, "PokerGPT.bet_proj_w"))
        
        round_pe = self._positional_encoding(4)
        action_pe = self._positional_encoding(self.config.num_players * self.config.max_round_bets)
        
        preflop = mx.reshape(mx.repeat(round_pe[0:1], self.config.max_round_bets*self.config.num_players), (-1, self.config.model_dim))
        flop = mx.reshape(mx.repeat(round_pe[1:2], self.config.max_round_bets*self.config.num_players), (-1, self.config.model_dim))
        turn = mx.reshape(mx.repeat(round_pe[2:3], self.config.max_round_bets*self.config.num_players), (-1, self.config.model_dim))
        river = mx.reshape(mx.repeat(round_pe[3:4], self.config.max_round_bets*self.config.num_players), (-1, self.config.model_dim))
        
        concat_round_pos = mx.concatenate([preflop, flop, turn, river], axis=0)
        bets = bets + concat_round_pos + mx.repeat(action_pe, 4, axis=0)
        
        x = mx.concatenate([card_emb, bets], axis=1)

        # transformer layers
        for i in range(self.config.num_layers):
            head_outputs = []
            for j in range(self.config.num_heads):
                head_out = self._apply_attention(x, i, j)
                head_outputs.append(head_out)
            
            x = mx.concatenate(head_outputs, axis=-1)
            x = self._apply_layer_norm(x, i)
            
        last_tokens = x[:, -1:]
        return mx.matmul(last_tokens, getattr(self, "PokerGPT.action_head"))
    # rest of the implementation stays the same...
    def _embed_cards(self, x):
        B = x.shape[0]  # batch size 
        num_cards = x.shape[1]  # number of cards
        
        x = mx.reshape(x, (B * num_cards,))
        valid = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        x = mx.maximum(x, mx.zeros_like(x))
        x = x.astype(mx.int32)
        
        # get indices
        rank_indices = x // 4
        suit_indices = x % 4
        
        # gather embeddings
        card_embs = mx.take(getattr(self, "PokerGPT.card_emb_w"), x, axis=0) 
        rank_embs = mx.take(getattr(self, "PokerGPT.rank_emb_w"), rank_indices, axis=0)
        suit_embs = mx.take(getattr(self, "PokerGPT.suit_emb_w"), suit_indices, axis=0)
        
        # combine embeddings
        embs = card_embs + rank_embs + suit_embs
        embs = embs * mx.expand_dims(valid, -1)
        
        return mx.reshape(embs, (B, num_cards, self.config.model_dim))

    def _positional_encoding(self, length: int):
        dim = self.config.model_dim
        pe = mx.zeros((length, dim))
        
        for i in range(0, dim, 2):
            # compute angles using broadcasting
            angles = mx.arange(length)[:, None] * mx.exp(-(i/dim) * mx.log(mx.array(10000.0)))
            pe[:, i] = mx.sin(angles).squeeze()
            if i + 1 < dim:
                pe[:, i + 1] = mx.cos(angles).squeeze()
        
        return pe

def regret_match_mx(logits): 
    if isinstance(logits, list):
        logits = mx.array(logits)
    n_actions = len(logits)
    relu_logits = mx.maximum(logits, 0)
    logits_sum = mx.sum(relu_logits)
    
    if logits_sum > 0:
        strat = relu_logits / logits_sum
    else:
        max_index = mx.argmax(relu_logits)
        strat = mx.zeros(n_actions)
        strat = strat.at[max_index].set(1.0)
    print(strat)
    return strat

def train():
    config = ModelConfig()
    model = PokerGPT(config)
    opt = optim.Adam(learning_rate=0.001)
    
    hand = mx.array([[10, 10]], dtype=mx.int32)
    bets = mx.zeros((1, config.num_players*config.max_round_bets*4))
    targets = mx.zeros((1, 1, config.num_actions))
    
    def loss_fn(model, hand, bets, targets):
        preds = model(hand, bets)  # use __call__ not forward
        return mx.mean((preds - targets)**2)
    
    for step in range(1000):
        value, grads = mx.value_and_grad(loss_fn)(model, hand, bets, targets)
        opt.update(model, grads)
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {value}")

from safetensors import safe_open

def model_forward(model, hands, bets):
    hands = mx.reshape(mx.array(hands), (-1, len(hands)))
    bets = mx.reshape(mx.array(bets), (-1, len(bets)))
    logits = model(hands, bets)
    return logits[0][0].tolist()

def inference():
    config = ModelConfig()
    model = PokerGPT(config)
    path = '/Users/minjunes/hete/out/20241109075618/9/0/model.safetensors'
    with safe_open(path, framework="pt") as f:
        # gets metadata + tensor names without loading the tensors
        metadata = f.metadata()
        keys = f.keys()
    print("safetensor keys:")
    for k in keys:
        print(k)
    print("current model keys:")
    for k,v in model.parameters().items():
        print(k)
    model = model.load_weights(path)
    print("all params loaded")
    return

def train():
    config = ModelConfig()
    model = PokerGPT(config)
    opt = optim.Adam(learning_rate=0.001)
    
    hand = mx.array([[10, 10]], dtype=mx.int32)
    hand = mx.reshape(hand, (1, 2))
    bets = mx.zeros((1, config.num_players*config.max_round_bets*4))
    targets = mx.zeros((1, 1, config.num_actions))
    
    def loss_fn(params, hand, bets, targets):
        # temporarily update model with params
        model.update(params)
        preds = model(hand, bets)
        strats = regret_match_mx(preds)
        print(strats.tolist()[0][0])
        return mx.mean((preds - targets)**2)
    
    for step in range(1000):
        # get model parameters
        params = model.parameters()
        # compute loss and grads using parameters
        value, grads = mx.value_and_grad(loss_fn)(params, hand, bets, targets)
        
        # update using grads
        opt.update(model, grads)
        mx.eval(model.parameters())
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {value}")

if __name__ == "__main__":
    import wandb

    api = wandb.Api()
    runs = api.runs("")
    for i in runs:
        print("run name = ",i.name," id: ", i.id)

    #inference()
    #train()