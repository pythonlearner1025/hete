import torch
import torch.nn as nn
import torch.nn.functional as F

class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self, input):
        B, num_cards = input.shape
        x = input.view(-1)
        valid = x.ge(0).float() # -1 means 'no card'
        x = x.clamp(min=0)
        x = x.to(torch.int64)
        rank_indices = torch.div(x, 4, rounding_mode='floor').clamp(0, 12)
        suit_indices = torch.remainder(x, 4).clamp(0, 3)
        embs = self.card(x) + self.rank(rank_indices) + self.suit(suit_indices)
        embs = embs * valid.unsqueeze(1) # zero out 'no card' embeddings
        return embs.view(B, num_cards, -1).sum(1)

class DeepCFRModel(nn.Module):
    def __init__(self, model_dim, num_actions, max_round_bets, num_players):
        super(DeepCFRModel, self).__init__()
        self.model_dim = model_dim
        self.num_actions = num_actions
        self.max_round_bets = max_round_bets
        self.num_players = num_players

        n_card_types = 4
        self.card_embeddings = nn.ModuleList(
            [CardEmbedding(model_dim) for _ in range(n_card_types)])
        self.card1 = nn.Linear(model_dim * n_card_types, model_dim)
        self.card2 = nn.Linear(model_dim, model_dim)
        self.card3 = nn.Linear(model_dim, model_dim)

        bet_input_size = (max_round_bets * num_players * 4) * 2
        self.bet1 = nn.Linear(bet_input_size, model_dim)
        self.bet2 = nn.Linear(model_dim, model_dim)

        self.comb1 = nn.Linear(model_dim * 2, model_dim)  # Added missing comb1 layer
        self.comb2 = nn.Linear(model_dim, model_dim)
        self.comb3 = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.action_head = nn.Linear(model_dim, num_actions)

    def forward(self, hand, flop, turn, river, bet_fracs, bet_status):
        # 1. card branch
        card_embs = []
        for embedding, card_group in zip(self.card_embeddings, [hand, flop, turn, river]):
            card_embs.append(embedding(card_group))
        card_embs = torch.cat(card_embs, dim=1)
        x = F.relu(self.card1(card_embs))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        # 2. bet branch
        bet_size = bet_fracs.clamp(0, 1e6)
        bet_occurred = bet_status.float()
        bet_feats = torch.cat([bet_size, bet_occurred], dim=1)
        y = F.relu(self.bet1(bet_feats))
        y = F.relu(self.bet2(y) + y)

        # 3. combined trunk
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)
        z = self.norm(z)
        return self.action_head(z)

# Example usage:
# model = DeepCFRModel(model_dim=256, num_actions=6, max_round_bets=3, num_players=2)