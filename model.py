# from appendix C of https://arxiv.org/pdf/1811.00164 

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
        embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid.unsqueeze(1) # zero out 'no card' embeddings
        # sum across the cards in the hole/board
        return embs.view(B, num_cards, -1).sum(1)

class DeepCFRModel(nn.Module):
    def __init__(self, n_card_types, n_players, n_bets, n_actions, dim=256):
        super(DeepCFRModel, self).__init__()
        self.card_embeddings = nn.ModuleList(
            [CardEmbedding(dim) for _ in range(n_card_types)])
        self.card1 = nn.Linear(dim * n_card_types, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        nrounds = 4
        self.bet1 = nn.Linear((n_bets * n_players * nrounds) * 2, dim)
        self.bet2 = nn.Linear(dim, dim)

        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, n_actions)

    def forward(self, cards, bet_fracs, bet_status):
        """
        cards: ((N x 2), (N x 3)[, (N x 1), (N x 1)]) # (hole, board, [turn, river])
        bet_fracs: N x n_bet_feats
        bet_status: N x n_bet_status
        """
        # 1. card branch
        # embed hole, flop, and optionally turn and river
        card_embs = []
        for embedding, card_group in zip(self.card_embeddings, cards):
            card_embs.append(embedding(card_group))
        card_embs = torch.cat(card_embs, dim=1)
        x = F.relu(self.card1(card_embs))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        # 1. bet branch
        bet_size = bet_fracs.clamp(0, 1e6)
        bet_occurred = bet_status
        bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1)
        y = F.relu(self.bet1(bet_feats))
        y = F.relu(self.bet2(y) + y)

        # 3. combined trunk
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)
        z = self.norm(z) # (z - mean) / std
        return self.action_head(z)

if __name__ == '__main__':    
    import time

    # First, include the DeepCFRModel and CardEmbedding classes here
    # (The code from your provided PyTorch model)

    # Benchmark parameters
    n_card_types = 4
    n_players = 2
    n_bets = 3
    n_actions = 6
    batch_size = 1
    n_iterations = 1000

    # Initialize the model
    model = DeepCFRModel(n_card_types, n_players, n_bets, n_actions)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy input data
    cards = [
        torch.randint(0, 52, (batch_size, 2), device=device),
        torch.randint(0, 52, (batch_size, 3), device=device),
        torch.randint(0, 52, (batch_size, 1), device=device),
        torch.randint(0, 52, (batch_size, 1), device=device)
    ]
    bet_fracs = torch.rand((batch_size, (n_bets * n_players * 4)), device=device)
    bet_status = torch.randint(0, 2, (batch_size, (n_bets * n_players * 4)), device=device)

    # Warm-up run
    with torch.no_grad():
        _ = model(cards, bet_fracs, bet_status)

    # Benchmark
    start_time = time.time()

    for i in range(n_iterations):
        if i > 0:
            start_time = time.time()
        with torch.no_grad():
            output = model(cards, bet_fracs, bet_status)
        # Ensure computation is done
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    end_time = time.time()

    total_time = end_time - start_time
    average_time = total_time / n_iterations

    print(f"Total time for {n_iterations} iterations: {total_time:.4f} seconds")
    print(f"Average time per iteration: {average_time*1000*1000*1000} ns")