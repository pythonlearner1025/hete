import mlx.core as mx
import mlx.nn as nn

class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def __call__(self, input):
        B, num_cards = input.shape
        x = input.reshape(-1)
        valid = mx.where(x >= 0, 1.0, 0.0)  # -1 means 'no card'
        x = mx.clip(x, 0, None)
        embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid[:, None]  # zero out 'no card' embeddings
        # sum across the cards in the hole/board
        return embs.reshape(B, num_cards, -1).sum(1)

class DeepCFRModel(nn.Module):
    def __init__(self, n_card_types, n_players, n_bets, n_actions, dim=256):
        super().__init__()
        self.card_embeddings = [CardEmbedding(dim) for _ in range(n_card_types)]
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

    def __call__(self, cards, bet_fracs, bet_status):
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
        card_embs = mx.concatenate(card_embs, axis=1)
        x = nn.relu(self.card1(card_embs))
        x = nn.relu(self.card2(x))
        x = nn.relu(self.card3(x))

        # 2. bet branch
        bet_size = mx.clip(bet_fracs, 0, 1e6)
        bet_occurred = bet_status
        bet_feats = mx.concatenate([bet_size, bet_occurred.astype(mx.float32)], axis=1)
        y = nn.relu(self.bet1(bet_feats))
        y = nn.relu(self.bet2(y) + y)

        # 3. combined trunk
        z = mx.concatenate([x, y], axis=1)
        z = nn.relu(self.comb1(z))
        z = nn.relu(self.comb2(z) + z)
        z = nn.relu(self.comb3(z) + z)
        z = self.norm(z)  # (z - mean) / std
        return self.action_head(z)

if __name__ == '__main__': 
    import time

    # First, include the DeepCFRModel and CardEmbedding classes here
    # (The code from the previous response)

    # Benchmark parameters
    n_card_types = 4
    n_players = 2
    n_bets = 3
    n_actions = 6
    batch_size = 1
    n_iterations = 1000

    # Initialize the model
    model = DeepCFRModel(n_card_types, n_players, n_bets, n_actions)

    # Create dummy input data
    cards = [
        mx.random.randint(0, 52, (batch_size, 2)),
        mx.random.randint(0, 52, (batch_size, 3)),
        mx.random.randint(0, 52, (batch_size, 1)),
        mx.random.randint(0, 52, (batch_size, 1))
    ]
    bet_fracs = mx.random.uniform(shape=(batch_size, (n_bets * n_players * 4)))
    bet_status = mx.random.randint(0, 2, shape=(batch_size, (n_bets * n_players * 4)))

    # Warm-up run
    _ = model(cards, bet_fracs, bet_status)

    # Benchmark
    start_time_1 = time.time()

    for i in range(n_iterations):
        if i > 0:
            start_time_2 = time.time()

        output = model(cards, bet_fracs, bet_status)
        # Ensure computation is done
        mx.eval(output)

    end_time = time.time()

    total_time = end_time - start_time_1
    total_time = end_time - start_time_2
    average_time = total_time / (n_iterations-1)

    print(f"Total time for {n_iterations-1} iterations: {total_time:.4f} seconds")
    print(f"Average time per iteration: {average_time*1000*1000*1000} ns")