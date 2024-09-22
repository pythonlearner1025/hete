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
        valid = x.ge(0).float()
        x = x.clamp(min=0)
        embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid.unsqueeze(1)
        return embs.view(B, num_cards, -1).sum(1)

class DeepCFRModel(nn.Module):
    def __init__(self, n_card_types, n_players, n_round_bets, n_actions, dim=256):
        super(DeepCFRModel, self).__init__()
        self.card_embeddings = nn.ModuleList(
            [CardEmbedding(dim) for _ in range(n_card_types)])
        
        self.hand = CardEmbedding(dim)
        self.flop = CardEmbedding(dim)
        self.turn = CardEmbedding(dim)
        self.river = CardEmbedding(dim)
        
        self.card1 = nn.Linear(dim * n_card_types, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        self.bet1 = nn.Linear((n_round_bets * n_players * 4) * 2, dim)
        self.bet2 = nn.Linear(dim, dim)

        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, n_actions)

    def forward(self, hand, flop, turn, river, bet_fracs, bet_status):
        card_embs = [self.hand(hand), self.flop(flop), self.turn(turn), self.river(river)]
        card_embs = torch.cat(card_embs, dim=1)
        inplace = True
        x = F.relu(self.card1(card_embs), inplace=inplace)
        x = F.relu(self.card2(x), inplace=inplace)
        x = F.relu(self.card3(x), inplace=inplace)

        bet_occurred = bet_status
        bet_feats = torch.cat([bet_fracs, bet_occurred.float()], dim=1)
        y = F.relu(self.bet1(bet_feats), inplace=inplace)
        y = F.relu(self.bet2(y) + y, inplace=inplace)

        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z), inplace=inplace)
        z = F.relu(self.comb2(z) + z, inplace=inplace)
        z = F.relu(self.comb3(z) + z, inplace=inplace)
        z = self.norm(z)
        return self.action_head(z)

# Fixed constraints
n_actions = 5
n_card_types = 4
n_players = 2
max_round_bets = 3
n_bets = max_round_bets * n_players * 4

# Initialize the model
model = DeepCFRModel(n_card_types, n_players, max_round_bets, n_actions)

# Move model to GPU if available
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create dummy input data
batch_size = 1
hand = torch.randint(0, 52, (batch_size, 2), device=device)
flop = torch.randint(0, 52, (batch_size, 3), device=device)
turn = torch.randint(0, 52, (batch_size, 1), device=device)
river = torch.randint(0, 52, (batch_size, 1), device=device)
bet_fracs = torch.rand((batch_size, n_bets), device=device)
bet_status = torch.randint(0, 2, (batch_size, n_bets), device=device)

# JIT compile the model
torch.jit.set_fusion_strategy([("STATIC", 0), ("DYNAMIC", 0)])

scripted_model = torch.jit.script(model)

# Save the JIT compiled model
scripted_model.save("jit_compiled_model.pt")

# Test the JIT compiled model
with torch.no_grad():
    output = scripted_model(hand, flop, turn, river, bet_fracs, bet_status)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        output = scripted_model(hand, flop, turn, river, bet_fracs, bet_status)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("JIT compilation successful. Model saved as 'jit_compiled_model.pt'")
print("Output shape:", output.shape)