from pokerkit.pokerkit import Card
from typing import List, Dict, Union

import torch

class Pubset:
    def __init__(self, starting_stacks: List[float], bet_fracs: List[float], bet_status: List[int], board: List[str], max_bets: int, n_rounds: int, n_players: int, act2name: Dict[int, str]):
        self.stacks = starting_stacks
        self.bet_fracs = bet_fracs
        self.bet_status = bet_status
        self.board = board
        self.max_bets = max_bets
        self.n_rounds = n_rounds
        self.n_players = n_players
        self.act2name = act2name

# convert PokerKit Card to str rep
def card_to_string(card: Card) -> str:
    """Convert a Card object to its string representation."""
    rank_str = card.rank.value
    suit_str = card.suit.value
    
    # Special case for 10, which is represented as 'T' in the all_cards set
    if rank_str == '10':
        rank_str = 'T'
    
    return f"{rank_str}{suit_str}"

# Define dictionaries for RANK and SUIT mappings
RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

def card2int(S: str) -> int:
    """
    Convert a card string to its integer representation.
    
    Args:
    S (str): A two-character string representing a card (e.g., '2s', 'Th', 'Ac')
    
    Returns:
    int: An integer between 0 and 51 representing the card
    """
    rank = RANK_MAP[S[0]]
    suit = SUIT_MAP[S[1]]
    return 4 * rank + suit

# enumerate all 52 possible poker cards
# CardIdx is an integer between 0 and 51, where CARD = 4 * RANK + SUIT
# rank ranges from 0 (deuce) to 12 (ace) and suit is from 0 (spade) to 3 (club)
all_cards = {'2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
             '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s',
             '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Tc', 'Td', 'Th', 'Ts',
             'Jc', 'Jd', 'Jh', 'Js', 'Qc', 'Qd', 'Qh', 'Qs', 'Kc', 'Kd', 'Kh', 'Ks',
             'Ac', 'Ad', 'Ah', 'As'}

cards2int = {card: card2int(card) for card in all_cards}
int2cards = {card2int(card): card for card in all_cards}

def pad_by(round_bet_history, amt, padding=-1):
    return round_bet_history + [padding] * amt

def card2int(card: Union[Card, str, None]):
    if card is None:
        return -1
        
    """
    Convert a card string (e.g., 'Qc', '6s', 'Jh') to integer representations.
    
    Returns:
    - rank: 0-12 (2-A)
    - suit: 0-3 (c, d, h, s)
    - card: 0-51 (unique index for each card)
    """
    if isinstance(card, Card):
        card_str = card.rank + card.suit
    else:
        card_str = card
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    
    rank = ranks.index(card_str[0])
    suit = suits.index(card_str[1])
    card = rank * 4 + suit
    
    return card

def prepare_infoset(
    hand, 
    board, 
    max_bets_per_player, 
    round_bet_fracs, 
    round_bet_status, 
    padding=0,
    n_rounds=4,
    n_players=2
    ):
    '''
        returns:
            cards: ((1 x 2), (1 x 3)[, (1 x 1), (1 x 1)]) # (hole, board, [turn, river])
            bet_fracs: 1 x n_bet_feats
            bet_status: 1 x n_bet_status
    '''
    nbets = n_players * max_bets_per_player * n_rounds - n_rounds

    # Prepare cards
    hole_cards = torch.tensor([card2int(c) for c in hand]).unsqueeze(0)  # (1, 2)
    flop = torch.tensor([card2int(c[0]) for c in board[:3]] if len(board) >= 3 else [-1]*3).unsqueeze(0)  # (1, 3)
    turn = torch.tensor([card2int(board[3][0])] if len(board) > 3 else [-1]).unsqueeze(0)  # (1, 1)
    river = torch.tensor([card2int(board[4][0])] if len(board) > 4 else [-1]).unsqueeze(0)  # (1, 1)

    cards = [hole_cards, flop, turn, river]  # List[int]

    # Prepare bets
    #print(len(round_bet_fracs))
    #print(len(round_bet_fracs))
    betting_fracs = pad_by(round_bet_fracs, nbets-len(round_bet_fracs), padding=padding)
    betting_status = pad_by(round_bet_status, nbets-len(round_bet_status), padding=padding)
    
    bet_fracs = torch.tensor(betting_fracs, dtype=torch.float).unsqueeze(0)  # N x (2 * nbets)
    bet_status = torch.tensor(betting_status, dtype=torch.float).unsqueeze(0)  # N x (2 * nbets)

    return cards, bet_fracs, bet_status

