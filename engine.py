from pokerkit.pokerkit import Automation
from pokerkit.pokerkit import NoLimitTexasHoldem, UnfixedLimitHoldem
from pokerkit.pokerkit import Card

import numpy as np
import random

def get_new_game(n_players, bb=2, alpha=1.0):
    # https://pokerkit.readthedocs.io/en/latest/simulation.html#pre-defined-games
    automations = (
        Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.CARD_BURNING,
        Automation.HOLE_DEALING,
        Automation.BOARD_DEALING,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
    )
  # Define Dirichlet distribution parameters
    a = [alpha] * n_players  # You can adjust these values to control the distribution
    # Generate Dirichlet distribution
    dirichlet_dist = tuple(np.random.dirichlet(a).tolist())

    # Adjust the distribution to ensure minimum value is 0.15
    buy_in = bb*50
    min_value = 0.1
    adjusted_dist = np.maximum(dirichlet_dist, min_value)
    adjusted_dist = adjusted_dist / adjusted_dist.sum()  # Renormalize
    adjusted_dist *= buy_in
    adjusted_dist = adjusted_dist / adjusted_dist.sum() * buy_in # Renormalize to ensure sum is 100
    adjusted_dist = np.round(adjusted_dist)  # Round to nearest integer
    
    # Convert to tuple
    dirichlet_dist = tuple(adjusted_dist.tolist())
    with open('log', 'w') as f:
        f.write(f'{dirichlet_dist}')

    state = NoLimitTexasHoldem.create_state(
        automations,
        True,
        0,
        [bb/2, bb], # blinds
        1/1000, # min bet,
        dirichlet_dist,
        n_players
    )

    return state

def is_terminal(game, p):
    stats = game.statuses
    return not stats[p]

if __name__ == '__main__':
    bb_frac = 0.02
    game = get_new_game(3, bb_frac=bb_frac)
    print(type(game))
    print(game.min_completion_betting_or_raising_to_amount)
    print(game.stacks)
    print(game.hole_cards)
    print(game.bets)
    # blinds
    # p1 sb, p2 bb

    # preflop
    # p3 cc, p1 cc, p2 cbr
    # p3 cc, p1 cc,

    # flop
    # p2 cbr, p3 cbr, p1 cbr
    # p2 cbr, p3 cbr, p1 cbr,
    # ...  
    # p2 cc, p3 cc
    print(game.actor_index) # action is on p3
    game.check_or_call() # p3 cc
    print(game.actor_index)
    game.check_or_call() # p1 cc
    print(game.actor_index)
    # fork the game
    game.complete_bet_or_raise_to(0.03) # p2 cbr
    game.check_or_call() # p3 cc
    game.check_or_call() # p1 cc
    print("flop?")
    print(game.bets)
    # flop
    game.complete_bet_or_raise_to(0.01) #p1 cbr
    game.fold() # p2 fold
    game.check_or_call() #p3 checks
    game.complete_bet_or_raise_to(0.01)
    game.fold()
    print(game.bets)
    print(game.stacks)
    print(game.payoffs)
    mbbs = [payoff/(bb_frac/1000) for payoff in game.payoffs]
    print(mbbs)
