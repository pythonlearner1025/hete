# local best response algo
# https://arxiv.org/pdf/1612.07547
from treys import Evaluator, Card, Deck
from itertools import combinations
from util import cards2int, int2cards, Pubset, prepare_infoset
from math import comb
from tqdm import tqdm
from copy import deepcopy

import time
import ompeval
import torch
import numpy as np
import torch.nn.functional as F


def construct_card_lookup(int2cards):
    lookup = []
    for idx in range(1326):
        # Calculate the first card index
        i = 0
        while comb(51 - i, 1) <= idx:
            idx -= comb(51 - i, 1)
            i += 1
        # Calculate the second card index
        j = i + 1 + idx
        lookup.append([int2cards[i], int2cards[j]])
    return lookup

def construct_hands_lookup(opp_range, card_lookup_table):
    assert len(opp_range) == 1326
    lookup = list() 
    for i in range(len(opp_range)):
        hand = card_lookup_table[i]
        lookup.append(hand)
    return lookup

lbr = ompeval.LBR()

def batch_regret_match(logits):
    '''
        logits: B x actions
    '''
    logits = F.relu(logits)
    logits_sum = logits.sum(dim=0)  # Sum along the row dimension
    if logits_sum > 0:
        return (logits / (torch.full_like(logits, logits_sum) - logits))
    else:
        max_index = torch.argmax(logits)
        one_hot = torch.zeros_like(logits)
        one_hot[max_index] = 1
        return one_hot

# get board state when this action is chosen
def get_board_at_action(pubset: Pubset, actidx):
    n_bets_made = actidx*2
    n_rounds =  n_bets_made // (pubset.n_players * pubset.max_bets)
    if n_rounds == 0: 
        return []
    elif n_rounds == 1:
        return pubset.deck[:3]
    elif n_rounds == 2:
        return pubset.deck[:4]
    else:
        return pubset.deck

# get sequence of action indices in order
def get_act_indices(bet_fracs, bet_status, player_idx):
    pass

def wprollout(player_hand, opp_range, board, deck_cards):
    player_hand = [cards2int[c] for c in player_hand]
    opp_range = opp_range.tolist()
    board = [cards2int[c] for c in board]
    deck_cards = [cards2int[c] for c in deck_cards]
    s = time.time()
    # ~440 milllion eval/sec, x220000 speedup from python v above
    wp = lbr.wprollout(player_hand, opp_range, board, deck_cards)
    e = time.time()
    print(f'wprollout took {e-s} s')
    print(wp)
    return wp

# calc pot based on bet history & starting stacsk
def calc_pot_stack(
    starting_stacks, 
    bet_fracs, 
    player_idx, 
    n_players
    ):
    stack = starting_stacks[player_idx]
    amt = 0
    for i in range(len(bet_fracs)):
        if (i+1) % n_players == player_idx:
            bet_amt = bet_fracs[i] * stack
            stack -= bet_amt
            amt += bet_amt
    return amt, stack

def get_lbr_act(
    policy, 
    player_hand, 
    player_idx,
    opp_idx,
    opp_range, 
    pubset: Pubset, 
    deck_cards,
    max_bets_per_player=3
    ):
    n_players = pubset.n_players
    bet_fracs = pubset.bet_fracs
    bet_status = pubset.bet_status
    starting_stacks = pubset.stacks

    # get expected showdown win percentage
    card_lookup_table = construct_card_lookup(int2cards)
    hands_lookup_table = construct_hands_lookup(opp_range, card_lookup_table)

    # update range based on action history
    if bet_fracs:
        for i, actidx in enumerate(list(range(opp_idx, n_players, len(bet_fracs)))):
            for h in range(len(opp_range)):    
                prob, opp_hand = opp_range[h], hands_lookup_table[h]
                infoset = prepare_infoset(
                    opp_hand, 
                    get_board_at_action(pubset, i),
                    max_bets_per_player,
                    bet_fracs[:i],
                    bet_status[:i]
                )
                logits = policy(*infoset)
                logits_sum = logits.sum().item()
                if logits_sum > 0: 
                    regrets = (logits / (torch.full_like(logits, logits_sum) - logits))
                else:
                    max_idx = torch.argmax(logits)
                    regrets = torch.zeros(logits.shape)
                    regrets[max_idx] = 1
                regrets = regrets.squeeze()
                opp_range[h] = prob * regrets[actidx]
    
    if len(pubset.board)>0:
        for hand in pubset.board:
            hand = cards2int[hand]
            c = 0
            for i in range(52):
                for j in range(i+1,52):
                    if i == hand or j == hand:
                        opp_range[c] = 0
                    c+=1
        
        # Count the number of zeros in opp_range
        num_zeros = sum(1 for prob in opp_range if prob == 0)

        # Calculate the new total card value
        new_total = len(opp_range) - num_zeros
        
        # Set all non-zero values to 1/(new_total)
        new_prob = 1 / new_total if new_total > 0 else 0
        for i in range(len(opp_range)):
            if opp_range[i] != 0:
                opp_range[i] = new_prob
    
    # ~440 milllion eval/sec, x220000 speedup from python v above
    wp = wprollout(player_hand, opp_range, pubset.board, deck_cards)

    player_pot, player_stack = calc_pot_stack(
        starting_stacks, 
        bet_fracs, 
        player_idx,
        n_players
    )

    opp_pot, opp_stack = calc_pot_stack(
        starting_stacks, 
        bet_fracs, 
        opp_idx,
        n_players
    )

    pubset.stacks = [player_stack, opp_stack]
    
    # calc expected call utility
    total_pot = opp_pot + player_pot
    asked = opp_pot - player_pot
    call_util = wp*(player_pot+opp_pot)-(1-wp)*asked
    foldidx = 0 

    # add fold, call actions
    action_utils = [0, call_util]

    # add all other actions
    for act in tqdm(list(pubset.act2name.keys())[2:]):
        # skip fold and call cases
        if act <= 1: continue 
        fp = 0

        # counterfactual pubset
        bet_frac = float(pubset.act2name[act][5:8])
        bet_amt = bet_frac * pubset.stacks[opp_idx]
        cf_opp_range = np.copy(opp_range)
        cf_board = deepcopy(pubset.board)
        cf_bet_fracs = deepcopy(pubset.bet_fracs)
        cf_bet_status = deepcopy(pubset.bet_status)
        cf_bet_fracs.append(bet_amt)
        cf_bet_status.append(1)

        # Batch inference
        batch_size = len(cf_opp_range)
        batch_cards = []
        batch_bet_fracs = []
        batch_bet_status = []

        for h1 in range(batch_size):
            opp_hand = hands_lookup_table[h1]
            cards, bet_fracs, bet_status = prepare_infoset(
                opp_hand, 
                cf_board,
                max_bets_per_player,
                cf_bet_fracs,
                cf_bet_status 
            )
            batch_cards.append(cards)
            batch_bet_fracs.append(bet_fracs)
            batch_bet_status.append(bet_status)
        # Combine batches
        batch_cards = [torch.cat([batch[i] for batch in batch_cards]) for i in range(4)]
        batch_bet_fracs = torch.cat(batch_bet_fracs)
        batch_bet_status = torch.cat(batch_bet_status)

        # Single policy call
        batch_logits = policy(batch_cards, batch_bet_fracs, batch_bet_status)

        for h1 in range(batch_size):
            prob = cf_opp_range[h1]
            regrets = batch_logits[h1]
            regrets_sum = regrets.sum().item()
            if regrets_sum > 0:
                regrets = (regrets / (torch.full_like(regrets, regrets_sum) - regrets))
            else:
                max_index = torch.argmax(regrets)
                one_hot = torch.zeros_like(regrets)
                one_hot[max_index] = 1
            fp += prob * regrets[foldidx]
            # update prob of being in all other hand
            cf_opp_range[h1] = prob * (1-regrets[foldidx])

        cf_opp_range /= cf_opp_range.sum()

        wp = wprollout(player_hand, cf_opp_range, pubset.board, deck_cards)

        action_util = fp * total_pot + (1-fp) * (wp * (total_pot + bet_amt)) \
            - (1-wp) * (asked + bet_amt) 
        action_utils.append(action_util) 

    # if max act util is greater than 0, take that act. else fold
    if max(action_utils) > 0:
        return action_utils.index(max(action_utils))
    else:
        return 0

def init_opp_range(my_hand):
    opp_range = np.full(1326, 1/1298)

    hand1 = cards2int[my_hand[0]]
    hand2 = cards2int[my_hand[1]]
    pos = hand1 * hand2

    c = 0
    nc = 0
    for i in range(52):
        for j in range(i+1,52):
            if c == pos:
                opp_range[pos] = 0
            elif i == hand1:
                opp_range[c] = 0
            elif j == hand2:
                opp_range[c] = 0
            else:
                nc+=1
            c+=1
    # Renormalize the range
    opp_range /= opp_range.sum()
    return opp_range

if __name__ == '__main__':
    # calc lbr for p2
    pass 



