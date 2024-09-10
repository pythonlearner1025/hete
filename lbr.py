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

'''
def wprollout(
    player_hand, 
    opp_range, 
    board, 
    deck_cards # all remaining cards, minus deck too if deck is not empty
    ):

    card_lookup_table = construct_card_lookup(int2cards)
    hands_lookup_table = construct_hands_lookup(opp_range, card_lookup_table)
    all_remaining_board_combo = list(combinations(deck_cards, 5-len(board)))
    total_won = 0
    for new_board in tqdm(all_remaining_board_combo):
        new_board = [cards2int[c] for c in new_board] + board
        my_weak = ompeval.evaluate_hand(new_board +  player_hand)
        won = 0
        for j in range(len(opp_range)):
            prob, opp_hand = opp_range[j], hands_lookup_table[j]
            if prob > 0 and not set(opp_hand).intersection(set(board)): 
                opp_weak = ompeval.evaluate_hand(board + player_hand)
                if my_weak < opp_weak: won += prob
        total_won += won * 1/len(all_remaining_board_combo) 
    return total_won
'''
lbr = ompeval.LBR()

def regret_match(logits):
    logits = logits[0]
    logits = F.relu(logits)
    logits_sum = logits.sum().item()  # Convert to scalar
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
    n_rounds =  n_bets_made // (pubset.n_player * pubset.max_bets)
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

    # update range based on action history
    if bet_fracs:
        for i, actidx in enumerate(list(range(opp_idx, n_players, len(bet_fracs)))):
            for h in range(len(opp_range)):    
                prob, opp_hand = opp_range[h]
                infoset = prepare_infoset(
                    opp_hand, 
                    get_board_at_action(pubset, i),
                    bet_fracs[:i],
                    bet_status[:i]
                )
                logits = policy(infoset)
                probs = regret_match(logits)
                opp_range[h] = prob * probs[actidx]
    
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
    
    # TODO why tf is kh, kc still in ?

    # get expected showdown win percentage
    card_lookup_table = construct_card_lookup(int2cards)
    hands_lookup_table = construct_hands_lookup(opp_range, card_lookup_table)

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
        cf_pubset = deepcopy(pubset)
        cf_pubset.bet_fracs.append(bet_amt)
        cf_pubset.bet_status.append(1)
        bet_amt = player_stack * cf_pubset.bet_fracs[-1]

        for h1 in range(len(opp_range)):
            prob, opp_hand = opp_range[h1], hands_lookup_table[h1]
            infoset = prepare_infoset(
                opp_hand, 
                cf_pubset.board,
                max_bets_per_player,
                cf_pubset.bet_fracs,
                cf_pubset.bet_status 
            )

            logits = regret_match(policy(*infoset))
            fp += prob * logits[foldidx]
            # update prob of being in all other hand
            for h2 in range(len(opp_range)):
                if h2 == h1: continue
                opp_range[h2] *= (1-logits[foldidx])

        # normalize
        opp_range /= sum(opp_range)

        # wprollout value is exact same, this ain't right
        wp = wprollout(player_hand, opp_range, pubset.board, deck_cards)

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



