from engine import (
    get_new_game,
    is_terminal
)
from pokerkit.pokerkit import Card
from pokerkit.pokerkit import State
from copy import deepcopy
from typing import Dict, List, Union, Tuple
from model import DeepCFRModel

import random
import torch
import torch.nn.functional as F

act2name = {
    0: 'fold',
    1: 'check/call',
    2: 'cbr 0.25 stack',
    3: 'cbr 0.5 stack',
    4: 'cbr 0.75 stack',
    5: 'cbr 1.0 stack'
}

def card2int(card : Card):
    """
    Convert a card string (e.g., 'Qc', '6s', 'Jh') to integer representations.
    
    Returns:
    - rank: 0-12 (2-A)
    - suit: 0-3 (c, d, h, s)
    - card: 0-51 (unique index for each card)
    """
    card_str = card.rank + card.suit
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    
    rank = ranks.index(card_str[0])
    suit = suits.index(card_str[1])
    card = rank * 4 + suit
    
    return card

def pad_by(round_bet_history, amt, padding=-1):
    return round_bet_history + [padding] * amt

def mbb(amt, bb):
    return amt/(bb/1000)

def take_action(game: State, action_index):
    if action_index == 0:
        game.fold()
    elif action_index == 1:
        if game.can_check_or_call():
            game.check_or_call()
        else:
            raise NotImplementedError
    else:
        actor = game.actor_index 
        stack = game.stacks[actor]
        bet_amt = stack * float(act2name[action_index][5:8])
        game.complete_bet_or_raise_to(bet_amt)

def verify_action(game: State, action_index):
    #print('actor', game.actor_index)
    #print('bets:', game.bets)
    #print('stacks:', game.stacks)
    #print("verifying bet")
    if action_index == 0:
        try:
            game.verify_folding()
        except Exception as e:
            #print(e)
            return False
    elif action_index == 1:
        try:
            game.verify_checking_or_calling()
        except Exception as e:
            #print(e)
            return False
    else:
        actor = game.actor_index 
        stack = game.stacks[actor]
        bet_amt = stack * float(act2name[action_index][5:8])
        #print(stack, float(act2name[action_index][5:8]))
        try:
            game.verify_completion_betting_or_raising_to(bet_amt)
        except Exception as e:
            #print(e)
            return False
    #print("bet is OK")
    return True

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

def get_round(n_board_cards):
    if n_board_cards == 0:
        return 0 
    elif n_board_cards == 3:
        return 1
    elif n_board_cards == 4:
        return 2
    else:
        return 3

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

def traverse(
    game : State, 
    p : int,
    traverse_advs : List[any], 
    nets : Dict[int, List[DeepCFRModel]], 
    t : int,
    max_bets_per_player : int,
    round_bet_fracs=[], 
    round_bet_status=[],
    ):
    '''
    print("-- traverse --")
    print(f"actor: {game.actor_index}")
    print(f'p: {p}')
    print(f'bets: {game.bets}')
    print(f'stacks: {game.stacks}')
    print(f'status: {game.statuses}')
    '''
    n_players = len(game.bets)
    bb_frac = game.blinds_or_straddles[1]
    # wait till game is terminal for all
    if game.actor_index is None or sum(game.statuses) == 1:
        #print("GAME OVER")
        amt = mbb(game.payoffs[p], bb_frac) # in mbb
        #print(game.payoffs)
        #print('mbbs:',amt)
        return amt
    elif game.actor_index == p:
        net = nets[p][t-1]
        hand = game.hole_cards[p]
        board = game.board_cards

        I = prepare_infoset(
            hand, 
            board, 
            max_bets_per_player, 
            round_bet_fracs, 
            round_bet_status
        ) 

        logits = net(*I)
        # TODO regret matching
        strat = regret_match(logits)
        acts = set(act2name.keys()) 
        values = [0] * len(acts)
        advs = [0] * len(acts)
        illegal_acts = set()
        for i,a in enumerate(acts):
            # update round_bet_fracs, round_bet_status, round_bet_counter
            new_game = deepcopy(game)
            new_round_bet_fracs = deepcopy(round_bet_fracs)
            new_round_bet_status = deepcopy(round_bet_status)

            # if calling is required
            action_index = a
            if action_index > 1 and \
                (len(round_bet_status)%(max_bets_per_player*n_players) >= (max_bets_per_player-1)*n_players): 
                action_index = 1

            if not verify_action(new_game, action_index):
                illegal_acts.add(action_index)
                #print("skip")
                continue
            
            take_action(new_game, action_index)
            # so check == 0 & 1, fold == 0 & 0 and can thus be distinguished by net
            # check if checking is possible
            new_round_bet_status.append(1 if action_index >= 1 else 0)
            # get bet amt
            if action_index == 0:
                bet_amt = 0
            elif action_index == 1: 
                bet_amt = game.bets[p-1]
            else:
                stack = game.stacks[p]
                bet_amt = stack * float(act2name[action_index][5:8]) 
            new_round_bet_fracs.append(bet_amt)
            #print(f"iter {i}")
            values[a] = traverse(
                new_game, 
                p, 
                traverse_advs, 
                nets, 
                t,
                max_bets_per_player,
                round_bet_fracs=new_round_bet_fracs,
                round_bet_status=new_round_bet_status
            )

        for a in (acts-illegal_acts):
            # relative advs of a to expected value of playingo by strat probs
            #print(strat.shape)
            adv = values[a] - sum([strat[na] * values[na] for na in (acts - {a} - illegal_acts)])
            advs[a] = adv if adv > 0 else 1/len(acts)

        traverse_advs.append((I,t,advs))

        # pass up expected value of this decision point
        return sum([strat[a] * values[a] for a in acts - illegal_acts])
    else:
        actor = game.actor_index
        net = nets[actor][t-1] 
        hand = game.hole_cards[actor]
        board = game.board_cards

        I = prepare_infoset(
            hand, 
            board, 
            max_bets_per_player, 
            round_bet_fracs, 
            round_bet_status
        ) 
        # TODO regret matching
        logits = net(*I)
        strat = regret_match(logits)
        try:
            # Replace inf with 1 in the specific case mentioned
            strat_fixed = torch.where(torch.isinf(strat), torch.tensor(1.0), strat)
            action_index = torch.multinomial(strat_fixed, num_samples=1).item()
        except Exception as e:
            print("Original logits:", logits)
            print("Original strat:", strat)
            print("Fixed strat:", strat_fixed)
            raise e

        while not verify_action(game, action_index):
            action_index = (action_index + 1) % logits.shape[-1]

        if action_index > 1 and \
            (len(round_bet_status) % max_bets_per_player*n_players >= max_bets_per_player*n_players): 
            action_index = 1

        take_action(game, action_index)

        # check if checking is possible
        round_bet_status.append(1 if action_index >= 1 else 0)
        # get bet frac
        if action_index == 0:
            bet_frac = 0
        elif action_index == 1: 
            if game.stacks[actor-1] + game.bets[actor-1] != 0:
                bet_frac = game.bets[actor-1] / (game.stacks[actor-1] + game.bets[actor-1])
            else:
                bet_frac = 0
        else:
            bet_frac = float(act2name[action_index][5:8]) 

        round_bet_fracs.append(bet_frac)

        return traverse(
            game, 
            p, 
            traverse_advs, 
            nets, 
            t,
            max_bets_per_player,
            round_bet_fracs=round_bet_fracs,
            round_bet_status=round_bet_status
        )

def batch_loader(all_player_advs: List[Tuple], batch_size: int):
    random.shuffle(all_player_advs)
    n = len(all_player_advs)
    for i in range(0, n, batch_size):
        batch = all_player_advs[i:min(i+batch_size, n)]
        
        cards_batch = [item[0][0] for item in batch]
        bet_fracs_batch = [item[0][1] for item in batch]
        bet_status_batch = [item[0][2] for item in batch]
        t_batch = [torch.tensor(item[1]) for item in batch]
        advs_batch = [item[2] for item in batch]

        hands = []
        flops = []
        turns = []
        rivers = []
        for hand,flop,turn,river in cards_batch:
            hands.append(hand)
            flops.append(flop)
            turns.append(turn)
            rivers.append(river)


        batched_cards = [
            torch.vstack(hands), torch.vstack(flops), 
            torch.vstack(turns), torch.vstack(rivers)
        ]
        batched_t = torch.stack(t_batch)
        batched_bet_fracs = torch.stack(bet_fracs_batch)
        batched_bet_status = torch.stack(bet_status_batch)

        if len(batched_bet_fracs.shape) > 2:
            batched_bet_fracs = batched_bet_fracs.squeeze(1)
            batched_bet_status = batched_bet_status.squeeze(1)
        elif len(batched_bet_fracs.shape) == 1:
            batched_bet_fracs = batched_bet_fracs.unsqueeze(0)
            batched_bet_status = batched_bet_status.unsqueeze(0) 
        

        batched_advs = []
        for advs in advs_batch:
            adv_tensor = torch.tensor(advs)
            batched_advs.append(adv_tensor)
        batched_advs = torch.stack(batched_advs)

        yield batched_cards, batched_bet_fracs, batched_bet_status, batched_advs, batched_t

if __name__ == '__main__':
    # perf/exploitability use unit of 1/1000 bb

    H = dict() # all seen infos
    # reservoir-sampled adv memories 
    bb = 2
    n_players = 2
    cfr_iters = 180
    traversals = 100000
    max_bets_per_player = 6
    batch_size = 10000  # or any other suitable batch size
    max_train_iter = 4000
    max_advs_size = 40*10e6

    all_advs = {
        p:list()
        for p in range(n_players)
    }
    # save as many v networks as cfr_iters
    all_nets = {
        p:[
            DeepCFRModel(
                n_card_types=4, 
                n_players=n_players, 
                n_actions=len(act2name.keys()),
                n_bets=max_bets_per_player
            )
            ]
        for p in range(n_players)
    }

    total_advs_c = 0

    for t in range(1,cfr_iters+1):
        for p in range(n_players):
            p = ((n_players-1)+p) % n_players
            for k in range(traversals):
                print(f't: {t}, p: {p}, k: {k}, infosets: {len(all_advs[p])}')
                traverse_advs = []
                game = get_new_game(n_players, bb=bb)
                traverse(game, p, traverse_advs, all_nets, t, max_bets_per_player)

                # reservoir sampling if buffer cap exceeded
                if len(all_advs[p]) >= max_advs_size:
                    for adv in traverse_advs:
                        if len(all_advs[p]) < max_advs_size:
                            all_advs[p].append(adv)
                        else:
                            j = random.randint(0, total_advs_c)
                            if j < max_advs_size:
                                all_advs[p][j] = adv
                else:
                    all_advs[p].extend(traverse_advs)

                total_advs_c += len(traverse_advs)

            # make sure to batch
            net = DeepCFRModel(
                n_card_types=4, 
                n_players=n_players, 
                n_actions=len(act2name.keys()),
                n_bets=max_bets_per_player
            )

            net = net.train()
            optimizer = torch.optim.Adam(net.parameters())
            # Use the batch_loader
            # is dataset player_advs or something else? 
            for i, (batched_cards, batched_bet_fracs, batched_bet_status, batched_advs, batched_t) in enumerate(batch_loader(all_advs[p], batch_size)):
                if i > max_train_iter:
                    break
                print(f'training iter {i}')
                outputs = net(batched_cards, batched_bet_fracs, batched_bet_status)
                
                # Compute MSE loss for each sample in the batch
                mse_loss = F.mse_loss(outputs, batched_advs, reduction='none')
                
                # Weight the loss by the respective t value
                weighted_loss = mse_loss * batched_t.unsqueeze(1)
                
                # Average the loss over all samples and actions
                loss = weighted_loss.mean()
                
                print(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            all_nets[p].append(net)


'''
    for free play
    indices = list(range(t + 1))
    probs = [i / sum(indices) for i in indices]
    sampled_idx = random.choices(indices, weights=probs, k=1)[0]
    p_net = all_nets[p][sampled_idx]
'''

# best response BR(strat) = max strat' payoff(strat', opp_strat)
# where payoff(strat', opp_strat) is the expected poyoff 
# playing according to strat' assuming opp plays deterministically following opp_strat

# exploitability is defined as difference between
# payoff(strat*, BR(strat*)) - payoff(strat, BR(strat))
# where strat* is the Nash equilibrium strategy
# the lower th ebetter 
