import os
import random
from treys import Card
from treys import Evaluator

sys = open('sys').read()
prompt = open('prompt').read()

int2round = {
    0 : 'preflop',
    1 : 'preflop',
    2 : 'flop',
    3 : 'turn',
    4 : 'river'
}

def normalize(amt, bb):
    return [a/bb for a in amt]

def calc_stack_bets(stack, bets, traj):
    for a in traj:
        if 'cbr' in a: 
            aggressor = a[:2] 
            amt = float(a[7:])
            stack[aggressor] -= amt
            bets[aggressor] += amt

def find_sb_player(actions):
    for act in actions:
        if 'pb' in act:
            return act[:2]

# get the latest deck state
def get_board(actions):
    deals = [a for a in actions if a[2:4] == 'db'] 
    board = []
    for deal in deals:
        cards = deal.split(' ')[-1]
        cards = [cards[i:i+2] for i in range(0, len(cards), 2)]
        for card in cards:
            board.append(Card.new(card))
    #Card.print_pretty_cards(board)
    return board

def extract_winner(evaluator, board, actions):
    # 1. fold win
    if actions[-1][-1] == 'f':
        folded = []
        for act in actions:
            if ' f' in act:
                folded.append(int(act[1]))
        all_players = [i for i in range(1,len(folded)+2)]
        for out in folded:
            all_players.remove(out)
        assert len(all_players) == 1
        winner = all_players[0]
        return f'p{winner}'
    # 2. showdown
    contenders = []
    i = -1
    while actions[i][3:5] == 'sm':
        try:
            hand = [Card.new(actions[i][-4:-2]), Card.new(actions[i][-2:])]
        except:
            player = int(actions[i][:2][1])-1
            hand = actions[player][-4:]
            hand = [Card.new(hand[:2]), Card.new(hand[2:])]
        
        contenders.append((actions[i][:2], hand))
        i -=1 
    # sort by p1, p2, p3
    contenders = sorted(contenders, key=lambda x:x[0])
    print(contenders)
    ranks = [(p, evaluator.evaluate(board, h)) for p, h in contenders]
    print(ranks)
    winner = sorted(ranks, key=lambda x:x[1])[0]
    # this is for sanity check
    #evaluator.hand_summary(board, [c[1] for c in contenders])
    return winner[0]

def get_player_traj(actions, player, n_players):
    #print('get_player_trajs')
    #print(locals())
    #print(actions)
    split_actions = []
    i = 0
    buffer = []
    while i < len(actions):
        if actions[i][:2] != player:
            if i < n_players and i != int(player[-1])-1:
                censored = actions[i][:-4] + '????'
                buffer.append(censored)
            else:
                buffer.append(actions[i])
        else:
            if not any(['sm' in b for b in buffer]): 
                split_actions.append(buffer)
            buffer = []
            # player action is target
            if not 'sm' in actions[i]:
                masked_action = [actions[i]]
                split_actions.append(masked_action)
        i+=1
    return split_actions

from typing import List

def normalize_cbrs(trajs: List[List[str]], bb: int):
    for traj in trajs:
        for i, action in enumerate(traj):
            if 'cbr' in action:
                parts = action.split()
                amount = int(parts[-1])
                normalized_amount = amount / bb
                parts[-1] = f"{normalized_amount:.2f}"
                traj[i] = ' '.join(parts)

def prune(trajs, player):
    print(f'in prune, len traj = {len(trajs)}')
    for i in range(len(trajs)-1, -1, -1):
        for traj in trajs[i]:
            print(traj)
            if player in traj:
                return trajs[:i+1]

evaluator = Evaluator()

def make_dataset(folder, out_dir, winner_p=0.5):
    dataset = []
    for filename in os.listdir(folder):
        print(filename)
        if filename.endswith('.phh'):  # Assuming the data files have .phh extension
            try:
                file_path = os.path.join(folder, filename)

                with open(file_path, 'r') as file:
                    txt = file.read()

                start_stacks = [int(stack) for stack in txt.split('starting_stacks = [')[1].split(']')[0].split(', ')]
                bb = eval(txt.split('min_bet = ')[1].split('\n')[0])
                small_bet = eval(txt.split('min_bet = ')[1].split('\n')[0])//2
                actions = eval(txt.split('actions = ')[1].split('\n')[0])
                antes = eval(txt.split('antes = ')[1].split('\n')[0])
                blinds = eval(txt.split('blinds_or_straddles = ')[1].split('\n')[0])
                players = eval(txt.split('players = ')[1].split('\n')[0])
    
                #board = get_board(actions)
                pluribus_index = players.index('Pluribus')
                winner_int = pluribus_index

                n_players = len(antes)
                prob_dist = [(1-winner_p) / (n_players - 1)] * n_players
                prob_dist[winner_int] = winner_p
                player = random.choices(range(n_players), weights=prob_dist)[0]
                player_str = f'p{player+1}'

                player_traj = get_player_traj(actions, player_str, n_players)
                player_traj = prune(player_traj, player_str)
                if not player_traj:
                    continue
                msgs = [{'role': 'system', 'content': sys}]
                assert len(player_traj) % 2 == 0

                n_db = 0

                stacks,antes,blinds = normalize(start_stacks, bb), normalize(antes, bb), normalize(blinds, bb)
                stacks = [st-at-bl for st,at,bl in zip(stacks,antes,blinds)]
                stacks = {f'p{i+1}':stacks[i] for i in range(n_players)}
                bets = {f'p{i+1}':blinds[i] for i in range(n_players)}

                normalize_cbrs(player_traj, bb)
                for i,traj in enumerate(player_traj):
                    if i % 2 == 0:
                        role = 'user'
                        db = [d for d in traj if 'db' in d]

                        if len(db) > 0:
                            n_db+=1
                        print(stacks)
                        calc_stack_bets(stacks, bets, player_traj[i])
                        content = prompt.format(
                            player = f'p{player+1}',
                            n_players = n_players,
                            round = int2round[n_db],
                            actions = traj,
                            stack = stacks,
                            pot = sum([v for _,v in bets.items()]),
                            bets = bets
                        )
                        msgs.append({'role': role, 'content': content})
                    else:
                        action = traj[0][3:]
                        msgs.append({'role': 'assistant', 'content': f'p{player+1} action: {action}'})
                dataset.append(msgs)
            except Exception as e:
                raise e
    
    # Save the dataset as a proper .jsonl file
    import json
    output_file = os.path.join(out_dir, f'{folder.split("/")[-1]}.jsonl')
    with open(output_file, 'w') as f:
        for data in dataset:
            json_line = json.dumps({'messages': data})
            f.write(json_line + '\n')

def simulate_game(data):
    with open(data, 'r') as f:
        da = f.readlines()
    #print(d) 
    for d in da:
        d = eval(d)
        msgs = d['messages']
        for msg in msgs:
            print(msg)
            input()

if __name__ == '__main__':
   # simulate_game('/Users/minjunes/poker/data/35.txt')
   # exit()
    out_path = 'data'
    folders_path = 'phh-dataset/data/pluribus'
    for folder in os.listdir(folders_path):
        folder_path = os.path.join(folders_path, folder)
        make_dataset(folder_path, out_path)
