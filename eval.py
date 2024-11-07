from poker_inference import forward

import requests
import sys
import argparse
import os
import wandb

host = 'slumbot.com'

NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
def encode():
    # return hand, flop, turn, river, bet_status and bet_history
    pass

def card2int(card_str):
    if card_str is None:
        return -1
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    
    rank = ranks.index(card_str[0])
    suit = suits.index(card_str[1])
    card = rank * 4 + suit
    return card

def ParseAction(action):
    """
    Returns a dict with information about the action passed in.
    Returns a key "error" if there was a problem parsing the action.
    pos is returned as -1 if the hand is over; otherwise the position of the player next to act.
    street_last_bet_to only counts chips bet on this street, total_last_bet_to counts all
      chips put into the pot.
    Handles action with or without a final '/'; e.g., "ck" or "ck/".
    """
    st = 0
    street_last_bet_to = BIG_BLIND
    total_last_bet_to = BIG_BLIND
    last_bet_size = BIG_BLIND - SMALL_BLIND
    last_bettor = 0
    sz = len(action)
    pos = 1
    if sz == 0:
        return {
            'st': st,
            'pos': pos,
            'street_last_bet_to': street_last_bet_to,
            'total_last_bet_to': total_last_bet_to,
            'last_bet_size': last_bet_size,
            'last_bettor': last_bettor,
        }

    check_or_call_ends_street = False
    i = 0
    while i < sz:
        if st >= NUM_STREETS:
            return {'error': 'Unexpected error'}
        c = action[i]
        i += 1
        if c == 'k':
            if last_bet_size > 0:
                return {'error': 'Illegal check'}
            if check_or_call_ends_street:
	        # After a check that ends a pre-river street, expect either a '/' or end of string.
                if st < NUM_STREETS - 1 and i < sz:
                    if action[i] != '/':
                        return {'error': 'Missing slash'}
                    i += 1
                if st == NUM_STREETS - 1:
	            # Reached showdown
                    pos = -1
                else:
                    pos = 0
                    st += 1
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
        elif c == 'c':
            if last_bet_size == 0:
                return {'error': 'Illegal call'}
            if total_last_bet_to == STACK_SIZE:
	        # Call of an all-in bet
	        # Either allow no slashes, or slashes terminating all streets prior to the river.
                if i != sz:
                    for st1 in range(st, NUM_STREETS - 1):
                        if i == sz:
                            return {'error': 'Missing slash (end of string)'}
                        else:
                            c = action[i]
                            i += 1
                            if c != '/':
                                return {'error': 'Missing slash'}
                if i != sz:
                    return {'error': 'Extra characters at end of action'}
                st = NUM_STREETS - 1
                pos = -1
                last_bet_size = 0
                return {
                    'st': st,
                    'pos': pos,
                    'street_last_bet_to': street_last_bet_to,
                    'total_last_bet_to': total_last_bet_to,
                    'last_bet_size': last_bet_size,
                    'last_bettor': last_bettor,
                }
            if check_or_call_ends_street:
	        # After a call that ends a pre-river street, expect either a '/' or end of string.
                if st < NUM_STREETS - 1 and i < sz:
                    if action[i] != '/':
                        return {'error': 'Missing slash'}
                    i += 1
                if st == NUM_STREETS - 1:
	            # Reached showdown
                    pos = -1
                else:
                    pos = 0
                    st += 1
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            last_bet_size = 0
            last_bettor = -1
        elif c == 'f':
            if last_bet_size == 0:
                return {'error', 'Illegal fold'}
            if i != sz:
                return {'error': 'Extra characters at end of action'}
            pos = -1
            return {
                'st': st,
                'pos': pos,
                'street_last_bet_to': street_last_bet_to,
                'total_last_bet_to': total_last_bet_to,
                'last_bet_size': last_bet_size,
                'last_bettor': last_bettor,
            }
        elif c == 'b':
            j = i
            while i < sz and action[i] >= '0' and action[i] <= '9':
                i += 1
            if i == j:
                return {'error': 'Missing bet size'}
            try:
                new_street_last_bet_to = int(action[j:i])
            except (TypeError, ValueError):
                return {'error': 'Bet size not an integer'}
            new_last_bet_size = new_street_last_bet_to - street_last_bet_to
            # Validate that the bet is legal
            remaining = STACK_SIZE - total_last_bet_to
            if last_bet_size > 0:
                min_bet_size = last_bet_size
	        # Make sure minimum opening bet is the size of the big blind.
                if min_bet_size < BIG_BLIND:
                    min_bet_size = BIG_BLIND
            else:
                min_bet_size = BIG_BLIND
            # Can always go all-in
            if min_bet_size > remaining:
                min_bet_size = remaining
            if new_last_bet_size < min_bet_size:
                return {'error': 'Bet too small'}
            max_bet_size = remaining
            if new_last_bet_size > max_bet_size:
                return {'error': 'Bet too big'}
            last_bet_size = new_last_bet_size
            street_last_bet_to = new_street_last_bet_to
            total_last_bet_to += last_bet_size
            last_bettor = pos
            pos = (pos + 1) % 2
            check_or_call_ends_street = True
        else:
            return {'error': 'Unexpected character in action'}

    return {
        'st': st,
        'pos': pos,
        'street_last_bet_to': street_last_bet_to,
        'total_last_bet_to': total_last_bet_to,
        'last_bet_size': last_bet_size,
        'last_bettor': last_bettor,
    }


def NewHand(token):
    data = {}
    if token:
        data['token'] = token
    # Use verify=false to avoid SSL Error
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/new_hand', headers={}, json=data)
    success = getattr(response, 'status_code') == 200
    if not success:
        print('Status code: %s' % repr(response.status_code))
        try:
            print('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        raise Exception("unhandled exception")

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        raise Exception("could not get JSON from response")

    if 'error_msg' in r:
        print('Error: %s' % r['error_msg'])
        raise Exception(r['error_msg'])
        
    return r


def Act(token, action):
    data = {'token': token, 'incr': action}
    # Use verify=false to avoid SSL Error
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/act', headers={}, json=data)
    success = getattr(response, 'status_code') == 200
    if not success:
        print('Status code: %s' % repr(response.status_code))
        try:
            print('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        raise Exception('unhandled exception')

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        raise Exception('could not get JSON from resposne')

    if 'error_msg' in r:
        print('Error: %s' % r['error_msg'])
        raise Exception(r['error_msg'])
        
    return r

import numpy as np

def regret_match(logits):
    if isinstance(logits, list):
        logits = np.array(logits)
    n_actions = len(logits)
    relu_logits = np.maximum(logits, 0)  # ReLU operation
    logits_sum = relu_logits.sum()
    strat = np.zeros(n_actions)
    
    if logits_sum > 0:
        strat = relu_logits / logits_sum
    else:
        max_index = np.argmax(relu_logits)
        strat[max_index] = 1.0
        
    return strat

def read_config(file_path):
    config = {}
    target_vars = ['NUM_PLAYERS', 'MODEL_DIM', 'NUM_ACTIONS', 'MAX_ROUND_BETS', 'NUM_TRAVERSALS', 'CFR_ITERS', 'TRAIN_BS', 'TRAIN_ITERS']
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            for var in target_vars:
                if line.startswith(f'constexpr size_t {var} = ') or \
                   line.startswith(f'constexpr int {var} = ') or \
                   line.startswith(f'constexpr int64_t {var} = '):
                    value = line.split('=')[1].strip().rstrip(';')
                    config[var] = int(value)
                    break
    
    return {target_var:config.get(target_var) for target_var in target_vars}



def net_forward(player_models, player_idx, hand, board, status, fracs):
    assert len(status) == len(fracs)
    hand = [card2int(h) for h in hand]
    board += [None] * (5-len(board))
    flops = [card2int(c) for c in board[:3]]
    turn = card2int(board[3]) 
    river = card2int(board[4])
    bet_status = status + [0] * (MAX_ROUND_BETS*NUM_PLAYERS*4 - len(status))
    bet_fracs = fracs + [0] * (MAX_ROUND_BETS*NUM_PLAYERS*4 - len(fracs))    
    
    logits = forward(
        player_models[player_idx],
        hand,
        flops,
        turn,
        river,
        bet_fracs,
        bet_status
    )
    
    return logits

def mask_illegals(logits, pot, min_bet_amt):
    for i in range(2, len(logits)):
        bet_amt = float(idx2act(i, pot, None)[1:])
        if bet_amt < min_bet_amt:
            logits[i] = 0

# action idx 2 slumbot api compatbile act
# TODO fix betting
# currently the betting options are increasing proportion of pot
# however, some of these options may be under min bet
# would be nice if logits of illegal acts are automatically masked to 0 
def idx2act(idx, pot, can_check):
    if idx == 0:
        return 'f'
    elif idx == 1:
        if can_check:
            return 'k'
        else:
            return 'c'
    else:
        inc = pot / NUM_ACTIONS
        bet_amt = 0
        for a in range(2, NUM_ACTIONS):
            if a == idx:
                return f'b{int(bet_amt)}'
            else:
                bet_amt += inc
        raise Exception("act out of bounds") 
 
def PlayHand(player_models, token):
    r = NewHand(token)
    # We may get a new token back from /api/new_hand
    new_token = r.get('token')
    if new_token:
        token = new_token
    print('Token: %s' % token)

    # pos is randomly assigned
    # so query right net based on pos
    #  Blinds of 50 and 100, stack sizes of 200 BB
    status = [] 
    fracs = [] 
    stack = 100*200
    pot = 150
    round = 0

    # big blind
    client_last_bet = 50
    # small blind
    if r.get('client_pos') == 0:
        client_last_bet = 100

    while True:
        print('-----------------')
        print(repr(r))
        action = r.get('action')
        newround = action.count('/')
        if newround != round:
            client_last_bet = 0
            round = newround
        client_pos = r.get('client_pos')
        hole_cards = r.get('hole_cards')
        board = r.get('board')
        winnings = r.get('winnings')
        if 'session_baseline_total' in r:
            baseline_totals.append(r.get("session_baseline_total"))
        print('Action: %s' % action)
        if client_pos:
            print('Client pos: %i' % client_pos)
        print('Client hole cards: %s' % repr(hole_cards))
        print('Board: %s' % repr(board))
        if winnings is not None:
            print('Hand winnings: %i' % winnings)
            return (token, winnings)
        # Need to check or call
        a = ParseAction(action)
        if 'error' in a:
            print('Error parsing action %s: %s' % (action, a['error']))
            raise Exception(a['error'])
        
        print(a) 
        assert a['last_bettor'] != client_pos
        last_bet = a['last_bet_size']
        bet_frac = last_bet / pot
        status.append(1 if bet_frac > 0 else 0) 
        fracs.append(bet_frac)
        pot += last_bet

        if action != '' and len(status) >= round * MAX_ROUND_BETS * 2:
            client_act = 'f'
        else:
            #if a['last_bettor'] != -1:, then you must cbr
            player_idx = 1 if client_pos == 0 else 0
            logits = net_forward(
                player_models,
                player_idx,
                hole_cards,
                board,
                status,
                fracs
            )
            can_check = a['last_bettor'] == -1
            min_bet_amt = a['street_last_bet_to'] - client_last_bet
            print(f'min_bet_amt: {min_bet_amt}')
            mask_illegals(logits, pot, min_bet_amt)
            # fold is illegal
            regrets = regret_match(logits)
            if r['action'] == '':
                regrets[0] = -1
            actidx = np.argmax(regrets).item()
            client_act = idx2act(actidx, pot, can_check)
            # check validity...
            if client_act == 'f':
                if r['action'][-1] == 'k':
                    client_act = 'k'
                    status.append(0)
                    fracs.append(0)
                    client_last_bet = a['street_last_bet_to']
            elif client_act == 'c':
                call_amt = min_bet_amt
                bet_frac = call_amt / pot
                if call_amt > stack:
                    call_amt = stack
                    bet_frac = call_amt / pot
                status.append(1)
                fracs.append(bet_frac)
                client_last_bet = a['street_last_bet_to']
                pot += call_amt
                stack -= call_amt
            elif client_act == 'k':
                print(f'last bet size: {a["last_bet_size"]}')
                status.append(0)
                fracs.append(0)
                client_last_bet = a['street_last_bet_to']
            else:
                print("betting")
                print(f"raw client act: {client_act}")
                bet_amt = float(client_act[1:])
                if bet_amt == 0.0:
                    client_act = 'k'
                    print(f'last bet size: {a["last_bet_size"]}')
                    status.append(0)
                    fracs.append(0)
                    client_last_bet = a['street_last_bet_to']
                else:
                    bet_frac = bet_amt / pot
                    incr = max(a['street_last_bet_to'], 150)
                    print(f'incr: {incr}')
                    print(f'bet_amt: {bet_amt}')
                    print(f'post_bet_amt: {int(incr+bet_amt)}')
                    post_bet_amt = int(incr+bet_amt)
                    client_act = f'b{post_bet_amt}'
                    status.append(1)
                    fracs.append(bet_frac)
                    client_last_bet = a['street_last_bet_to'] + bet_amt       
                    pot += bet_amt
                    stack -= bet_amt
        print('Sending incremental action: %s' % client_act)
        r = Act(token, client_act)
    # Should never get here
        
def Login(username, password):
    data = {"username": username, "password": password}
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/login', json=data)
    success = getattr(response, 'status_code') == 200
    if not success:
        print('Status code: %s' % repr(response.status_code))
        try:
            print('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        print('Error: %s' % r['error_msg'])
        sys.exit(-1)
        
    token = r.get('token')
    if not token:
        print('Did not get token in response to /api/login')
        sys.exit(-1)
    return token

def auto(log_path, start_iter, num_hands=10, use_wandb=True, config={}, run_name=""):
    import time
    eval_cfr_iter = start_iter
    if use_wandb:
        print("initializing wandb")
        wandb.init(
            project = "poker ai",
            config={**config},
            name=run_name
        )

    while 1:
        both_player_exists = True
        for i in range(NUM_PLAYERS):
            path = os.path.join(MODELS_PATH, str(eval_cfr_iter), str(i), 'model.pt')
            if not os.path.exists(path):
                both_player_exists = False
        
        if both_player_exists:
            player_models = dict()

            for i in range(NUM_PLAYERS):
                assert eval_cfr_iter == -1 or eval_cfr_iter < len(os.listdir(MODELS_PATH))
                path = os.path.join(MODELS_PATH, str(eval_cfr_iter), str(i), 'model.pt')
                player_models[i] = path   

            if username and password:
                token = Login(username, password)
            else:
                token = None

            errors = []
            winnings = 0
            bb = 150
            for h in range(num_hands):
                try:
                    (token, hand_winnings) = PlayHand(player_models, token)
                    winnings += hand_winnings
                except Exception as e:
                    print(f"skipping hand {h} due to error: {e}")
                    errors.append(str(e))

            eval_results = {
                'eval_cfr_iter': eval_cfr_iter,
                'total_winnings_mbb': winnings/bb*1000,
                'mbb_per_hand': (winnings/bb)/num_hands*1000,
                'session_baseline_total_avg_mbb': sum(baseline_totals)/len(baseline_totals)/bb*1000 if baseline_totals else 0
            }

            if use_wandb:
                wandb.log(eval_results)

            with open(f'{log_path}/eval.log', 'a') as f:
                for key, value in eval_results.items():
                    log_line = f'{key} = {value}\n'
                    f.write(log_line)
            
            with open(f'{log_path}/eval_errs.log', 'a') as f:
                for err in errors:
                    f.write(err + '\n')
            eval_cfr_iter += 1
        else:
            time.sleep(10)
            print('sleeping 10 seconds...')
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    global baseline_totals, NUM_PLAYERS, MODEL_DIM, NUM_ACTIONS, MAX_ROUND_BETS

    baseline_totals = []
    parser = argparse.ArgumentParser(description='Slumbot API example')
    parser.add_argument('--username', type=str, default="")
    parser.add_argument('--password', type=str, default="")
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--num_hands', type=int, default=1000)
    parser.add_argument('--plot_all',type=int, default=1)
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--plot_intervals',type=int, default=1)
    parser.add_argument('--auto', type=int, default=1)
    parser.add_argument('--write', type=int, default=1)
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--name', type=str, default="")
    args = parser.parse_args()
    username = args.username
    password = args.password
    log_path = args.log_path
    num_hands = args.num_hands
    plot_all = args.plot_all
    start_iter = args.start_iter
    plot_intervals = args.plot_intervals
    automatic = args.auto
    write = args.write
    use_wandb = args.wandb
    run_name = args.name

    if not log_path:
        log_path = os.path.join('./out',sorted(os.listdir('out'))[-1])

    FILE_PATH = f'{log_path}/const.log'
    MODELS_PATH = log_path
    config = read_config(FILE_PATH)
    NUM_PLAYERS, MODEL_DIM, NUM_ACTIONS, MAX_ROUND_BETS, _, _, _, _ = tuple(config.values())

    if automatic:
        auto(log_path, start_iter, num_hands=num_hands, use_wandb=use_wandb, config=config, run_name=run_name)
        exit(-1)

    only_dirs = [dir for dir in os.listdir(MODELS_PATH) if os.path.splitext(dir)[1] == '']
    total_iters = len(only_dirs) if plot_all else 1

    for eval_cfr_iter in range(start_iter, total_iters+1, plot_intervals):

        player_models = dict()

        for i in range(NUM_PLAYERS):
            assert eval_cfr_iter == -1 or eval_cfr_iter < len(os.listdir(MODELS_PATH))
            path = os.path.join(MODELS_PATH, str(eval_cfr_iter), str(i), 'model.pt')
            player_models[i] = path   

        if username and password:
            token = Login(username, password)
        else:
            token = None

        # To avoid SSLError:
        #   import urllib3
        #   urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        errors = []
        winnings = 0
        bb = 150
        for h in range(num_hands):
            try:
                (token, hand_winnings) = PlayHand(player_models, token)
                winnings += hand_winnings
            except Exception as e:
                print(f"skipping hand {h} due to error: {e}")
                errors.append(str(e))

        eval_results = {
            'eval_cfr_iter': eval_cfr_iter,
            'total_winnings_mbb': winnings/bb*1000,
            'mbb_per_hand': (winnings/bb)/num_hands*1000,
            'session_baseline_total_avg_mbb': sum(baseline_totals)/len(baseline_totals)/bb*1000 if baseline_totals else 0
        }

        if write:
            with open(f'{log_path}/eval.log', 'a') as f:
                for key, value in eval_results.items():
                    log_line = f'{key} = {value}\n'
                    f.write(log_line)
            
            with open(f'{log_path}/eval_errs.log', 'a') as f:
                for err in errors:
                    f.write(err + '\n')
