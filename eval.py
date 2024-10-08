# The API utilizes HTTP POST requests.  Requests and responses have a JSON body.
# There are three endpoints:
#   /api/login
#   /api/new_hand
#   /api/act
# To initiate a new hand, send a request to /api/new_hand.  To take an action, send a
# request to /api/act.
#
# The body of a sample request to /api/new_hand:
#   {"token": "a2f42f44-7ff6-40dd-906b-4c2f03fcee57"}
# The body of a sample request to /api/act:
#   {"token": "a2f42f44-7ff6-40dd-906b-4c2f03fcee57", "incr": "c"}
#
# A sample response from /api/new_hand or /api/act:
#   {'old_action': '', 'action': 'b200', 'client_pos': 0, 'hole_cards': ['Ac', '9d'], 'board': [], 'token': 'a2f42f44-7ff6-40dd-906b-4c2f03fcee57'}
#
# Note that if the bot is first to act, then the response to /api/new_hand will contain the
# bot's initial action.
#
# A token should be passed into every request.  With the exception that on the initial request to
# /api/new_hand, the token may be missing.  But all subsequent requests should contain a token.
# The token can in theory change over the course of a session (usually only if there is a long
# pause) so always check if there is a new token in a response and use it going forward.
#
# A client_pos of 0 indicates that you are the big blind (second to act preflop, first to act
# postflop).  1 indicates you are the small blind.
#
# Sample action that you might get in a response looks like this:
#   b200c/kk/kk/kb200
# An all-in can contain streets with no action.  For example:
#   b20000c///
#
# "k" indicates "check", "c" indicates "call", "f" indicates "fold" and "b" indicates "bet"
# (either an initial bet or a raise).
#
# Bet sizes are the number of chips that the player has put into the pot *on that street* (only).
# Consider this action:
#
#   b200c/kb400
#
# The flop bet here is a pot-size bet to 400 because there are 400 chips in the pot after the
# preflop action.  If the bet is called, then each player will have put a total of 600 chips into
# the pot counting both the preflop and the flop.
# 
# Slumbot plays with blinds of 50 and 100 and a stack size of 200 BB (20,000 chips).  The stacks
# reset after each hand.
import requests
import sys
import argparse

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
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        print('Error: %s' % r['error_msg'])
        sys.exit(-1)
        
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
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        print('Error: %s' % r['error_msg'])
        sys.exit(-1)
        
    return r

import torch

def regret_match(logits):
    if isinstance(logits, list):
        logits = torch.tensor(logits)
    n_actions = len(logits)
    relu_logits = torch.relu(logits)
    logits_sum = relu_logits.sum().item()
    strat = torch.zeros(n_actions)
    if logits_sum > 0:
        strategy_tensor = relu_logits / logits_sum
        strat = strategy_tensor.cpu()
    else:
        max_index = torch.argmax(relu_logits).item()
        strat[max_index] = 1.0
    return strat

def read_config(file_path):
    config = {}
    target_vars = ['NUM_PLAYERS', 'MODEL_DIM', 'NUM_ACTIONS', 'MAX_ROUND_BETS']
    
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
    
    return (config.get('NUM_PLAYERS'),
            config.get('MODEL_DIM'),
            config.get('NUM_ACTIONS'),
            config.get('MAX_ROUND_BETS'))

from poker_inference import forward
import os

FILE_PATH = 'out/20241005234550/const.log'
MODELS_PATH = 'out/20241005234550'
CFR_ITER = -1
NUM_PLAYERS, MODEL_DIM, NUM_ACTIONS, MAX_ROUND_BETS = read_config(FILE_PATH)

player_models = dict()
for i in range(NUM_PLAYERS):
    assert CFR_ITER == -1 or CFR_ITER < len(os.listdir(MODELS_PATH))
    only_dirs = [dir for dir in os.listdir(MODELS_PATH) if os.path.splitext(dir)[1] == '']
    print(only_dirs)
    iter_models = sorted(only_dirs)[CFR_ITER]
    path = os.path.join(MODELS_PATH, iter_models, str(i), 'model.pt')
    player_models[i] = path

def net_forward(player_idx, hand, board, status, fracs):
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
 
def PlayHand(token):
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

    # big blind
    client_last_bet = 50
    # small blind
    if r.get('client_pos') == 0:
        client_last_bet = 100

    while True:
        print('-----------------')
        print(repr(r))
        action = r.get('action')
        round = action.count('/')
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
            sys.exit(-1)
        
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
                player_idx,
                hole_cards,
                board,
                status,
                fracs
            )
            can_check = a['last_bettor'] == -1
            min_bet_amt = a['total_last_bet_to'] - client_last_bet
            print(f'min_bet_amt: {min_bet_amt}')
            mask_illegals(logits, pot, min_bet_amt)
            actidx = torch.argmax(regret_match(logits)).item()
            client_act = idx2act(actidx, pot, can_check)
            # check validity...
            if client_act == 'f':
                pass
            elif client_act == 'c':
                call_amt = min_bet_amt
                bet_frac = call_amt / pot
                if call_amt > stack:
                    call_amt = stack
                    bet_frac = call_amt / pot
                status.append(1)
                fracs.append(bet_frac)
                client_last_bet = a['total_last_bet_to']
                pot += call_amt
                stack -= call_amt
            elif client_act == 'k':
                print(f'last bet size: {a["last_bet_size"]}')
                status.append(0)
                fracs.append(0)
                client_last_bet = a['total_last_bet_to']
            else:
                print("betting")
                print(f"raw client act: {client_act}")
                bet_amt = float(client_act[1:])
                bet_frac = bet_amt / pot
                if round == 0 and a['street_last_bet_to']:
                    incr = 150 
                else:
                    incr = a['street_last_bet_to'] 
                print(f'incr: {incr}')
                print(f'bet_amt: {bet_amt}')
                client_act = f'b{int(incr+bet_amt)}'
                status.append(1)
                fracs.append(bet_frac)
                client_last_bet = a['total_last_bet_to'] + bet_amt       
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

def main():
    global baseline_totals 
    baseline_totals = []
    parser = argparse.ArgumentParser(description='Slumbot API example')
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    args = parser.parse_args()
    username = args.username
    password = args.password
    if username and password:
        token = Login(username, password)
    else:
        token = None

    # To avoid SSLError:
    #   import urllib3
    #   urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    num_hands = 100
    winnings = 0
    for h in range(num_hands):
        (token, hand_winnings) = PlayHand(token)
        winnings += hand_winnings
    print('Total winnings: %i' % winnings)
    print('bb/100: ', winnings/150)
    print('session_baseline_total avg: ', sum(baseline_totals)/len(baseline_totals))
    
if __name__ == '__main__':
    main()
