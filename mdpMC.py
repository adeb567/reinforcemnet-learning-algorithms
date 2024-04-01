#on-policy first-visit Monte Carlo control
import random, numpy
from math import inf
import sys

if len(sys.argv) != 4:
    print("command to run: $python mdpMC.py reward discount_factor epsilon")
    sys.exit(1)

actions = {'up':(0, 1), 'down':(0, -1), 'left':(-1, 0), 'right':(1, 0)}

num_rows = 3
num_columns = 4

all_states = []

for col in range(num_columns):
    for row in range(num_rows):
        if (col == 1 and row == 1):
            continue
        else:
            all_states.append((col+1, row+1))

states = all_states[:-2]
terminal_states = all_states[-2:]

reward = float(sys.argv[1]) #-2
discount_factor = float(sys.argv[2]) #0.9
epsilon = float(sys.argv[3]) #0.4
probability = 1 - epsilon #optimal

policy = {state: random.choice(list(actions.keys())) for state in states}

returns = dict()
state_action_values = dict()

def add_tuples(t1, t2):
    if len(t1) == 0:
        return ()
    else:
        return (t1[0] + t2[0],) + add_tuples(t1[1:], t2[1:])

num_ep = 40000
print("number of itterations", num_ep)

for _ in range(num_ep):
    curr_state = random.choice(states)
    episode_states = []
    episode_policies = []

    while curr_state not in terminal_states and len(episode_states) < 20:
        if random.random() > probability:
            policy[curr_state] = random.choice(list(actions.keys()))

        episode_states.append(curr_state)
        episode_policies.append(policy[curr_state])

        next_state = add_tuples(actions[policy[curr_state]], curr_state)

        if next_state in all_states:
            curr_state = next_state


    G = 0

    if curr_state == (4, 3):
        G = 1
    elif curr_state == (4, 2):
        G = -1
    else:
        G = reward


    while episode_states:
        curr_state = episode_states.pop()
        curr_action = episode_policies.pop()
        state_action_pair = (curr_state, curr_action)

        G = discount_factor * G + reward

        Flag = True
        for (state, action) in zip(episode_states, episode_states):
            if (state, action) == state_action_pair:
                Flag = False

        if Flag:
            if state_action_pair not in returns:
                returns[state_action_pair] = []
            returns[state_action_pair].append(G)

            state_action_values[state_action_pair] = numpy.average(returns[state_action_pair])

            max_val = float("-inf")
            for action in actions:
                if (curr_state, action) in state_action_values and state_action_values[(curr_state, action)] > max_val:
                    max_val = state_action_values[(curr_state, action)]
                    policy[curr_state] = action



print("State: Action-values q(s, a)         Policy")
for state in states:
    print(state, ": ", end="")

    max_val = float("-inf")
    for action in actions:
        value = state_action_values.get((state, action), None)
        if value > max_val:
            max_val = value
            policy[state] = action
        if value is not None:
            value = round(value, 3)  # Round to three decimals
            print(f"{action}: {value},", end=" ")
        else:
            print(f"{action}: None,", end=" ")
    print(policy[state])

