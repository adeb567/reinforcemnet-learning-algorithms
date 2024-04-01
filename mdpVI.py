#value iteration
import sys

if len(sys.argv) != 3:
    print("command to run: $python mdpVI.py reward discount_factor")
    sys.exit(1)

actions = [(0, 1), (0, -1), (-1, 0), (1, 0)] 
action_names = ['up', 'down', 'left', 'right']
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

reward = float(sys.argv[1]) #-0.2
discount_factor = float(sys.argv[2]) #0.9
max_error = 0.001

utilities = {state: 0 for state in states}

utilities[(4,3)] = 1
utilities[(4,2)] = -1


def add_tuples(t1, t2):
    if len(t1) == 0:
        return ()
    else:
        return (t1[0] + t2[0],) + add_tuples(t1[1:], t2[1:])


def max_next_utility(state):
    max = float("-inf")
    
    for primary_action in actions:
        temp = 0

        for action in actions:
            new_state = add_tuples(state, action)

            if action == primary_action:
                if new_state not in all_states:
                    temp += 0.8 * utilities[state]
                else:
                    temp += 0.8 * utilities[new_state]

            elif add_tuples(primary_action, action) != (0, 0):
                if new_state not in all_states:
                    temp += 0.1 * utilities[state]
                else:
                    temp += 0.1 * utilities[new_state]

        if temp > max:
            max = temp

    return max

def get_best_action(state):
    values = []
    
    for primary_action in actions:
        temp = 0

        for action in actions:
            new_state = add_tuples(state, action)

            if action == primary_action:
                if new_state not in all_states:
                    temp += 0.8 * utilities[state]
                else:
                    temp += 0.8 * utilities[new_state]

            elif add_tuples(primary_action, action) != (0, 0):
                if new_state not in all_states:
                    temp += 0.1 * utilities[state]
                else:
                    temp += 0.1 * utilities[new_state]

        values.append(temp)

    return action_names[values.index(max(values))]



while True:
    max_utility = 0

    for state in states:
        utilities_next = reward + discount_factor * max_next_utility(state)
        max_utility = max(max_utility, abs(utilities_next - utilities[state]))
        utilities[state] = utilities_next

    if max_utility < max_error * (1 - discount_factor)/discount_factor:
        break
        
print("State : State-values v(s), Policy")
for state in states:
    print(state,":", round(utilities[state], 3), get_best_action(state))