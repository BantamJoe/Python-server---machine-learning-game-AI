import numpy as np
import pandas as pd


learning_rate = 0.01
gamma = 0.9
epsilon = 0.9

def choose_action(state, qmatrix, numb_actions):
    qmatrix = check_state_exist(state, qmatrix, numb_actions)
    # action selection
    if np.random.uniform() < epsilon:
        # choose best action
        state_action = qmatrix.loc[state, :]
        state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
        action = state_action.idxmax()
    else:
        # choose random action
        action = np.random.choice(list(range(numb_actions))) 
    return action

def learn(state, action, reward, next_state, qmatrix, numb_actions):
    qmatrix = check_state_exist(next_state, qmatrix, numb_actions)
    q_predict = qmatrix.loc[state, int(action)]
    q_target = int(reward) + gamma * qmatrix.loc[next_state, :].max() 
    qmatrix.loc[state, int(action)] += learning_rate * (q_target - q_predict)  # update
    return qmatrix

def check_state_exist(state, qmatrix, numb_actions):
    if state not in qmatrix.index:
        # append new state to q table
        qmatrix = qmatrix.append(
            pd.Series(
                [0]*numb_actions,
                index=qmatrix.columns,
                name=state,
            )
        )
    return qmatrix

def generate_empty_qmatrix(numb_actions):
    return (pd.DataFrame(columns=list(range(numb_actions)), dtype=np.float64))
    