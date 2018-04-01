import numpy as np
import pandas as pd


learning_rate = 0.01
gamma = 0.9
epsilon = 0.9

def choose_action(state, qmatrix, numb_actions):
    """Makes a choice of what action the client should do

    Keyword arguments:
    qmatrix -- the qmatrix where the decision would be made from
    numb_actions -- the number of possible actions that can be done
    """

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
    """Trains the Qmatrix based on reward, and the previous state done etc

    Keyword arguments:
    state -- the previous state
    action -- the action that was taken before
    reward -- the reward for completing the previous action, in the previous state
    next_state -- the current state that occured after getting the reward
    qmatrix -- the qmatrix where the training will be stored
    numb_actions -- the number of possible actions that can be done
    """

    qmatrix = check_state_exist(next_state, qmatrix, numb_actions)
    q_predict = qmatrix.loc[state, int(action)]
    q_target = int(reward) + gamma * qmatrix.loc[next_state, :].max() 
    qmatrix.loc[state, int(action)] += learning_rate * (q_target - q_predict)  # update
    return qmatrix

def check_state_exist(state, qmatrix, numb_actions):
    """Checks if the current state is in the Qmatrix otherwise adds that empty row

    Keyword arguments:
    state -- the current state being checked if it exists in Qmatrix
    qmatrix -- the Qmatrix that will be updated if the state doesnt exist in it
    numb_actions -- the number of possible actions that can be done
    """

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
    """Creates a empty Qmatrix with columns of size of number of possible actions

    Keyword arguments:
    numb_actions -- the number of possible actions that can be done
    """

    return (pd.DataFrame(columns=list(range(numb_actions)), dtype=np.float64))
    