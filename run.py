import socket, select
import pandas as pd
import numpy as np
import RLLogic
import atexit

CONNECTION_LIST = []    # list of socket clients
NUMB_MOVE_ACTIONS = 9
NUMB_ACTION_ACTIONS = 4

MOVE_QMATRIX_LIST = [RLLogic.generate_empty_qmatrix(NUMB_MOVE_ACTIONS), 
                RLLogic.generate_empty_qmatrix(NUMB_MOVE_ACTIONS),
                RLLogic.generate_empty_qmatrix(NUMB_MOVE_ACTIONS),
                RLLogic.generate_empty_qmatrix(NUMB_MOVE_ACTIONS),
                RLLogic.generate_empty_qmatrix(NUMB_MOVE_ACTIONS)]    # list of Qmatrix

ACTION_QMATRIX_LIST = [RLLogic.generate_empty_qmatrix(NUMB_ACTION_ACTIONS), 
                RLLogic.generate_empty_qmatrix(NUMB_ACTION_ACTIONS),
                RLLogic.generate_empty_qmatrix(NUMB_ACTION_ACTIONS),
                RLLogic.generate_empty_qmatrix(NUMB_ACTION_ACTIONS),
                RLLogic.generate_empty_qmatrix(NUMB_ACTION_ACTIONS)]    # list of Qmatrix


PREV_SCORE_LIST = []    # list of prev Scores
PREV_STATE_LIST = []    # list of prev States
PREV_MOVE_ACTION_LIST = []    # list of prev move actions
PREV_ACTION_ACTION_LIST = []    # list of prev actions (not move actions)

RECV_BUFFER = 10000 
#socket.setdefaulttimeout(20)
SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 5000

#NOTE: WILL HAVE TO WORK OUT IF LOST OR WON, BECAUSE GOOD GAME DESIGN LULZ

def get_client_numb(sock, limit=True):
    id = CONNECTION_LIST.index(sock)
    if (limit):
        client_numb = ((id - 1) % 5)
    else:
        client_numb = (id - 1)
    return client_numb

def recieved_state(sock, message, move_qmatrix, action_qmatrix, prev_state_idx):
    state = str(message)
    move_qmatrix = RLLogic.check_state_exist(state, move_qmatrix, NUMB_MOVE_ACTIONS)
    action_qmatrix = RLLogic.check_state_exist(state, action_qmatrix, NUMB_ACTION_ACTIONS)

    MOVE_QMATRIX_LIST[get_client_numb(sock)] = move_qmatrix
    ACTION_QMATRIX_LIST[get_client_numb(sock)] = action_qmatrix

    #chose action
    move_action = str(RLLogic.choose_action(state, move_qmatrix, NUMB_MOVE_ACTIONS))
    action_action = str(RLLogic.choose_action(state, action_qmatrix, NUMB_ACTION_ACTIONS))

    #send action
    send_data(sock, "MOVE_ACTION " + move_action)
    send_data(sock, "ACTION_ACTION " + action_action)

    PREV_STATE_LIST[get_client_numb(sock, False)] = state
    PREV_MOVE_ACTION_LIST[get_client_numb(sock, False)] = move_action
    PREV_ACTION_ACTION_LIST[get_client_numb(sock, False)] = action_action

def recieved_reward(sock, message, move_qmatrix, action_qmatrix, prev_state_idx):
    #get previous state
    prev_state = PREV_STATE_LIST[get_client_numb(sock, False)]

    #get this reward
    reward = str(message[0])

    prev_move_action = PREV_MOVE_ACTION_LIST[get_client_numb(sock, False)]
    prev_action_action = PREV_ACTION_ACTION_LIST[get_client_numb(sock, False)]
    #print(PREV_MOVE_ACTION_LIST)
    #get this state
    next_state = str(message[1:])

    #train
    
    move_qmatrix = RLLogic.check_state_exist(next_state, move_qmatrix, NUMB_MOVE_ACTIONS)
    
    action_qmatrix = RLLogic.check_state_exist(next_state, action_qmatrix, NUMB_ACTION_ACTIONS)
    move_qmatrix = RLLogic.learn(prev_state, prev_move_action, reward, next_state, move_qmatrix, NUMB_MOVE_ACTIONS)
    MOVE_QMATRIX_LIST[get_client_numb(sock)] = move_qmatrix

    action_qmatrix = RLLogic.learn(prev_state, prev_action_action, reward, next_state, action_qmatrix, NUMB_ACTION_ACTIONS)
    ACTION_QMATRIX_LIST[get_client_numb(sock)] = action_qmatrix

    #Send next action
    recieved_state(sock, next_state, move_qmatrix, action_qmatrix, prev_state_idx)

def recieved_score(sock, message, move_qmatrix, action_qmatrix, prev_state_idx):
    #Calculate if score changed - myteam : enemy team
    score = message[0] + " " + message[1]
    if (score == PREV_SCORE_LIST[get_client_numb(sock, False)]):
        return

    #if so then calculate the reward (lose or win etc)
    prev_score = PREV_SCORE_LIST[get_client_numb(sock, False)].split()
    if ((PREV_SCORE_LIST[get_client_numb(sock, False)] == "") or (score == "0 0")):
        PREV_SCORE_LIST[get_client_numb(sock, False)] = score
        return
    reward = 0
    if (score[0] > prev_score[0]): #means I just gained a point
        reward += 10
    if (score[1] > prev_score[1]): #means enemy just gained a point
        reward -= 10

    #then train and send new state
    message[1] = str(reward)
    recieved_reward(sock, message[1:], move_qmatrix, action_qmatrix, prev_state_idx)
    PREV_SCORE_LIST[get_client_numb(sock, False)] = score

def received_complete(sock, message, move_qmatrix, action_qmatrix, prev_state_idx):
    
    #get the completed action
    completed_action = str(message[0])

    state = PREV_STATE_LIST[get_client_numb(sock, False)]
    if completed_action == "action":
        action_action = str(RLLogic.choose_action(state, action_qmatrix, NUMB_ACTION_ACTIONS))
        send_data(sock, "ACTION_ACTION " + action_action)
        PREV_ACTION_ACTION_LIST[get_client_numb(sock, False)] = action_action
    elif completed_action == "move":
        move_action = str(RLLogic.choose_action(state, move_qmatrix, NUMB_MOVE_ACTIONS))
        send_data(sock, "MOVE_ACTION " + move_action)
        PREV_MOVE_ACTION_LIST[get_client_numb(sock, False)] = move_action
    else:
        print("Unknown complete type")
    return

API_ACCESS= {
    'COMPLETE' : {"function": received_complete},
    'STATE' : {"function": recieved_state},
    'SCORE' : {"function": recieved_score},
    'REWARD' : {"function": recieved_reward}
}

#Sends data to the designated socket only
def send_data(sock, message):
    sock.send((message + "\n").encode())
    #print("Sending data: %s \n" % message)


def intercept_message(sock, message, move_qmatrix, action_qmatrix):
    split_message = message.split()
    if split_message[0] in API_ACCESS:
        prev_state_idx = get_client_numb(sock)
        API_ACCESS[split_message[0]]["function"](sock, split_message[1:], move_qmatrix, action_qmatrix, prev_state_idx)
    else:
        send_data(sock, "ERROR API call not found!")

def save_qmatrix():
    for x in range(0, 5):
        filename = str(x) + "_MOVE.csv"
        MOVE_QMATRIX_LIST[x].to_csv(filename)
        filename = str(x) + "_MOVE.pkl"
        MOVE_QMATRIX_LIST[x].to_pickle(filename)

        filename = str(x) + "_ACTION.csv"
        ACTION_QMATRIX_LIST[x].to_csv(filename)
        filename = str(x) + "_ACTION.pkl"
        ACTION_QMATRIX_LIST[x].to_pickle(filename)

def load_qmatrix():
    for x in range(0, 5):

        filename = str(x) + "_MOVE.pkl"
        MOVE_QMATRIX_LIST[x] = pd.read_pickle(filename)
        print(str(MOVE_QMATRIX_LIST[x]))

        filename = str(x) + "_ACTION.pkl"
        ACTION_QMATRIX_LIST[x] = pd.read_pickle(filename)



def create_server():
    global SERVER_SOCKET

    SERVER_SOCKET.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    SERVER_SOCKET.bind(("0.0.0.0", PORT))
    SERVER_SOCKET.listen(10) # TODO: Only allow 10 connections for now

    atexit.register(save_qmatrix)
    load_qmatrix()
 
    # Add server socket to the list of readable connections
    CONNECTION_LIST.append(SERVER_SOCKET)
 
    print("Unity AI server started on port " + str(PORT))
 
    while True:
        # Get the list sockets which are ready to be read through select
        try:
            read_sockets,_,_ = select.select(CONNECTION_LIST,CONNECTION_LIST,[])
        except select.error:
            print("error occured")
        for sock in read_sockets:
            
            #New connection
            if sock == SERVER_SOCKET:
                # Handle the case in which there is a new connection recieved through SERVER_SOCKET
                sockfd, addr = SERVER_SOCKET.accept()
            
                CONNECTION_LIST.append(sockfd)
                
                print("Client (%s, %s) connected" % addr)
                send_data(sockfd, "CONNECTED")
                PREV_SCORE_LIST.append("")
                PREV_STATE_LIST.append("")
                PREV_MOVE_ACTION_LIST.append("")
                PREV_ACTION_ACTION_LIST.append("")
                #print(CONNECTION_LIST.index(sockfd))
                
                  
            #Some incoming message from a client
            else:
                try: 
                    data = sock.recv(RECV_BUFFER)
                    # echo back the client message
                    if data:
                        message = data.decode("utf-8")
                        move_qmatrix = MOVE_QMATRIX_LIST[get_client_numb(sock)]
                        action_qmatrix = ACTION_QMATRIX_LIST[get_client_numb(sock)]
                        for mess in message.splitlines():
                            #print("Data recieved " + str(get_client_numb(sock)) + ": " +  mess)
                            intercept_message(sock, mess, move_qmatrix, action_qmatrix)
                    elif not data:
                        raise select.error
                        # client disconnected, so remove from socket list
                except Exception as e:
                    print(e)
                    print("Client (%s, %s) discconected" % addr)
                    sock.close()
                    del PREV_SCORE_LIST[get_client_numb(sock, False)]
                    del PREV_STATE_LIST[get_client_numb(sock, False)]
                    del PREV_MOVE_ACTION_LIST[get_client_numb(sock, False)]
                    del PREV_ACTION_ACTION_LIST[get_client_numb(sock, False)]
                    CONNECTION_LIST.remove(sock)
                    continue
         
    SERVER_SOCKET.close()


if __name__ == "__main__":
    create_server()