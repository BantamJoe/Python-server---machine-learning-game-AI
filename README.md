# Python-server---machine-learning-game-AI
Q learning, python server for a machine learning game AI

This code represents the server aspect of my machine learning game AI. It provides
a way for multiple clients to connect, and train 5 Q matrices simultaneously. 

It establishes a TCP socket connect with the clients, whom transmit state and reward data to the server.
The server then interprets the recieved message, trains the Q matrix (if needed), and then
transmit a new set of instructions for the client.

# To run:

To launch the server, simply run:

python3 run.py
