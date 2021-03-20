"""
@author: Charles Petersen and Jamison Barsotti
"""
import torch

#GAME STATE PARAMETERS
MAX_NODES = 32    #max number of nodes ina game playable by the agent
ENCODED_STATE_SHAPE = (4, MAX_NODES, MAX_NODES)    #shape of encoded game states
UP = 0    #token for Up player
DOWN = 1    #token for Down player
RGB_DIST = (0.5, 0.5)    #distribution of game node colorings in self-play/eval : (RGB, all green)

#MODEL_PARAMETERS
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    #device to put model on        
NUM_FILTERS = 96    #number of conv. filters in the input/residual blocks
NUM_RES_BLOCKS = 9    #number of residual blocks
NUM_POL_FILTERS = 2     #number of conv. filters in the policy head
NUM_VAL_FILTERS = 1    #number of conv. filters in the value head
NUM_HIDDEN_VAL = 96    #size of the hidden layer in the value head

#MCTS PARAMETERS
DIRICHLET_EPS = 0.25   #fraction of dirichlet noise to add to root probabilities
DIRICHLET_ALPHA = 1.0  #dirichlet distribution parameter
C_PUCT = 1.0    #puct formula parameter

#SELF_PLAY_PARAMETERS
ASYNC_SELF_PLAYS = 8    #the number of parallel self-play processes
SELF_PLAY_MARKOV_EXP = 1    #exponent for markov process in self play game generation
SELF_PLAY_SEARCH_ITERS = 600   #number of mcts search iterations in self play
TEMP = 1.0    #self play temperature
TEMP_THRSHLD = 2    #self play temperature threshold

#TRAINING PARAMETERS
GAMES_TO_TRAIN = 1000   #how long to wait for replay buffer to fill during training (seconds)
MAX_REPLAY_BUFFER = 100000    #max length of training replay buffer
BATCH_SIZE = 2    #number of training examples per epoch (2)
NUM_SYMMETRIES = 16    #number of symmetries per training example per epoch (16)
LEARN_RATE = 0.01    #sgd learning rate
MOMENTUM = 0.9    #sgd momentum
REGULAR = 0.0001   # L2 weight regularization parameter for sgd

#EVALUATION PARAMETERS
ASYNC_EVAL_PLAYS = 2    #the number of parallel evaluation processes
PLAYS_PER_EVAL = 100   #number of games per evaluation processs
NUM_EVAL_PLAYS = ASYNC_EVAL_PLAYS * PLAYS_PER_EVAL    #total evaluation games played
EPOCHS_TO_EVAL = 150000   #number of training epochs between evaluations
EVAL_PLAY_MARKOV_EXP = 2    #exponent for markov process in evaluation game generation
EVAL_PLAY_SEARCH_ITERS = 200      #number of mcts search iterations in evaluation play (100)
WIN_RATIO = 0.55    #win ratio the apprentice must overcome to be considered the new alpha