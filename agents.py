"""
@author: Charles Petersen and Jamison Barsotti
"""
import utils
from gameState import GameState
import  mcts
import random
import torch
import numpy as np

def alpha(game, player_to_move, look_ahead=800):
    '''Returns the deterministic move suggetsed by current best model after 
    performing an MCTS with 'search_iters' iterations on 'game_state'.

    Parameters
    ----------
    game : UpDown
        an upset-downset game.
    look_ahead : int (nonegative), optional
        the number of MCTS iterations to perform. The default is 800.

    Returns
    -------
    move : int (nonnegative)
        node in 'game'.

    '''
    
    #load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = utils.load_model('alpha', device)
    
    # get game state
    current_player = GameState.UP if player_to_move =='up' \
        else GameState.DOWN
    game_state = GameState(game, current_player)
    
    # run mcts from root
    root = mcts.PUCTNode(game_state)
    mcts.MCTS(root, net, device, search_iters=look_ahead)
    
    # get move via deterministic mcts policy
    actions = np.arange(GameState.NUM_ACTIONS)
    policy = mcts.MCTS_policy(root, temp=0)
    move = np.random.choice(actions, p=policy)
    
    return move

def random_agent(game, player):
    ''' Returns a random move in the upset-downset 'game' for 'player'.
    
    Parameters
    ----------
    game : UpDown
        a game of upset-downset
    player : str
        'Up' if the agent is to be the Up player, and 'Down' if the agent is
        to be the Down player.

    Returns
    -------
    int
        element in the poset underlying the upset-downset 'game' on which 
        the agent is to make its random play.

    '''
    # Determine which nodes can be played
    options = game.up_nodes() if player == 'up' else game.down_nodes()

    return random.choice(options)
