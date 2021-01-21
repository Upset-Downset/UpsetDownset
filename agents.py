#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import model 
import gameState as gs
import mcts

import random
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def alpha_zero(game, player, device=device, search_iters=800):
    '''Returns the deterministic move suggetsed by current best model after 
    performing an MCTS with 'search_iters' iterations on 'game' from the
    perspective of 'player'.

    Parameters
    ----------
    game : UpDown
        a game of upset-downset.
    device : str, optional
        the device to run the model on. 'cuda' if available, else 'cpu'.
    player : str
        'Up' if the agent is to be the Up player, and 'Down' if the agent is
        to be the Down player.
    search_iters : int (nonegative), optional
        teh number of MCTS iterations to perform. The default is 800.

    Returns
    -------
    move : int (nonnegative)
        node in 'game'.

    '''
    cur_player = gs.UP if player == 'Up' else gs.DOWN
    
    #load agent
    agent = model.UpDownNet(
        input_shape=gs.STATE_SHAPE, 
        actions_n=gs.UNIV).to(device)
    agent.load_state_dict(torch.load('alpha_net.pt'))
    
    # encode game
    state = gs.to_state(game, to_move=cur_player)
    root = mcts.PUCTNode(state)
    
    # run mcts from root
    mcts.MCTS(root, agent, device, num_iters=search_iters)
    
    # get move via deterministic mcts policy
    actions = np.arange(gs.UNIV)
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
    options = game.up_nodes() if player == 'Up' else game.down_nodes()

    return random.choice(options)
