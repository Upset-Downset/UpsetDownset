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
    agent = model.Net(input_shape=model.OBS_SHAPE, actions_n=gs.UNIV).to(device)
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

def approx_outcome(game, device=device):
    ''' Returns the predicted outcome of 'game' from the current best model.

    Parameters
    ----------
    game : UpDown
        a game of upset-downset.
    device : str, optional
        the device to run the model on. 'cuda' if available, else 'cpu'.

    Returns
    -------
    aprx_out : str
        the outcome of 'game' as predicetd by the model:
        'Next', Next player (first player to move) wins.
        'Previous', Previous player (second player to move) wins.
        'Up', Up can force a win. (Playing first or second.) 
        'Down', Down can force a win. (Playing first or second.)
        
    '''
    
    aprx_out = None
    N, P, L, R = 'Next', 'Previous', 'Up', 'Down'
    
    # load model
    agent = model.Net(input_shape=model.OBS_SHAPE, actions_n=gs.UNIV).to(device)
    agent.load_state_dict(torch.load('alpha_net.pt'))
    
    # game state from each players perspective
    up_state = gs.to_state(game, to_move=gs.UP)
    down_state = gs.to_state(game, to_move=gs.DOWN)
    
    # get value from net from Ups perspective
    encoded_up_state = torch.from_numpy(
                up_state).float().reshape(1, 4, gs.UNIV, gs.UNIV).to(device)
    _, up_value = agent(encoded_up_state)
    up_value = up_value.item()
    
    # get value from net from downs perspective
    encoded_down_state = torch.from_numpy(
                down_state).float().reshape(1, 4, gs.UNIV, gs.UNIV).to(device)
    _, down_value = agent(encoded_down_state)
    down_value = down_value.item()
    
    # determine the approximate outcome
    if up_value > 0 and down_value > 0:
        aprx_out = N
    elif up_value > 0 and down_value < 0:
        aprx_out = L
    elif up_value < 0 and down_value < 0:
        aprx_out = P
    elif up_value < 0 and down_value > 0:
        aprx_out = R
        
    return aprx_out
    
    
    
    return 