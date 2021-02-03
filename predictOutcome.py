#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:55:15 2021

@author: Charles Petersen and Jamison Barsotti
"""
import gameState as gs
import mcts
import utils

import numpy as np
import torch

def outcome_play(initial_state, net, device, search_iters):
    ''' Returns the oucome of the upset-downset game 'game' as seen from the 
    perspective of the first player to move.

    Parameters
    ----------
    game : UpDown
        a game of upset-downset
    net : UpDownNet
        the model to play.
    device : str
        the device the model is on : 'cuda' or 'cpu'.
    search_iters : int (nonnegative), optional
         the number of iterations of  MCTS to be performed for each turn. 
        The default is 800.

    Returns
    -------
    out : str
        'N' if the first player wins and 'P' if the second player wins.

    '''
    actions = np.arange(gs.UNIV)
    root = mcts.PUCTNode(initial_state)
    move_count = 0
    
    # play until a terminal state is reached
    while not gs.is_terminal_state(root.state):
        mcts.MCTS(root, net, device, search_iters)
        policy = mcts.MCTS_policy(root, 0)
        move = np.random.choice(actions, p=policy)
        root = root.edges[move]
        root.to_root()
        move_count += 1
        
    # update outcome: 'P' for second player to move 
    # and 'N' for first player to move
    out = 'P' if (move_count % 2 == 0) else 'N'

    return out

def outcome_prediction(game, alpha_or_apprentice, search_iters):
    ''' Returns the outcome of the upset-downset game 'game' as predicted
    by most recent iteration of the model 'alpha_ or_apprentice'.( The 
    prediction is made by having the model 'alpha_or_apprentrice' perform two 
    instances of self-play on 'game' : with Up to play first and Down to 
    play first.) 

    Parameters
    ----------
    game : UpDown
        a game of upset-downset.
    alpha_or_apprentice : str
        the model to be used for prediction : 'alpha' or 'apprentice'.
    search_iters : int (nonnegative), optional
         the number of iterations of  MCTS to be performed for each turn. 
        The default is 800.

    Returns
    -------
    predict_out : str
        the predicted outcome of 'game':
        'Next', Next player (first player to move) wins.
        'Previous', Previous player (second player to move) wins.
        'Up', Up can force a win. (Playing first or second.) 
        'Down', Down can force a win. (Playing first or second.)
        
    '''
    # get the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    net = utils.get_model(alpha_or_apprentice, device)
    
    
    # game state from each players perspective
    up_start = gs.to_state(game, to_move=gs.UP)
    down_start = gs.to_state(game, to_move=gs.DOWN)
    
    # get outcomes
    up_start_out = outcome_play(up_start, net, device, search_iters)
    down_start_out = outcome_play(down_start, net, device, search_iters)
    
    # get outcome prediction
    if up_start_out == 'P' and down_start_out == 'P':
        pred_out = 'Previous'
    elif up_start_out == 'N' and down_start_out == 'N':
        pred_out = 'Next'
    elif up_start_out == 'P' and down_start_out == 'N':
        pred_out = 'Down'
    elif up_start_out == 'N' and down_start_out == 'P':
        pred_out = 'Up'
        
    return pred_out
