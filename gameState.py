#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charlie petersen and Jamison Barsotti
"""

import numpy as np
import digraph
import upDown as ud
import copy

# universe size our games live in: the maximum number of nodes 
# a game our model can play
UNIV = 10

# for tracking the current player 
UP = 0
DOWN = 1

def to_state(game, dim = UNIV, to_move = UP):
    ''' Returns the encoded representation of 'game' from the persepective of 
    player 'to_move'. For use as input to a neural network.
       
    NOTE: The encoded representation of an upset-dowsnet game is always 
    viewed from the current players perspective (the player to move). We use 
    the convention that all players play from the perspective of Up. 
    I.e., if it is Downs move in the game, then we change perspective so 
    that Up is to move in the negtive of the game.

    Parameters
    ----------
    game : UpDown
        a game of upset-downset.
    dim : int (nonnegative), optional
        must be at least as large as the number of nodes in 'game'. 
        The default is UNIV.
    to_move : UP or DOWN, optional
        The player to move. The default is UP.

    Returns
    -------
    state : 3D-numpy array of shape (4, UNIV, UNIV)
        the encoded representation of 'game' is a 4 channel dim x dim 
        image stack. The four, dim x dim channels C_k (for k=0,1,2,3) together 
        constitute a binary feature encoding of the state of 'game' via the 
        adjacecny matrix of the directed acyclic graph underlying 'game', the
        node coloring of the graph and the player 'to_move'. 
        
        The first three channels C_0, C_1, C_2 encode the presence of nodes 
        (resp. blue, green, red)  and their edge relationships amongst one 
        another: for k=0,1,2 entry C_k(i,j) = 1 if node j has color 1+k and 
        node i is an ancestor of node j. The last channel C_3 encodes the
        player 'to_move': C_3 = 0 if Up is to move and 0 if Down is to
        move.
     
    '''
    assert len(game) <= dim, 'The game is to large.'

    # if the move is to Down, get the negative game
    # and set the last channel to a constant 1
    state = np.zeros((4, dim, dim), dtype=np.int8)
    if to_move == DOWN:
        state[3,:,:] = 1
        game = -game
  
    # get the underlying dag, its transitive closure and node coloring
    dag = game.dag
    tc = digraph.transitive_closure(dag)
    color_dict = game.coloring

    # fill in the encoded representation
    for node in tc:
      # set the diagonals of the first three channels according to
      # nodes which are present in the game and their colors 
      color = color_dict[node]
      state[1 - color, node, node] = 1
    
      # now set the remainder of each nodes upset 
      # in their proper channel, row and column according to color,
      # node and cover, respectively
      for cover in tc[node]:
        color = color_dict[cover]
        state[1 - color, node, cover] = 1
        
    return state

def to_game(state):
    '''Returns the upset-downset game from the encoded ''state'.
    
    Parameters
    ----------
    state : 3D-numpy array of shape (4, UNIV, UNIV)
        encoded representation of an upset-downset game.

    Returns
    -------
    UpDown
        the decoded upset-downset game corresponding to 'state'.

    '''
    # get node indices and their colors present in 'state'
    node_info = np.diagonal(state[0:3], axis1=1, axis2=2)
    node_info = np.argwhere(node_info)
    
    # set colors of nodes present in 'state'
    colors = { node : 1 - channel for channel, node in node_info}
    # set the corresponding directed acyclic graph
    dag = {}
    for info in node_info:
        node = info[1]
        upset = sum(state[[0,1,2],node, :])
        upset[node] = 0
        upset = np.argwhere(upset).reshape(-1)
        dag[node] = list(upset)
        
    return ud.UpDown(dag, coloring = colors)

def take_action(state, a):
    ''' Returns the next state of the game 
    Parameters
    ----------
    state : 3D-numpy array of shape (4, UNIV, UNIV)
        encoded representation of an upset-downset game.
    a : int (nonnegative)
        a valid node to be played in 'state'.

    Returns
    -------
    next_state : 3D-numpy array of shape (4, UNIV, UNIV)
        the current position of the upset-downset game after the upset of node
        'a' has been removed from the previous position 'state', and the 
        persepctive of tha game has been changed to reflect the current player 
        to move.

    ''' 
    assert a in valid_actions(state), 'That is not a valid action.'
    
    # remove node 'a' and its upset in 'state'
    # (could probably do this witopy, but i'm too lazy)
    take_a = copy.deepcopy(state)
    # rows is a 2-D array in which the (i,j) entry tells us to zero 
    # out the jth column in the ith array of 'take_a'
    rows = take_a[[0,1,2],[a],:]
    # remove move said columns
    for row in np.argwhere(rows):
      take_a[[row[0]],:,[row[1]]] = 0
    
    # initialize array for next state
    shp = state.shape
    next_state = np.zeros(shp, dtype=np.int8)
    # set player to move in next_state
    cur_player = state[3,0,0]
    next_state[3,:,:] = 1-cur_player
    # get 2-D array of the nodes still on the board and their color
    # after 'a' has been taken
    diags = np.diagonal(take_a[[0,1,2],:,:], axis1=1, axis2=2)
    idxs = np.argwhere(diags)
    
    #build 'next state' from 'take_a' and 'idxs'
    for row in idxs:
        node = row[1]
        color = 2-row[0]
        # sum over all node-th rows (gives the upset of the node)
        # which is to be the downset of the node in next state
        col = sum(take_a[[0,1,2],[node],:])
        # set the downset of node in 'next_state'
        next_state[[color],:,[node]] = col
        
    return next_state

def valid_actions(state):
    ''' Returns the current players valid actions in the upset-downset game
    encoded in 'state'

    Parameters
    ----------
    state : 3D-numpy array of shape (4, UNIV, UNIV)
        encoded representation of an upset-downset game.

    Returns
    -------
    1-D numpy array
        nodes available to play in 'state' from the persepective of the 
        current player.

    '''   
    d = np.sum(np.diagonal(state[[0,1]], 0,2),0)
    
    return np.nonzero(d)[0]

def is_terminal_state(state):
    ''' Returns wether current player has lost the game. (Check before current
     player attemots to takes an action!)

    Parameters
    ----------
    state : 3D-numpy array of shape (4, UNIV, UNIV)
        encoded representation of an upset-downset game.

    Returns
    -------
    bool
        True if the current player has lost the game, and False otherwise.

    '''
    return True if len(valid_actions(state)) == 0 else False

def exploit_symmetries(training_triple, dim = UNIV, num_samples=10):
    '''Returns 'num_samples' symmetries of 'training_triple'.
    (A reindexing of node labels in any upset-downset game provides a
     symmetry of the gameboard.)
    
    Parameters
    ----------
    training_triple : tuple
        a triple of state, policy and value for training from self-play.
    dim : int (nonnegative), optional
        must be at least as large as the number of nodes in 'game'. 
        The default is UNIV.
    num : int (nonnegative), optional
        The number of symmetries to take (sampled w/ repitition).
        The default is 10.

    Returns
    -------
    sym_train_data : list
        'num_samples' triples of training data after symmetries have 
        been applied to 'training data'.

    ''' 
    state, policy, value =  training_triple
    sym_train_data = []
    
    for _ in range(num_samples):
        # get random permutation on dim # letters
        p = np.random.permutation(dim)
        # re-index nodes by permuting columns and rows
        state_sym = state[:,:,p]
        state_sym = state_sym[:,p,:]
        # permute nodes in policy too!
        policy_sym = policy[p]
        sym_train_data. append((state_sym, policy_sym, value))
        
    return sym_train_data
  
     
