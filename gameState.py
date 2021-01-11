#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:26:28 2021

@author: Charlie petersen and Jamison Barsotti
"""

import numpy as np
import dagUtility as dag
import copy


UNIV = 10
UP = 0
DOWN = 1

def to_state(G, dim = UNIV, first_move = UP):
  '''Take a game and spit out the encoded tensor. The tensor representation of the game 
  state is  represented by four UNIV x UNIV arrays:
  - the ij-th entry of the 0-th array is a 1 if node i >= node j and node j is 
  blue and 0 otherwise.
   and node j is blue.
  - the ij-th entry of a the 1-st array is a 1 node i >= node j and node j is 
  green and 0 otherwise.  
  - the the ij-th entry of a the 2-nd array is a 1 node i >= node j and node j 
  is red and 0 otherwise.
  - the 3-rd array is a constant 0. (UP MOVES FIRST BY DEFAULT.)'''

  # Check to make sure the game G fits in the universe.
  assert len(G) <= dim

  #Create a state matrix.
  #If the first play is down, get the dual game
  state = np.zeros((4, dim, dim), dtype=np.int8)
  if first_move == DOWN:
      state[3,:,:] = 1
      G = -G
  
  cr = G.cover_relations
  tc = dag.transitive_closure(cr)
  color_dict = G.coloring

  #Get a node and place a 1 on the diagonal of the face corresponding to its color in state
  for node in tc:
    color = color_dict[node]
    state[1 - color, node, node] = 1
    
    #Now place node's upset (covers) in the state
    #Each cover gets a 1 placed on the face of its color, in row node and column cover. 
    for cover in tc[node]:
      color = color_dict[cover]
      state[1 - color, node, cover] = 1

  return state

def take_action(state, a):
    '''Call this method once an action a is chosen to remove the upset of a and 
    change the perspectivnp.diagonal(state[i]) for i in range(state.shape[0])e of the game to the next player'''
    assert a in valid_actions(state)
    
    ##### remove node 'a' and its upset in 'state' ####
    take_a = copy.deepcopy(state)
    # rows is a 2-D aarray in which the (i,j) entry tells us to zero 
    # out the jth column in the ith array of 'take_a'
    rows = take_a[[0,1,2],[a],:]
    # loop to move said columns
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
        # get node label
        node = row[1]
        # swap node color
        color = 2-row[0]
        # sum over all node-th rows (gives the upset of the node)
        # which is to be the downset of the node in next state
        col = sum(take_a[[0,1,2],[node],:])
        # set the downset of node in 'next_state'
        next_state[[color],:,[node]] = col
    return next_state

def valid_actions(state):
    '''A method returning (as a tensor) the valid actions for the current player.'''
    
    #Get the diagonals of the first and second matrices and sum their columns
    d = np.sum(np.diagonal(state[[0,1]], 0,2),0)
    
    return np.nonzero(d)[0]

def is_terminal_state(state):
    ''' Returns True if current player has lost the game. (Check before current
     player takes action!)
    '''
    if len(valid_actions(state)) == 0:
      return True
    else:
      return False
  
     
