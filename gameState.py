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
TYPE = np.int8
UP = 0
DOWN = 1

def game_to_state(G, UNIV, to_move, TYPE):
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
  assert len(G) <= UNIV

  state = np.zeros((4, UNIV, UNIV), dtype=TYPE)
  if to_move == DOWN:
      state[3,:,:] = 1
  cr = G.cover_relations
  tc = dag.transitive_closure(cr)
  color_dict = G.coloring

  for node in tc:
    color = color_dict[node]

    if color == 1:
      state[0, node,node] = 1
    elif color == 0:
      state[1, node, node] = 1
    else:
      state[2, node, node] = 1
    
    for cover in tc[node]:
      color = color_dict[cover]
      if color == 1:
        state[0,node, cover] = 1
      elif color == 0:
        state[1, node, cover] = 1
      else:
        state[2, node, cover] = 1

  return state

def valid_actions(encoded_game):
    '''A method returning (as a tensor) the valid actions for the current player.'''
    
    #Get the diagonals of the first and second matrices and sum their columns
    d = np.sum(np.diagonal(encoded_game[[0,1]], 0,2),0)
    
    #Return the indices of the nonzero entries in d as a row tensor
    return np.nonzero(d)[0]

def take_action(state, a):
    '''Call this method once an action a is chosen to remove the upset of a and 
    change the perspective of the game to the next player'''
    assert a in valid_actions(state)
    next_state = copy.deepcopy(state)
    cur_player = next_state[3,0,0]
    
    #the (i,j) entry of rows tells us to zero out the jth column in the ith matrix 
    #of next_state 
    rows = next_state[[0,1,2],[a],:]
    
    # zero out the jth column in the ith matrix of next_state 
    for row in np.argwhere(rows):
      next_state[[row[0]],:,[row[1]]] = 0
    
    #changes next_state to next players perspective. Takes the tensor next_state:
    #transposes the dimensions 1 and 2, swaps
    #the 0th and 2nd matrices, and changes the last matrix from zeros to ones 
    #or ones to zeros'''
    
    next_state = np.transpose(next_state, [0,2,1])
    next_state[[0,2]] = next_state[[2,0]]  
    next_state[3,:,:] = 1-cur_player
    
    return next_state

def previous_player_won(state):
    ''' Returns True if current player has lost the game. (Check before current
     player takes action!)
    '''
    if len(valid_actions(state)) == 0:
      return True
    else:
      return False