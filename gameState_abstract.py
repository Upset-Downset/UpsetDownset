#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti

NOTE: The encoded representation of an upset-dowsnet game is always 
viewed from the current players perspective (the player to move).
We use the convention that all players play from the perspective of Up. 
I.e., if it is Downs move in the game, then we change perspective so 
that Up is to move in the negtive of the game.

"""
import digraph
import UpDown as ud
import randomUpDown as rud
import utils

import numpy as np

class GameState(object):
    '''An abstract class for representing a upset-downset game in a format 
    for use with the AlphaZero algorithm.
    '''
    
    # class variables which control the shape of the encoded 
    # representation of upset-downset.
    CHANNELS = 4
    MAX_NODES = 10
    SHAPE = (CHANNELS, MAX_NODES, MAX_NODES)
    
    def __init__(self, game, current_player):
        self.game = game
        self.current_player = current_player
    
    @property
    def encoded_state(self):
        return self.encode()
        
    @encoded_state.setter
    def encoded_state(self, x):
        if self.encoded_state is None:
            self.encoded_state = x
        
    def valid_actions(self):
        '''Returns the valid actions from the current game state.
        
        Returns
        -------
        list
            nodes available to play in game state. (As seen from the 
            perspective the current player.)

        '''
        return self.game.up_nodes()
    
    def take_action(self, x):
        ''' Returns the next game state after 'x' has been played.
        
        Parameters
        ----------
        x : int (nonegative)
            valid action from game state

        Returns
        -------
        GameState
            the game state after the upset of node'x' has been removed. (From 
            and the persepctive of the player to move.)

        '''
        assert x in self.valid_actions(), 'That is not a valid action.'
        
        # remove upset of x
        option = self.game.up_play(x)
        
        # change the perspecvtive to the player to move
        negative = -option
        player_to_move = 'Down' if self.current_player == 'Up' else 'Up'
        
        return GameState(negative, player_to_move)
    
    def is_terminal(self):
        ''' Returns wether the current player has lost the game. I.e., a 
        terminal state is a game state in which the current player has no 
        valid moves. (Check before current player attempts to take an action!)

        Returns
        -------
        bool
            True if the current player has lost the game, and False otherwise.

    '''
        True if len(self.valid_actions()) == 0 else False
        
    def encode(self):
        ''' Returns the encoded representation of the game from the 
        persepective of the current player.
       
        Returns
        -------
        encoded_game : 3D-numpy array 
            the encoded representation is a 4 channel image stack. 
            The four channels C_k (for k=0,1,2,3) together 
            constitute a binary feature encoding of the state of the game
            via the adjacecny matrix of the underlying DAG, the node
            coloring, and the curent player. 
        
            The first three channels C_0, C_1, C_2 encode the presence of
            nodes (resp. blue, green, red)  and their edge relationships 
            amongst one another: for k=0,1,2 entry C_k(i,j) = 1 if node j 
            has color 1-k and node i is an ancestor of node j. The last 
            channel C_3 encodes the current player: C_3 = 0 if Up is to 
            move and 1 if Down is to move. 
     
        '''
        assert len(self.game) <= GameState.MAX_NODES, 'The game is to large.'

        # if the move is to Down, get the negative game
        # and set the last channel to a constant 1
        encoded_game = np.zeros(GameState.SHAPE, dtype=np.int8)
        if self.current_player == 'Down':
            encoded_game[3,:,:] = 1
            self.game = -self.game
  
        # get the underlying dag, its transitive closure and node coloring
        dag = self.game.dag
        tc = digraph.transitive_closure(dag)
        color_dict = self.game.coloring

        # fill in the encoded representation
        for node in tc:
          # set the diagonals of the first three channels according to
          # nodes which are present in the game and their colors 
          color = color_dict[node]
          encoded_game[1 - color, node, node] = 1
    
          # now set the remainder of each nodes upset 
          # in their proper channel, row and column according to color,
          # node and cover, respectively
          for cover in tc[node]:
            color = color_dict[cover]
            encoded_game[1 - color, node, cover] = 1
        
        return encoded_game
    
    @staticmethod
    def decode_to_game(encoded_state):
        '''Returns the upset-downset game from the 'encoded state'.
    
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
        node_info = np.diagonal(encoded_state[0:3], axis1=1, axis2=2)
        node_info = np.argwhere(node_info)
    
        # set colors of nodes present in 'state'
        colors = { node : 1 - channel for channel, node in node_info}
        # set the corresponding directed acyclic graph
        dag = {}
        for info in node_info:
            node = info[1]
            upset = sum(encoded_state[[0,1,2],node, :])
            upset[node] = 0
            upset = np.argwhere(upset).reshape(-1)
            dag[node] = list(upset)
    
        G = ud.UpDown(dag, coloring=colors)
    
        # change perspecive if its Downs turn!
        cur_player = encoded_state[3,0,0]
        if cur_player == 1:
            G = -G
        
        return G
        
    
    @staticmethod
    def initial_state(play_type, prcs_id, train_iter):
        pass
