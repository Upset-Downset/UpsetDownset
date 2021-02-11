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
import utils
import digraph
import randomUpDown as rud

import numpy as np

class GameState(object):
    '''An abstract class for representing a upset-downset game in a format 
    for use with the AlphaZero algorithm.
    '''
    
    # class variables which control the shape of the encoded 
    # representation of upset-downset.
    STATE_SHAPE = (4,10,10)
    NUM_ACTIONS = 10
    UP = 0
    DOWN = 1
    
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
        '''Returns the valid actions from the game state.
        
        Returns
        -------
        list
            nodes available to play in game state. (As seen from the 
            perspective the current player.)

        '''
        return self.game.up_nodes()
    
    def take_action(self, x):
        ''' Returns the next game state after node 'x' has been played.
        
        Parameters
        ----------
        x : int (nonegative)
            valid action from game state

        Returns
        -------
        GameState
            the game state after the current player has played node 'x'. In 
            the returned game state the move on node 'x' has been made and 
            the persepective has been changed to the next player to move.

        '''
        assert x in self.valid_actions(), 'That is not a valid action.'
        
        # remove upset of x
        option = self.game.up_play(x)
        
        # change the perspecvtive to the player to move
        option = -option
        player_to_move = GameState.DOWN \
            if self.current_player == GameState.UP  else GameState.UP
        
        return GameState(option, player_to_move)
    
    def is_terminal_state(self):
        ''' Returns wether the current player has lost the game. I.e., a 
        terminal state is a game state in which the current player has no 
        valid moves. (Check before current player attempts to take an action!)

        Returns
        -------
        bool
            True if the current player has lost the game, and False otherwise.

    '''
        return True if len(self.valid_actions()) == 0 else False
    
    def plot(self):
        ''' Plots the upset-downset game underlying the game state from the 
        absolute perspective.
        
        Returns
        -------
        None.

        '''
        # if looking at the game from Downs perspective, then change 
        # perspective
        game = self.game if \
            self.current_player == GameState.UP else -self.game
            
        # plot the game   
        game.plot()
        
    def encode(self):
        ''' Returns the encoded representation of the game state.
       
        Returns
        -------
        encoded_state : 3D-numpy array 
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
        assert len(self.game) <= GameState.NUM_ACTIONS, 'The game is too large.'

        # if the move is to Down, get the negative game
        # and set the last channel to a constant 1
        encoded_state = np.zeros(GameState.STATE_SHAPE, dtype=np.int8)
        if self.current_player == GameState.DOWN:
            encoded_state[3,:,:] = 1
            
        # we can actually do all of this faster by building the encoded 
        # state while finding the trasnisitive closure...too lazy
        
        # get the underlying dag, its transitive closure and node coloring
        dag = self.game.dag
        tc = digraph.transitive_closure(dag)    
        color_dict = self.game.coloring

        # fill in the encoded representation
        for node in tc:
          # set the [node,node] entry in the diagonal of the proper 
          # channel according to the color of node.
          color = color_dict[node]
          encoded_state[1 - color, node, node] = 1
    
          # now for each adjacent node set the off diagonal entry 
          # in the proper channel (adjacent node color), row (node) and 
          # column (adjacent_node)
          for adjacent_node in tc[node]:
            color = color_dict[adjacent_node]
            encoded_state[1 - color, node, adjacent_node] = 1
            
        return encoded_state       
    
    @staticmethod
    def initial_state_generator(prcs_id, train_iter):
        start = utils.get_latest_markov(prcs_id, train_iter)
        while True:
            random_player = np.random.choice([GameState.UP, GameState.DOWN])
            random_game = rud.RandomGame(GameState.NUM_ACTIONS, 
                                         RGB=True, 
                                         start=start)
            random_state = GameState(random_game, random_player)
            start = random_game.dag
            yield random_state
