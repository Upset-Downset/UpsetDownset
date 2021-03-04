"""
@author: Charles Petersen and Jamison Barsotti

NOTE: The encoded representation of an upset-dowsnet game is always 
viewed from the current players perspective (the player to move).
We use the convention that all players play from the perspective of Up. 
I.e., if it is Downs move in the game, then we change perspective so 
that Up is to move in the negtive of the game.

"""
from randomUpDown import RandomGame
from upDown import UpDown
import digraph
import numpy as np

class GameState(object):
    '''An abstract class for representing a upset-downset game in a format 
    for use with the AlphaZero algorithm.
    '''
    
    # class variables which control the shape of the encoded 
    # representation of upset-downset.
    STATE_SHAPE = (4, 20, 20)
    NUM_ACTIONS = 20
    UP = 0
    DOWN = 1
    
    def __init__(self, game, current_player):
        self.game = game 
        self.current_player = current_player
        self._encoded_state = None
        
    @property
    def encoded_state(self):
        if self._encoded_state is None:
            self._encoded_state = self.encode()
        return self._encoded_state
        
    @encoded_state.setter
    def encoded_state(self, x):
        self._encoded_state = x
        
    def valid_actions(self):
        '''Returns the valid actions from the game state.
        
        Returns
        -------
        list
            nodes available to play in game state.

        '''
        moves = self.game.up_nodes() if self.current_player == GameState.UP \
            else self.game.down_nodes()
            
        return moves
    
    def take_action(self, x):
        ''' Returns the next game state after node 'x' has been played.
        
        Parameters
        ----------
        x : int (nonegative)
            valid action from game state

        Returns
        -------
        GameState
            the game state after the node 'x' has been played. 
        '''
        assert x in self.valid_actions(), 'Not a valid action.'
        
        # remove upset of x
        option = self.game.up_play(x) if self.current_player == GameState.UP \
            else self.game.down_play(x)
        
        # set the player to move i.e., the new current player
        player_to_move = GameState.DOWN \
            if self.current_player == GameState.UP else GameState.UP
        
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
        ''' Plots the underlying upset-downset game.
        
        Returns
        -------
        None.

        '''
        self.game.plot()
        
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
        encoded_state[3,:,:] = self.current_player
            
        # we can actually do all of this faster by building the encoded 
        # state while finding the trasnisitive closure...too lazy
        
        # get the underlying dag, its transitive closure and node coloring.
        # make sure viewing board from correct persepective
        game_state = -self.game if self.current_player == GameState.DOWN \
            else self.game
        dag = game_state.dag
        tc = digraph.transitive_closure(dag)    
        color_dict = game_state.coloring

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
    def to_game_state(encoded_state):
        '''Returns the game_state from the 'encoded_state'.
    
        Parameters
        ----------
        state : 3D-numpy array of shape (4, UNIV, UNIV)
        encoded representation of an upset-downset game.
        
        Returns
        -------
        UpDown
            the decoded game state corresponding to 'encoded_state'.
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
            
        # get game from absolute perspective
        game = UpDown(dag, coloring=colors)
        current_player = encoded_state[3,0,0]
        game = -game if current_player == GameState.DOWN else game
        
        return GameState(game, current_player)
    
    @staticmethod
    def state_generator(markov_exp=1, 
                        extra_steps=0,
                        RGB=True):
        start_markov = {i: [] for i in range(GameState.NUM_ACTIONS)}
        while True:
            random_player = np.random.choice([GameState.UP, GameState.DOWN])
            random_game = RandomGame(GameState.NUM_ACTIONS, 
                                     markov_exp=markov_exp,
                                     extra_steps=extra_steps,
                                     RGB=RGB, 
                                     start=start_markov)
            random_state = GameState(random_game, random_player)
            start_markov = random_game.dag
            
            yield random_state
