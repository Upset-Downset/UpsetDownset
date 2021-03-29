"""
@author: Charles Petersen and Jamison Barsotti
"""
from config import *
import digraph
from randomDag import uniform_random_dag
from upDown import UpDown
import numpy as np

class GameState(object):
    '''An abstract class for representing upset-downset games in a format 
    for use with the AlphaZero algorithm.
    '''
    
    def __init__(self, game, current_player):
        ''' Initializes a GameState object.

        Parameters
        ----------
        game : UpDown
            a game of upset-downset.
        current_player : int (0 or 1)
            encoded representation of the p[layer to move: 0 if Up is to move 
            and 1 if Down is to move. (See UP and DOWN in config.py)

        Returns
        -------
        None.

        '''
        self.game = game 
        self.current_player = current_player
        self._encoded_state = None
        
    @property
    def encoded_state(self):
        if self._encoded_state is None:
            self._encoded_state = self.encode()
        return self._encoded_state
        
        
    def valid_actions(self):
        '''Returns the valid actions.
        
        Returns
        -------
        list
            nodes available to play.
        '''
        
        return self.game.up_nodes() if self.current_player == UP \
            else self.game.down_nodes()
    
    def take_action(self, x):
        ''' Returns the GameState after node 'x' has been played.
        
        Parameters
        ----------
        x : int (nonegative)
            a valid action.

        Returns
        -------
        GameState
            the GameState after the node 'x' has been played. 
        '''
        # make sure the action is valid!
        assert x in self.valid_actions(), f'{x} is not a valid action.'   
        # remove upset of x
        option = self.game.up_play(x) if self.current_player == UP \
            else self.game.down_play(x)    
        # set the player to move i.e., the new current player
        player_to_move = DOWN if self.current_player == UP \
            else UP
        
        return GameState(option, player_to_move)
    
    def is_terminal_state(self):
        ''' Returns wether the current player has lost the game. A 
        terminal state is a game state in which the current player has no 
        valid moves. (Check before current player attempts to take an action!)

        Returns
        -------
        bool
            True if the current player has lost the game, and False otherwise.

    '''
        return not len(self.valid_actions())
    
    def plot(self):
        ''' Plots the underlying upset-downset game from the perepsctive of 
        the current player.
        
        Returns
        -------
        None.

        '''
        # put game in perspective of current player
        in_perspective = self.game if self.current_player == UP \
            else -self.game
        in_perspective.plot()
        
    def encode(self):
        ''' Returns the encoded representation of the GameState. Called 
        internally once the encoded representation is needed, @property
        sets the corresponding attribute self._encoded_state.
        
        The encoded representation of an upset-dowsnet game is always 
        viewed from the current players perspective (the player to move).
        We use the convention that all players play from the perspective of Up. 
        I.e., if it is Downs move in the game, then we change perspective so 
        that Up is to move in the negtive of the game.
       
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
        # make sure the game is not too large for the model
        assert len(self.game) <= MAX_NODES, 'The game is too large.'
        # initialize an empty array, and set the last channel to the 
        # current player
        encoded_state = np.zeros(ENCODED_STATE_SHAPE, dtype=np.int8)
        encoded_state[3,:,:] = self.current_player       
        # get the underlying dag, its transitive closure and node coloring.
        # make sure viewing board from correct persepective
        game_state = -self.game if self.current_player == DOWN \
            else self.game
        dag = game_state.dag
        tc = digraph.transitive_closure(dag)    
        color_dict = game_state.coloring
        # encode the game
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
    def decode(encoded_state):
        '''Returns the decoded GameState from the 'encoded state'.
    
        Parameters
        ----------
        state : numpy array
            encoded representation of an upset-downset game.
        
        Returns
        -------
        GameState
            the decoded GameState corresponding to 'encoded_state'.
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
        game = -game if current_player == DOWN else game
        
        return GameState(game, current_player)
    
    @staticmethod
    def state_generator(markov_exp, color_dist=RGB_DIST):
        ''' Returns a generator of GameStates. Proceeds along the Markov chain 
        described in randomDag.py starting from the empty dag to produce 
        GameStates (almost) uniformly randomly.

        Parameters
        ----------
        markov_exp : float, optional
            The exponent determining the number of steps taken in 
            the markov chain. 
        color_dist : tuple, optional
            the distribution of node colorings for the generated games. 
            The first entry is the proportion of red-green-blue games and 
            the second entry is the proportion of all green games. 
            The default is RGB_DIST.

        Yields
        ------
        random_state : GameState
            has MAX_NODES number of nodes, and the current player is chosen 
            uniformly randomly.
        '''
        # start the MArkov chain with the empty dag
        start_markov = {i: [] for i in range(MAX_NODES)}
        #proceed to randomly generate GAmeStates
        while True:
            #unifrmly randomly pick who moves first
            random_player = np.random.choice([UP, DOWN])
            # (almost) unfirmly ranbdomly get the DAG udelying the game
            random_dag = uniform_random_dag(
                MAX_NODES, 
                exp=markov_exp,
                X0=start_markov
                )
            # decide if the game will have all green nodes or not. and 
            # produce the coloring
            RGB = np.random.choice([True, False], p=RGB_DIST)
            random_colors = {i: np.random.choice([-1,0,1]) 
                             for i in range(MAX_NODES)} if RGB else \
                {i: 0 for i in range(MAX_NODES)}
            # instantiate the game and the corresponding state
            random_game = UpDown(random_dag, random_colors)
            random_state = GameState(random_game, random_player)
            # update where to start for the next step in the Markov chain
            start_markov = random_dag
            
            yield random_state
