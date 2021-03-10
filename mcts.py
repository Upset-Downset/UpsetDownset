"""
@author: Charles Petersen and Jamison Barsotti
"""

from config import *
import numpy as np
import math

class PUCTNode(object):
    '''Abstract class for construction of a node in a polynomial 
    upper confidence tree (PUCT).
    '''
    def __init__(self, state, parent=None, action=None):
        '''
        Parameters
        ----------
        state : GameState
           state representation of an upset-downset game.
        parent : PUCTNode or None, optional
            The parent node to 'self'. I.e., the node containg the previous 
            state and its action stats. If root, then parent is None. The 
            default is None.
        action : int (nonnegative), optional
            edge chosen from parent node leading to 'self'. I.e., the action 
            taken in the parent state which lead to 'state'. If 
            node is root, then action is NONE. The default is None.
        Returns
        -------
        None.
        '''
        self.state = state
        self.action = action 
        self.parent  = parent  
        # wether node has been expanded in search
        self.is_expanded = False 
        # children of 'self' keyed by action 
        self.edges = {} 
        # 1-D numpy array: 
        # action a --> P(self.state,a) for all actions a
        # to be updated with probs from NET upon expansion of node
        self.edge_probs = np.zeros(MAX_NODES, dtype=np.float32)
        # 1-D numpy array: 
        # action a --> W(self.state,a) for all actions a
        # to be updated upon backup
        self.edge_values = np.zeros(MAX_NODES, dtype=np.float32) 
        # 1-D numpy array: 
        # action a --> N(self.state,a) for all actions a
        # to be updated upon backup
        self.edge_visits = np.zeros(MAX_NODES, dtype=np.float32)
        # list to store valid actions from self.state
        # to be updated upon expansion of node
        self.valid_actions = []
    
    def add_edge(self, a):
        ''' Add  edge/child node to tree (if not already present) and
        return the corresponding child node.
        
        Parameters
        ----------
        a : int (nonnegative)
            action leading to next state.
        Returns
        -------
        PUCTNode
            child node along edge 'a'.
        '''
        if a not in self.edges:
            next_state = self.state.take_action(a)
            self.edges[a] = PUCTNode(next_state, parent=self, action=a)
        return self.edges[a]
    
    def add_dirichlet_noise(self, epsilon=DIRICHLET_EPS, alpha=DIRICHLET_ALPHA):
        ''' Returns edge probs with Dirichlet noise added.
        
        Parameters
        ----------
        epsilon : float
            constant between 0 and 1. Partially controls the level
            of exploration from a root node. The default is 0.25.
        alpha : float
            constant between 0 and 1. Parameter of symmetric Dirichelt 
            distribution. Partially controls the level of exploration
            from a root node. The default is 1.
        Returns
        -------
        None.
        '''
        probs = np.copy(self.edge_probs)
        # if self is root, add dirichlet noise
        probs = (1 - epsilon)*probs + \
            epsilon*np.random.dirichlet([alpha]*MAX_NODES)
            
        return probs
    
    def PUCT_action(self, c_puct=C_PUCT):
        ''' Returns next action via the PUCT formula.
        
        Parameters
        ----------
        c_puct : float, optional
            constant which partially controls level of exploration. 
            The default is 1.0.
        Returns
        -------
        puct_action : int (nonnegative)
            action chosen by PUCT formula.
        '''
        # if self is root, add dirichlet noise
        probs = self.add_dirichlet_noise() \
            if self.parent is None else self.edge_probs
        # mean action values
        # 1-D numpy array: 
        # action a --> Q(self.state, a) for all actions a
        Q = self.edge_values / (1 + self.edge_visits)
        # controls exploration
        # 1-D numpy array: 
        # action a --> U(self.state, a) for all actions a
        U = c_puct*math.sqrt(np.sum(self.edge_visits)) * (
            probs / (1 + self.edge_visits))
        # PUCT 
        # 1-D numpy array: 
        # action a --> PUCT(self.state, a) for all actions a
        PUCT = Q + U
        # need set PUCT(self.state, a) to negative infinity 
        # for all invalid actions a from self.state
        invalid_actions = list(set(range(MAX_NODES)) 
                               - set(self.valid_actions))
        PUCT[invalid_actions] = -np.Inf

        return int(np.argmax(PUCT))
    
    def find_leaf(self):
        ''' Returns a leaf or terminal node in tree.
        
        Returns
        -------
        current : PUCTNode
            a node which has not yet been expanded. In the case of a
            terminal node: may have been visited already, but we do not 
            expand terminal nodes.
        '''
        current = self
        # find a leaf, taking PUCT recommended action as we go
        while current.is_expanded:
            next_action = current.PUCT_action()
            current = current.add_edge(next_action)
        return current
    
    def expand(self, probs, actions):
        ''' Expand node: update valid_actions, edge_probs and is_expanded
        attributes.
        
        Parameters
        ----------
        probs : 1-D numpy array of shape (gs.UNIV,)
            prior probabilities of actions queried from neural network.
        actions : list
            valid actions from 'self.state'.
        Returns
        -------
        None.
        '''
        self.valid_actions = actions
        self.edge_probs = probs
        self.is_expanded = True
    
    def backup(self, value):
        ''' Backup edge visit counts and action values in tree after MCTS.
        
        Parameters
        ----------
        value : int
            wether current game is a winning position from the perspective
            of the current player (resp. +1, -1 ). If node is terminal 
            we derive 'value' from the rules of upset-downset. Otherwise, 
            'value' is queried from the neural network.
        Returns
        -------
        None.
        '''
        current = self
        # as the value is from the current players persepective
        # we need to flip the sign as we backup the tree
        sgn = -1
        # backup values until root
        while current.parent != None:
            current.parent.edge_visits[current.action] += 1
            current.parent.edge_values[current.action] += sgn*value 
            sgn *= -1
            current = current.parent
    
    def to_root(self):
        ''' Return as root node. After an MCTS has been performed 
        and an action has been taken from root via the MCTS_policy(), 
        we make declare the new root and discard the portion of the tree that 
        is no longer needed.
        
        Returns
        -------
        self
            PUCTNode
        '''
        self.parent = None
        self.action = None
        return self
