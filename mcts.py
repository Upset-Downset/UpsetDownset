#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import gameState as gs
import numpy as np
import copy
import math
import torch

class PUCTNode(object):
    '''Abstract class for construction of a node in a polynomial 
    upper confidence tree (PUCT).
    '''
    def __init__(self, state, parent=None, action=None):
        '''
        Parameters
        ----------
        state : 3D-numpy array of shape (4, UNIV, UNIV)
            encoded representation of an upset-downset game.
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
        self.edge_probs = np.zeros([gs.UNIV], dtype=np.float32)
        # 1-D numpy array: 
        # action a --> W(self.state,a) for all actions a
        # to be updated upon backup
        self.edge_values = np.zeros([gs.UNIV], dtype=np.float32) 
        # 1-D numpy array: 
        # action a --> N(self.state,a) for all actions a
        # to be updated upon backup
        self.edge_visits = np.zeros([gs.UNIV], dtype=np.float32)
        # list to store valid actions from self.state
        # to be updated upon expansion of node
        self.valid_actions = []
    
    def add_edge(self, a):
        ''' Add  edge/child node to tree (if not already present) and
        return the corresponding child node.
        
        Parameters
        ----------
        a : int (nonnegative)
            action leading from self.state to next state.

        Returns
        -------
        PUCTNode
            child node under self along edge 'a'.

        '''
        if a not in self.edges:
            next_state = gs.take_action(self.state, a)
            self.edges[a] = PUCTNode(next_state, parent=self, action=a)
        return self.edges[a]
    
    def PUCT_action(self, c_puct = 1.0, eps = 0.25, eta = 1.0):
        ''' Returns next action via the PUCT formula.
        
        Parameters
        ----------
        c_puct : float, optional
            constant which partially controls level of exploration. 
            The default is 1.0.
        eps : float, optional
            constant between 0 and 1. Partially controls the level
            of exploration from a root node. The default is 0.25.
        eta : float, optional
            constant between 0 and 1. Parameter of symmetric Dirichelt 
            distribution. Partially controls the level of exploration
            from a root node. The default is 1.

        Returns
        -------
        puct_action : int (nonnegative)
            action chosen by PUCT formula.

        '''
        probs = copy.deepcopy(self.edge_probs)
        # if self is root, add dirichlet noise
        if self.parent == None:
            probs = (1-eps)*probs \
                + eps*np.random.dirichlet([eta]*gs.UNIV)
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
        invalid_actions = list(set(range(gs.UNIV)) 
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
            prior probabilities of actions from 'self.state'
            queried from neural network.
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
            wether self.state is a winning position from the perspective
            of the current player (resp. +1, -1 ). If 'self' is terminal 
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
        ''' Returns 'self' as root node. After an MCTS has been perfomred 
        and an action has been taken leading to 'self' via the MCTS_policy(), 
        we make 'self' the new root (and hopefully discard the remainder of 
        the tree.)
        
        Returns
        -------
        self
            PUCTNode

        '''
        self.parent = None
        self.action = None
        return self         
            
            
def MCTS(root, net, device, search_iters):
    '''Performs an MCTS from 'root' and returns 'root'.

    Parameters
    ----------
    root : PUCTNode
        root of tree.
    net : UpDownNet
        model used for agent.
    device : str
        the device to run the model on ('cuda' if available, else 'cpu').
    search_iters : int (nonnegative)
        the number of iterations of search to be performed. 
    Returns
    -------
    root : PUCTNode
        root of tree with edge visit counts and action values 
        from MCTS updated via backup. 

    '''
    for _ in range(search_iters):
        leaf = root.find_leaf()
        # if leaf is a terminal state, i.e. previous player won
        if gs.is_terminal_state(leaf.state):
            # the value should be from the current players perspective
            value = -1
            leaf.backup(value)
        # no winner yet
        else:
            # query the net
            encoded_leaf_state = torch.from_numpy(
                leaf.state).float().reshape(1, 4, gs.UNIV, gs.UNIV).to(device)
            probs, value = net(encoded_leaf_state)
            del encoded_leaf_state; torch.cuda.empty_cache()
            probs = probs.detach().cpu().numpy().reshape(-1)
            value = value.item()
            # expand and backup\
            actions = gs.valid_actions(leaf.state)
            leaf.expand(probs, actions)
            leaf.backup(value)
            
    return root

def MCTS_policy(root, temp):
    ''' Returns the probabilities of choosing an edge from 'root'. Depending 
    on 'temp', probabilities are either proprtional to the exponential edge 
    visit count or deterministic, choosing the edge with highest visit count.
    
    Parameters
    ----------
    root : PUCTNode
        root of tree
    temp : float
        controls exploration. If 0, the policy is deterministic and the
        edge with highest visit count is chosen. 

    Returns
    -------
    policy : 1-D numpy array of shape (gs.UNIV,)
        probabilities of chosing edges from 'root'

    '''
    if temp == 0:
        policy = np.zeros(gs.UNIV, dtype=np.float32)
        max_visit = np.argmax(root.edge_visits)
        policy[max_visit] = 1
    else:
        policy = ((root.edge_visits)**(1/temp))/np.sum((
            root.edge_visits)**(1/temp))
    return policy
    
def self_play(initial_state, net, device, search_iters=800, temp=1, tmp_thrshld=3):
    ''' Returns training data after self-play staring from 'initial state'.
    
    Parameters
    ----------
    initial_state : 3D-numpy array of shape (4, UNIV, UNIV)
        encoded representation of an upset-downset game.
    net : UpDownNet
        model used for agent.
    device : str
        the device to run the model on ('cuda' if available, else 'cpu').
    search_iters : int (nonnegative), optional
         the number of iterations of  MCTS to be performed for each turn. 
        The default is 800.
    temp : float, optional
        controls exploration in move choice until the temperature 
        threshold has been surpassed. The default is 1.
    tmp_thrshld : int (nonnegative), optional
        controls the number of moves in play until actions are chosen 
        deterministically via their visit count in MCTS. The default is 3.
    Returns
    -------
    train_data : list
        list of orderd triples (state, policy, value) encountered 
        during self play. For each state visited during self-play we 
        also record the MCTS policy found at that state and the value of 
        the state as determined by the outcome of the self-play from the 
        current players perspective. We do not include terminal states.
    '''
    states = []
    policies = []
    move_count = 0
    actions = np.arange(gs.UNIV)
    root = PUCTNode(initial_state)
    
    # play until a terminal state is reached
    while not gs.is_terminal_state(root.state):
        MCTS(root, net, device, search_iters)
        if move_count <= tmp_thrshld:
            t = temp
        else:
            t = 0
        policy = MCTS_policy(root, t)
        move = np.random.choice(actions, p=policy)
        states.append(root.state)
        policies.append(policy)
        root = root.edges[move]
        root.to_root()
        move_count += 1
        
    # update state values as seen from current players perspective
    if move_count %2 == 0:
        values = [(-1)**(i+1) for i in range(move_count)]
    else:
        values = [(-1)**i for i in range(move_count)]
        
    train_data = [(state, policy, value) for state, policy, value 
                  in zip(states, policies, values)]
    
    return train_data  
