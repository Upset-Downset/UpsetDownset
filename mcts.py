#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:20:55 2021

@author: Charlie Petersen and Jamison Barsotti
"""

import gameState as gs
import numpy as np
import random
import copy
import math
import collections
import torch

class MCTNode(object):
    def __init__(self, state, action, parent=None):
        # encoded game state
        self.state = state
        # action leading from parent.state to self.state
        self.action = action 
        # MCTNode unless node is root, in which case parent is None
        self.parent  = parent  
        # bool, wether node has been expanded in search
        self.is_expanded = False 
        # dict to store edges/children nodes: MCTNodes keyed by action 
        self.edges = {} 
        # 1-D numpy array: action a --> P(self.state,a) for all actions a. To be updated with probs 
        # UpDownNet(self.state, a) upon expansdion of node
        self.prior_probs = np.zeros([gs.UNIV], dtype=np.float32)
        # 1-D numpy array: action a --> W(self.state,a) for all actions a. To be updated upon backup
        self.total_values = np.zeros([gs.UNIV], dtype=np.float32) 
        # 1-D numpy array: action a --> N(self.state,a) for all actions a. To be updated upon backup
        self.number_visits = np.zeros([gs.UNIV], dtype=np.float32)
        # list to store valid actions from self.state. To be updated upon expansion of node
        self.valid_actions = []
    
    def add_edge(self, a):
        # add edge/node if not already there
        if a not in self.edges:
            # take action a in self.state
            next_state = gs.take_action(self.state, a)
            # update self.children/create new node in tree
            self.edges[a] = MCTNode(next_state, a, parent=self)
        return self.edges[a]
    
    def get_action(self, c_puct = 1.0, eps = 0.25, eta = 0.03):
        probs = copy.deepcopy(self.prior_probs)
        # if self is root node add dirichlet noise
        if self.parent == None:
            probs = (1-eps)*probs + eps*np.random.dirichlet([eta]*gs.UNIV)
        # 1-D numpy array: action a --> Q(self.state, a) for all actions a
        Q = self.total_values / (1 + self.number_visits)
        # 1-D numpy array: action a --> U(self.state, a) for all actions a
        U = c_puct*math.sqrt(np.sum(self.number_visits)) * (
            probs / (1 + self.number_visits))
        # 1-D numpy array: action a --> PUCT(self.state, a) for all actions a
        PUCT = Q + U
        # set PUCT(self.state, a) to negative infinity for all invalid actions a from self.state
        invalid_actions = list(set(range(gs.UNIV)) - set(self.valid_actions))
        PUCT[invalid_actions] = -np.Inf
        # get the action recommended by PUCT algorithm
        puct_action = int(np.argmax(PUCT))
        return puct_action
    
    def find_leaf(self):
        # start at current node, i.e. root
        current = self
        # select leaf, taking PUCT recommended action as we go
        while current.is_expanded:
            next_action = current.get_action()
            current = current.add_edge(next_action)
        return current
    
    def expand(self, probs, actions):
        self.is_expanded = True
        self.valid_actions = actions
        self.prior_probs = probs
    
    def backup(self, value):
        current = self
        sgn = -1
        while current.parent is not None:
            current.parent.number_visits[current.action] += 1
            current.parent.total_values[current.action] += sgn*value 
            sgn *= -1
            current = current.parent
            
    def flip_values(self):
        '''
        This method should flip the sign of all the values in the total_values attribute for
        self and all children of self. This will enable us to re-use MCTS data from earlier MCTSes. 
        Can use Breadth-First Search?
        '''
        que = collections.deque()
        que.append(self)
        while que:
            node = que.popleft()
            node.total_values = {a:-node.total_values[a] for a in node.total_values}
            for a in node.edges:
                que.append(node.edges[a])
    
    
    def to_root(self):
        '''
        This method uses the  the flip_values_method and discard method to repurpose the MCTS from previous 
        iteration of MCTS for the next iteratin of MCTS. I.e., flip all values, discard 
        unused part of tree and make the selection from the previous MCTS policy the new root.
        '''
        self.parent = None
        self.action = None
        self.flip_values()            
            
            
                       
            
def MCTS(state, num_iters, net):
    root = MCTNode(state, action=None, parent=None)
    for i in range(num_iters):
        leaf = root.find_leaf()
        # if leaf is a terminal state, i.e. previous player won
        if gs.previous_player_won(leaf.state) == True:
            # the value should be from the current players perspective
            value = -1
            leaf.backup(value)
        # no winner yet
        else:
            ### WE DONT HAVE A NET YET ###
            #encoded_leaf_state = torch.from_numpy(leaf.state).float()
            #probs, value = net(encoded_leaf_state)
            #probs = probs.detach().cpu().numpy().reshape(-1)
            #value = value.item()
            #### WE DONT HAVE A NET YET ###
            probs = np.random.dirichlet([1.0]*gs.UNIV)
            value = random.choice([-1,1])
            actions = gs.valid_actions(leaf.state)
            leaf.expand(probs, actions)
            leaf.backup(value)
    return root

def MCTS_policy(root, temp=1):
    return ((root.number_visits)**(1/temp))/np.sum(root.number_visits**(1/temp))

def self_play():
    pass

def eval_play():
    pass
