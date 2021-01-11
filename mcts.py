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
import torch

class PUCTNode(object):
    def __init__(self, state, action=None, parent=None):
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
        self.edge_probs = np.zeros([gs.UNIV], dtype=np.float32)
        # 1-D numpy array: action a --> W(self.state,a) for all actions a. To be updated upon backup
        self.edge_values = np.zeros([gs.UNIV], dtype=np.float32) 
        # 1-D numpy array: action a --> N(self.state,a) for all actions a. To be updated upon backup
        self.edge_visits = np.zeros([gs.UNIV], dtype=np.float64)
        # list to store valid actions from self.state. To be updated upon expansion of node
        self.valid_actions = []
    
    def add_edge(self, a):
        # add edge/node if not already there
        if a not in self.edges:
            # take action a in self.state
            next_state = gs.take_action(self.state, a)
            # update self.children/create new node in tree
            self.edges[a] = PUCTNode(next_state, action=a, parent=self)
        return self.edges[a]
    
    def PUCT_action(self, c_puct = 1.0, eps = 0.25, eta = 0.03):
        probs = copy.deepcopy(self.edge_probs)
        # if self is root node add dirichlet noise
        if self.parent == None:
            probs = (1-eps)*probs + eps*np.random.dirichlet([eta]*gs.UNIV)
        # 1-D numpy array: action a --> Q(self.state, a) for all actions a
        Q = self.edge_values / (1 + self.edge_visits)
        # 1-D numpy array: action a --> U(self.state, a) for all actions a
        U = c_puct*math.sqrt(np.sum(self.edge_visits)) * (
            probs / (1 + self.edge_visits))
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
            next_action = current.PUCT_action()
            current = current.add_edge(next_action)
        return current
    
    def expand(self, probs, actions):
        self.is_expanded = True
        self.valid_actions = actions
        self.edge_probs = probs
    
    def backup(self, value):
        current = self
        sgn = -1
        while current.parent != None:
            current.parent.edge_visits[current.action] += 1
            current.parent.edge_values[current.action] += sgn*value 
            sgn *= -1
            current = current.parent
    
    def to_root(self):
        self.parent = None
        self.action = None
        return self         
            
            
                       
            
def MCTS(root, net, num_iters = 400):
    for _ in range(num_iters):
        leaf = root.find_leaf()
        # if leaf is a terminal state, i.e. previous player won
        if gs.is_terminal_state(leaf.state):
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

def MCTS_policy(root, temp):
    return ((root.edge_visits)**(1/temp))/np.sum((root.edge_visits)**(1/temp))

def self_play(initial_state, net, temp=1, tmp_thrshld=3, eps=0.01):
    ''' I THINK  THIS IS MOSTLY CORRECT. NEED SOME TESTING AND SUCH.
    '''
    states = []
    policies = []
    values = None
    train_data = None
    cur_state = initial_state
    move_count = 0
    moves = []
    actions = np.arange(gs.UNIV)
    root = PUCTNode(cur_state, action=None, parent=None)
    while not gs.is_terminal_state(cur_state):
        MCTS(root, net)
        if move_count < tmp_thrshld:
            t = temp
        else:
            t = eps
        policy = MCTS_policy(root, t)
        ### I GOT THIS ERROR A FEW TIMES 'ValueError: probabilities contain NaN' ###
        move = np.random.choice(actions, p=policy)
        states.append(cur_state)
        policies.append(policy)
        moves.append(move)
        root = root.edges[move]
        root.to_root()
        cur_state = root.state
        move_count += 1
    ### NEED TO DETERMINE WHAT TO DO FOR POLICY WHEN STATE IS TERMINAL... ###
    if move_count %2 == 0:
        values = [(-1)**(i+1) for i in range(move_count)]
    else:
        values = [(-1)**i for i in range(move_count)]
    train_data = [[state, policy, value] for state, policy, value 
                  in zip(states, policies, values)]
    return train_data, moves

def eval_play():
    '''
    SIMILAR TO self_play BUT USES TWO MCTSes AND AN INFINTESIMAL TEMP ALWAYS
    '''
    pass
