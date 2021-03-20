#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:10:25 2021

@author: charlie
"""
from config import *
from agent import Agent
from gameState import GameState
from mcts import PUCTNode
import torch
import ray
import numpy as np
import os
import pickle

@ray.remote(num_gpus=0.075)
class SelfPlay(object):
    
    def __init__(self):
        self.agent = Agent(path='./model_data/alpha.pt')
        self.update = False
        
    def trigger_update(self):
        self.update = True
        return self.update
    

    def run(self,
            replay_buffer,
            update_signal,
            process_id,
            search_iters=SELF_PLAY_SEARCH_ITERS,
            markov_exp=SELF_PLAY_MARKOV_EXP,
            temp=TEMP, 
            temp_thrshld=TEMP_THRSHLD):
        
        # setup gameplay
        self.agent.model.eval()
        actions = np.arange(MAX_NODES)
        state_generator = GameState.state_generator(markov_exp)
        idx = 1
        #start indefinite self-play loop
        while True:
            
            # check for parameter updates after evaluation
            if ray.get(update_signal.get_update.remote(process_id)):
                print('updating self-play alpha paramaters...')
                self.agent.model.load_state_dict(
                    torch.load('./model_data/alpha.pt'))
                update_signal.confirm_update.remote(process_id)
            
            initial_state = next(state_generator)
            root = PUCTNode(initial_state)
            states = []
            policies = []
            move_count = 0

            # play until a terminal state is reached
            while not root.state.is_terminal_state():
                t = temp if move_count <= temp_thrshld else 0
                policy = self.agent.MCTS(root, search_iters, t)
                move = np.random.choice(actions, p=policy)
                states.append(root.state.encoded_state)
                policies.append(policy)
                root = root.edges[move]
                root.to_root()
                move_count += 1
        
            # update state values as seen from current players perspective
            if move_count %2 == 0:
                values = [(-1)**(i+1) for i in range(move_count)]
            else:
                values = [(-1)**i for i in range(move_count)]
                
            # construct training data from sel-play
            train_data = [(state, policy, value) for state, policy, value 
                          in zip(states, policies, values)]
            
            # add to replay buffer
            replay_buffer.add.remote(train_data)
            
            filename = f'self_play_process_{process_id}_game_{idx}'
            path = os.path.join('./self_play_data', filename)
            with open(path, 'wb') as write:
                pickle.dump(train_data, write)
            
            idx += 1