#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:46:51 2021

@author: charlie
"""
from config import *
from agent import Agent
from gameState import GameState
from mcts import PUCTNode
import numpy as np
import ray
import pickle 
import os

@ray.remote(num_gpus=0.075)
class EvalPlay(object):
    
    def __init__(self):
        self.alpha_agent = Agent(path='./model_data/alpha.pt')
        self.apprentice_agent= Agent(path='./model_data/apprentice.pt')
        
    def update_alpha_parameters(self):
        torch.save(
            self.apprentice_agent.model.state_dict(), 
            './model_data/alpha.pt'
            )        

    def run(self,
            k,
            num_plays=PLAYS_PER_EVAL,
            search_iters=EVAL_PLAY_SEARCH_ITERS,
            markov_exp=EVAL_PLAY_MARKOV_EXP):
        
        print(f'evaluation {k} in progress...')
        # put models in eval mode...
        self.alpha_agent.model.eval()
        self.apprentice_agent.model.eval()
        
        # track the models
        alpha = 0
        apprentice = 1
        actions = np.arange(MAX_NODES)
        state_generator = GameState.state_generator(markov_exp)
        apprentice_wins = 0
        
        for i in range(num_plays):      
            #store states encountered
            states = []
    
            # uniformly randomly choose which model plays first
            next_move = np.random.choice([alpha, apprentice])     
        
            # play a randomly generated game of upset-downset
            game_state = next(state_generator)
            states.append(game_state.encoded_state)

            while not game_state.is_terminal_state():
                root = PUCTNode(game_state)
                policy = self.alpha_agent.MCTS(root, search_iters, 0) \
                    if next_move == alpha \
                        else self.apprentice_agent.MCTS(root, search_iters, 0)
                move = np.random.choice(actions, p=policy)
                game_state = root.edges[move].state
                states.append(game_state.encoded_state)
                next_move = 1 - next_move
            
            # decide winner
            winner = 1 - next_move
            if winner == apprentice:
                apprentice_wins += 1
            
            filename = f'evaluation_process_{k}_game_{i+1}'
            path = os.path.join('./evaluation_data', filename)
            with open(path, 'wb') as write:
                pickle.dump(states, write)
                
        print(f'evaluation {k} had {apprentice_wins} wins.')
        
        return apprentice_wins