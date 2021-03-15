"""
@author: Charles Petersen and Jamison Barsotti
"""
from config import *
from gameState import GameState
from agent import Agent
from mcts import PUCTNode
from writeLock import write_to_file
import numpy as np
import torch
import pickle
import os
import ray

@ray.remote(num_gpus=0.1)
def self_play(scheduler,
              process_id,
              search_iters=SELF_PLAY_SEARCH_ITERS,
              markov_exp=SELF_PLAY_MARKOV_EXP,
              temp=TEMP, 
              temp_thrshld=TEMP_THRSHLD):
    
    print(f'Self-play process {process_id} in progress...')
    # if continuing, get apprentice and alpha for training/evaluation
    # if continuing, get apprentice and alpha for training/evaluation
    alpha = Agent(path='./model_data/alpha.pt')
    alpha.model.eval()

    actions = np.arange(MAX_NODES)
    state_generator = GameState.state_generator(markov_exp)
    game_id = 1
    
    #start self-play loop
    while True:
        # check for parameter updates after evaluation
        if ray.get(scheduler.get_signal.remote(process_id)):
            print(f'Updating self-play agent: {process_id}...')
            alpha.model.load_state_dict(
                torch.load('./model_data/alpha.pt', map_location=alpha.device))
            scheduler.reset_signal.remote(process_id)
            
        initial_state = next(state_generator)
        root = PUCTNode(initial_state)
        states = []
        policies = []
        move_count = 0

        # play until a terminal state is reached
        while not root.state.is_terminal_state():
            if move_count <= temp_thrshld:
                t = temp
            else:
                t = 0
            policy = alpha.MCTS(root, search_iters, t)
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
        
        train_data = [(state, policy, value) for state, policy, value 
                      in zip(states, policies, values)]
        
        scheduler.add_training_data.remote(train_data)
        
        filename = f'self_play_process_{process_id}_game_{game_id}'
        path = os.path.join('./self_play_data', filename)
        with open(path, 'wb') as write:
            pickle.dump(train_data, write)
            
        game_id += 1
        
@ray.remote(num_gpus=0.1)        
def eval_play(scheduler,
              num_plays=NUM_EVAL_PLAYS,
              search_iters=EVAL_PLAY_SEARCH_ITERS,
              markov_exp=EVAL_PLAY_MARKOV_EXP,
              win_ratio=WIN_RATIO):
    
    # get alpha/apprentice models, put in eval mode
    alpha = Agent(path='./model_data/alpha.pt')
    apprentice = Agent(path='./model_data/apprentice.pt')
    alpha.model.eval()
    apprentice.model.eval()
    
    # track the models
    alpha_token = 0
    apprentice_token = 1
    actions = np.arange(MAX_NODES)
    state_generator = GameState.state_generator(markov_exp)
    
    while True:
        # evaluation
        apprentice_wins = 0
        for k in range(num_plays):      
            #store states encountered
            states = []
    
            # uniformly randomly choose which model plays first
            next_move = np.random.choice([alpha_token, apprentice_token])     
        
            # play a randomly generated game of upset-downset
            game_state = next(state_generator)
            states.append(game_state.encoded_state)
            move_count = 0

            while not game_state.is_terminal_state():
                root = PUCTNode(game_state)
                policy = alpha.MCTS(root, search_iters, 0) if next_move == alpha_token \
                    else apprentice.MCTS(root, search_iters, 0)
                move = np.random.choice(actions, p=policy)
                game_state = root.edges[move].state
                states.append(game_state.encoded_state)
                next_move = 1 - next_move
                move_count += 1
            
            # decide winner
            winner = 1 - next_move
            if winner == apprentice_token:
                apprentice_wins += 1
    
        print(f'The apprentice won {apprentice_wins} evaluation games.')
        update = (apprentice_wins/num_plays) > win_ratio
        if update:
            print('The alpha is being updated...')
            ray.get(
                save_with_lock.remote(
                    apprentice, './model_data/alpha.pt'
                    )
                )
            alpha.model.load_state_dict(
                torch.load('./model_data/alpha.pt', map_location=alpha.device))
            scheduler.send_signal.remote()
        
        apprentice.model.load_state_dict(
            torch.load('./model_data/apprentice.pt', map_location=apprentice.device))