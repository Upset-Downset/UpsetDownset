"""
@author: Charles Petersen and Jamison Barsotti
"""
from gameState import GameState
from agent import Agent
from mcts import PUCTNode
import numpy as np
import torch
import ray

@ray.remote(num_gpus=0.25)
def self_play(scheduler,
              process_id,
              search_iters,
              temp, 
              temp_thrshld):
    
    print(f'Self-play process {process_id} in progress...')
    # get agent
    agent = Agent(alpha=True)
    agent.model.eval()

    actions = np.arange(GameState.NUM_ACTIONS)
    state_generator = GameState.state_generator()
    
    while True:
        # check for parameter updates after evaluation
        if ray.get(scheduler.get_signal.remote(process_id)):
            print(f'Updating self-play agent: {process_id}...')
            agent.model.load_state_dict(torch.load(
                './model_data/alpha_model.pt'))
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
            policy = agent.MCTS(root, search_iters, t)
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
        
        
def eval_play(alpha_agent,
               apprentice_agent,
               num_plays,
               search_iters,
               win_ratio):

    # put apprentice model in eval mode
    apprentice_agent.model.eval()
    
    # track the models
    alpha = 0
    apprentice = 1
    
    # evaluation
    apprentice_wins = 0
    actions = np.arange(GameState.NUM_ACTIONS) 
    state_generator = GameState.state_generator(markov_exp=2)
    for k in range(num_plays):      
        #store states encountered
        states = []
    
        # uniformly randomly choose which model plays first
        agent_to_start = np.random.choice([alpha, apprentice])     
        
        # play a randomly generated game of upset-downset
        game_state = next(state_generator)
        states.append(game_state.encoded_state)
        agent = agent_to_start
        move_count = 0
        winner = None
        
        while not game_state.is_terminal_state():
            root = PUCTNode(game_state)
            policy = alpha_agent.MCTS(root, search_iters, 0) if agent == alpha \
                else apprentice_agent.MCTS(root, search_iters, 0)
            move = np.random.choice(actions, p=policy)
            game_state = root.edges[move].state
            states.append(game_state.encoded_state)
            agent = 1 - agent
            move_count += 1
            
        # decide winner
        winner = 1 - agent_to_start if move_count %2 == 0 else agent_to_start
        if winner == apprentice:
            apprentice_wins += 1 
    
    print(f'The apprentice won {apprentice_wins} evaluation games.')
    
    # put apprentice model back in training mode   
    apprentice_agent.model.train()

    return (apprentice_wins/num_plays) > win_ratio