"""
@author: Charles Petersen and Jamison Barsotti
"""
from config import *
from agent import Agent
from gameState import GameState
from puctNode import PUCTNode
import numpy as np
import ray
import pickle 
import os

@ray.remote(num_gpus=0.075)
class Evaluation(object):
    '''Abstract class for construction of remote evaluation actors.
    '''
    def __init__(self):
        ''' Instantiates a Evaluation actor
        Returns
        -------
        None.

        '''
        self.alpha_agent = Agent(path='./model_data/alpha.pt')
        self.apprentice_agent= Agent(path='./model_data/apprentice.pt')
        
    def update_alpha_parameters(self):
        self.apprentice_agent.save_parameters(
            path='./model_data/alpha.pt'
            )    

    def run(self,
            evaluation_id,
            num_plays=PLAYS_PER_EVAL,
            search_iters=EVAL_PLAY_SEARCH_ITERS,
            markov_exp=EVAL_PLAY_MARKOV_EXP):
        '''Starts an evaluation. The evaluation process is synchronized 
        with the self-play processes and evaluation processes via instances of  
        UpdateSignal and AsyncSignal, respectively, in the main script: 
        asyn_training.py. The UpdateSignal triggers an update in each of the 
        self-play processes if the total number of apprentice wins to total 
        evaluation games surpasses the declared win ratio while the AsyncSignal
        triggers the evaluation processes.

        Parameters
        ----------
        evaluation_id : int (nonnegative)
            unique identifier for the evaluation process
        num_plays : int (positive), optional
            The numnber of evaluation games to play. The default is 
            PLAYS_PER_EVAL.
        search_iters : int (positive), optional
            the number of search iterations to perform during MCTS. 
            The default is EVAL_PLAY_SEARCH_ITERS.
        markov_exp : float, optional
            The exponent determining the number of steps taken in 
            the markov chain in generating games for evaluation.

        Returns
        -------
        apprentice_wins : int (nonegative)
            the number of apprentice wins.

        '''
        
        print(f'evaluation {evaluation_id} in progress...')
        # put models in eval mode...
        self.alpha_agent.model.eval()
        self.apprentice_agent.model.eval()     
        # setup gameplay
        alpha = 0
        apprentice = 1
        actions = np.arange(MAX_NODES)
        state_generator = GameState.state_generator(markov_exp)
        apprentice_wins = 0
        # start evaluation game play       
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
            # pickle the evaliuation data
            filename = f'evaluation_process_{evaluiation_id}_game_{i+1}'
            path = os.path.join('./evaluation_data', filename)
            with open(path, 'wb') as write:
                pickle.dump(states, write)                
        
        return apprentice_wins