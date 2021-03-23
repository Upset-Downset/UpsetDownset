"""
@author: Charles Petersen and Jamison Barsotti
"""
from config import *
from agent import Agent
from gameState import GameState
from puctNode import PUCTNode
import ray
import numpy as np
import os
import pickle

@ray.remote(num_gpus=0.075)
class SelfPlay(object):
    '''Abstract class for construction remote self-play actors.
    '''
    def __init__(self):
        '''Instantiates a self-play actor.
        
        Returns
        -------
        None.

        '''
        self.agent = Agent(path='./model_data/alpha.pt')
        
    def run(self,
            replay_buffer,
            update_signal,
            process_id,
            search_iters=SELF_PLAY_SEARCH_ITERS,
            markov_exp=SELF_PLAY_MARKOV_EXP,
            temp=TEMP, 
            temp_thrshld=TEMP_THRSHLD):
        '''Starts indefinite self-play loop. The games for self-play are 
        generated via an ongoing Markov chain as described in randomDag.py.
        The self-play processes are synchronized with one another, train 
        and evaluation processes via the 'replay_buffer' and 'update_signal', 
        respectively. 'replay_buffer' stores the self-play data and triggers 
        the start of training while 'update_signal' triggers model parameter 
        updates.
        
        Parameters
        ----------
        replay_buffer : ReplayBuffer
            remote actor for managing self-play data between self-play processes 
            and the Train process. Also carries the signal to start training.
        update_signal : UpdateSignal
            remote actor for synchronization between self-play processes and 
            evaluation processes. Triggers model parameter updates.
        process_id : int (nonnegative)
            unique identifier for the self-play process.
        search_iters : int (positve), optional
             the number of search iterations to perform during MCTS. 
             The default is SELF_PLAY_SEARCH_ITERS.
        markov_exp : float, optional
            The exponent determining the number of steps taken in 
            the markov chain in generating games for self-play.
        temp : float (nonnegative)
            partially controls exploration. If 0, the policy is deterministic 
            and the position with highest visit  from MCTS is chosen.
        temp_thrshld : int (nonnegative), optional
            The number of moves after which the policy becomes determnistic.
            I.e., temp is set to 0. (See temp, above.) The default is 
            TEMP_THRSHLD.

        Returns
        -------
        None.

        '''
        
        # put agent in evaluation mode
        self.agent.model.eval()
        # the action space...
        actions = np.arange(MAX_NODES)
        # game state generator via an ongoing Markov chain
        state_generator = GameState.state_generator(markov_exp)
        # track the number of games played
        idx = 1    
        #start indefinite self-play loop
        while True:     
            # check for parameter updates
            if ray.get(update_signal.get_update.remote(process_id)):
                print('updating self-play alpha paramaters...')
                self.agent.load_parameters(path='./model_data/alpha.pt')
                update_signal.confirm_update.remote(process_id)
            
            # get a game and play untila a terminal state is reached
            initial_state = next(state_generator)
            root = PUCTNode(initial_state)
            states = []
            policies = []
            move_count = 0
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
            # construct training data from self-play
            train_data = [(state, policy, value) for state, policy, value 
                          in zip(states, policies, values)]     
            # add training data to replay buffer
            replay_buffer.add.remote(train_data)
            # pickel training data      
            filename = f'self_play_process_{process_id}_game_{idx}'
            path = os.path.join('./self_play_data', filename)
            with open(path, 'wb') as write:
                pickle.dump(train_data, write)
            # update game index
            idx += 1