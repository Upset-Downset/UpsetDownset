"""
@author: Charles Petersen and jamison Barsotti
"""
from config import *
from model import AlphaZeroNet
from gameState import GameState
from mcts import PUCTNode
import numpy as np
import torch
import os
class Agent(object):
    def __init__(self, path=None, device=DEVICE):
        self._model = None
        self.path = path
        self.device = device
               
    @property
    def model(self):
        if self._model is None:
            # if no path variable is given, initialize parameters
            if self.path is None:
                Agent.initialize()
                self.path = './model_data/initial.pt'
            # initialize model
            self._model = AlphaZeroNet()
            #load params on device
            self._model.load_state_dict(
                torch.load(self.path, map_location=self.device))
            #put model on device
            self._model.to(self.device)
            
        return self._model
        
    def predict_next_move(self, game, player, search_iters):
        player_dict = {'up': UP, 'down': DOWN}
        game_state = GameState(game, player_dict[player.casefold()])
        root = PUCTNode(game_state)
        policy = self.MCTS(root, search_iters, 0)
        return np.argmax(policy)
        
    
    def predict(self, encoded_game_state, training=False):
        # query the net
        if not training:
            with torch.no_grad():
                state = torch.from_numpy(
                    encoded_game_state).float().to(self.device)
                probs, value = self.model(state)
                del state; torch.cuda.empty_cache()
                probs = probs.detach().cpu().numpy().reshape(-1)
                value = value.item()
        else:
            probs, value = self.model(encoded_game_state)
        
        return probs, value
    
    def MCTS(self, root, search_iters, temp): 
        # perform search
        for _ in range(search_iters):
            leaf = root.find_leaf()
            # if leaf is a terminal state, i.e. previous player won
            if leaf.state.is_terminal_state():
                # the value should be from the current players perspective
                value = -1
                leaf.backup(value)
            # no winner yet
            else:
                # query the agent
                probs, value = self.predict(leaf.state.encoded_state)
                # expand and backup
                actions = leaf.state.valid_actions()
                leaf.expand(probs, actions)
                leaf.backup(value)
                
        # get mcts policy
        if temp == 0:
            policy = np.zeros(MAX_NODES, dtype=np.float32)
            max_visit = np.argmax(root.edge_visits)
            policy[max_visit] = 1
        else:
            policy = ((root.edge_visits)**(1/temp))/np.sum(
                (root.edge_visits)**(1/temp))
        
        return policy
    
    def approximate_outcome(self, game, search_iters):
        self.model.eval()
        # game state from each players persepective
        up_start = GameState(game, UP)
        down_start = GameState(game, DOWN)
        
        actions = np.arange(MAX_NODES)
        outcomes = []
        
        #self play game with each player moving first
        for game_state in [up_start, down_start]:
            root = PUCTNode(game_state)
            move_count = 0
    
            # play until a terminal state is reached
            while not root.state.is_terminal_state():
                policy = self.MCTS(root, search_iters, temp=0)
                move = np.random.choice(actions, p=policy)
                root = root.edges[move]
                root.to_root()
                move_count += 1
        
            # update outcome: 'P' for second player to move 
            # and 'N' for first player to move
            out = 'P' if (move_count % 2 == 0) else 'N'
            outcomes.append(out)
        
        up_start_out, down_start_out = outcomes
        # get outcome prediction
        if up_start_out == 'P' and down_start_out == 'P':
            approx_out = 'Previous'
        elif up_start_out == 'N' and down_start_out == 'N':
            approx_out = 'Next'
        elif up_start_out == 'P' and down_start_out == 'N':
            approx_out = 'Down'
        elif up_start_out == 'N' and down_start_out == 'P':
            approx_out = 'Up'
            
        self.model.train()
        return approx_out
    
    @staticmethod
    def initialize(path=None):
        '''
        In the future this can take a path to initialize the model...
        '''
        initial_model = AlphaZeroNet()
        if not os.path.isdir('./model_data'):
                   os.mkdir('./model_data')  
        torch.save(initial_model.state_dict(),
                       './model_data/initial.pt')
        torch.save(initial_model.state_dict(),
                       './model_data/apprentice.pt')
        torch.save(initial_model.state_dict(),
                       './model_data/alpha.pt')
        
        del initial_model