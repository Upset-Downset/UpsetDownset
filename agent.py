"""
@author: Charles Petersen and jamison Barsotti
"""
from config import *
from model import AlphaZeroNet
from gameState import GameState
from puctNode import PUCTNode
import numpy as np
import torch
import os

class Agent(object):
    ''' Abstract class for constructing an agent for self-play/train/evaluation,
    playing against in real-time (see the predict_next_move() method)
    or approximating outcomes (see the approximate_oucome() method).
    '''
    def __init__(self, path=None, device=DEVICE):
        ''' Instantiates an Agent.

        Parameters
        ----------
        path : str, optional
            path to the parameters used for the agents model. If no path is 
            passed, a new agent will be initialized and the path to initial
            paramters will be assigned to the path attribute. 
            The default is None. (See the @staticmethod iniialize().) 
        device : torch.device, optional
            An available torch.device. The default is DEVICE.
            (See https://pytorch.org/docs/stable/tensor_attributes.html.)

        Returns
        -------
        None.

        '''
        self._model = None
        self.path = path
        self.device = device
               
    @property
    def model(self):
        ''' Returns the agents model on the device given in the agents device 
        attribute and parameters loaded from the path stored in the agents 
        path attribute. I no path was passed in the agents constructor a new
        model model is instantiated. (See the @staticmethod initialize().)
        
        Returns
        -------
        AlphaZeroNet
            pytorch model. (See model.py)

        '''
        if self._model is None:
            # if no path variable is given, initialize a new agent
            if self.path is None:
                Agent.initialize()
                self.path = './model_data/initial.pt'
            # model
            self._model = AlphaZeroNet()
            # load params on device
            self._model.load_state_dict(
                torch.load(self.path, map_location=self.device)
                )
            #put model on device
            self._model.to(self.device)
            
        return self._model
    
    def save_parameters(self, path=None):
        '''Save agent model parameters.
        
        Parameters
        ----------
        path : str, optional
            path to save the the parameters. If no path is passed the
            parameters will be saved to path assigned to the agents
            path attribute. The default is None.

        Returns
        -------
        None.

        '''
        path = self.path if path is None else path
        torch.save(self.model.state_dict(), path)
    
    def load_parameterse(self, path):
        '''Load parameters to agents model from 'path'. Updates the agents
        path attribute to 'path'.

        Parameters
        ----------
        path : str,
            path to the parameters to be loaded. 

        Returns
        -------
        None.

        '''
        # load parameters
        self._model.load_state_dict(
                torch.load(path, map_location=self.device)
                )
        # update agents path attribute
        self.path = path
    
    def MCTS(self, root, search_iters, temp):
        ''' Returns the Policy found from a MCTS.
        
        Parameters
        ----------
        root : PUCTNode
            root node to start MCTS.
        search_iters : int (positive)
            the number of search iterations to perform during MCTS.
        temp : float (nonnegative)
            partially controls exploration. If 0, the policy is deterministic 
            and the edge with highest visit count is chosen.

        Returns
        -------
        policy : numpy array
            root egde probabilities derived from MCTS.

        '''
        # if the root has not been expanded, then expand...
        if not root.is_expanded:
            probs, value = self.predict(leaf.state.encoded_state)
            actions = leaf.state.valid_actions()
            leaf.expand(probs, actions)
        # perform MCTS
        for _ in range(search_iters):
            leaf = root.find_leaf()
            # if leaf is a terminal state, then previous player won...
            if leaf.state.is_terminal_state():
                # the value should be from the current players perspective
                value = -1
                leaf.backup(value)
            # if no winner yet...
            else:
                # predict
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
        
    def predict_next_move(self, game, current_player, search_iters):
        ''' Returns the (deterministic) best move choice predicted by the 
        agent. This method is for use in evaluation and when playing against
        the agent (or two agents playing one another) in real-time. (E.g. via 
        the play() method of the UpDown class.)
        
        Parameters
        ----------
        game : UpDown
            a game of upset-downset.
        current_player : int or str
            the current player to move. Can be either the integers 0/1 
            (see UP/DOWN in config.py) or the strings 'up'/'down' 
            (for Up/Down resp.). 
        search_iters : int (nonnegative)
            the number of search iterations to perform during MCTS.
            

        Returns
        -------
        int
            the best action in 'game' for 'current_player' as predicted by the 
            agent via the deterministic MCTS policy (i.e., temp=0)
        '''
         # make sure the model is in evaluation mode.
        self.model.eval()
        # if passed a string for current player, convert
        if isinstance(current_player, str):
            player_dict = {'up': UP, 'down': DOWN}
            current_player = player_dict[current_player.casefold()]
        # get the MCTS policy
        game_state = GameState(game, current_player)
        root = PUCTNode(game_state)
        policy = self.MCTS(root, search_iters, temp=0)
        
        return np.argmax(policy)
        
    
    def predict(self, encoded_game_state, training=False):
        ''' Returns the polcy and value predictions from the agents model.
        
        Parameters
        ----------
        encoded_game_state : nump array
            encoded state of an upset-downset game.
        training : bool, optional
            True if the prediction is to be made during training and False 
            otehrwise. The default is False.

        Returns
        -------
        probs : numpy array or torch tensor (depending on training)
            policy predicted by the agenst model.
        value : float or torch array (depending on training )
            the game value predicted by the agnets model. 

        '''
        if not training:
            with torch.no_grad():
                state = torch.from_numpy(
                    encoded_game_state
                    ).float().to(self.device)
                probs, value = self.model(state)
                del state; torch.cuda.empty_cache()
                probs = probs.cpu().numpy().reshape(-1)
                value = value.item()
        else:
            probs, value = self.model(encoded_game_state)
        
        return probs, value
    
    def approximate_outcome(self, game, search_iters):
        '''Approxuimates the outcome of game. For this to malke sense the game
        must be a normal play short partisan combinatorial game (a combinatorial 
        game which ends in a finite number of moves and the last player to 
        move wins).
        
        The approximate outcome is found by having the agent self-play
        the game twice: once with each player starting. The results are
        combined via the outcome rules for normal play short partisan 
        combintorial games to determine an approximate outcome.
        
        (For more in for on combinatorila game see :
        https://en.wikipedia.org/wiki/Combinatorial_game_theory)
            
        
        Parameters
        ----------
        game : UpDown
            a game of upset-downset.
        search_iters : int (positive),
            the number of search iterations to perform during each MCTS.

        Returns
        -------
        approx_out : str,
            'Next', if the Next player (first player to move) wins.
            'Previous', if the Previous player (second player to move) wins.
            'Up', if Up can force a win. (Playing first or second). 
            'Down', if Down can force a win. (Playing first or second).

        '''
        # make sure the model is in evaluation mode.
        self.model.eval()
        # game state from each players persepective
        up_start = GameState(game, UP)
        down_start = GameState(game, DOWN)
        # action space and stor for outcomes found.
        actions = np.arange(MAX_NODES)
        outcomes = []
        #self-play, once with each player moving first
        for game_state in [up_start, down_start]:
            # set root
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
        
        # get outcome prediction
        up_start_out, down_start_out = outcomes
        if up_start_out == 'P' and down_start_out == 'P':
            approx_out = 'Previous'
        elif up_start_out == 'N' and down_start_out == 'N':
            approx_out = 'Next'
        elif up_start_out == 'P' and down_start_out == 'N':
            approx_out = 'Down'
        elif up_start_out == 'N' and down_start_out == 'P':
            approx_out = 'Up'
            
        return approx_out
    
    @staticmethod
    def initialize(path=None):
        '''Initalizes model paramters for a new Agent: initial.pt, 
        apprentice.pt and alpha.pt. This method is called at the start of
        self-play/train/evaluation 

        Parameters
        ----------
        path : str, optional
            path to directory where model parameters will be saved. If no path 
            is passed the parameters will be saved in the directory 'model_data'
            insde the current working directory. If a path is passed it is 
            assumed that it exists. The default is None.

        Returns
        -------
        None.

        '''
        # set directory path
        path = './model_data' if path is None else pat
        if not os.path.isdir(path):
                   os.mkdir(path) 
        # set path to the various models  
        initial = os.path.join(path, 'initial.pt' )
        apprentice = os.path.join(path, 'apprentice.pt' )
        alpha = os.path.join(path, 'alpha.pt' )
        # save the inital parameters to each model. They shpould all be the 
        # same at the start!
        initial_model = AlphaZeroNet()
        for model in [initial, apprentice, alpha]:
            torch.save(initial_model.state_dict(), model)
        # clean up
        del initial_model