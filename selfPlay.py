#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""
import gameState as gs
import mcts
import randomUpDown as rud
import model
import utils

import numpy as np
import torch
import torch.multiprocessing as mp

import os

def self_play(alpha_net, 
              device, 
              num_plays,
              search_iters,
              train_iter,
              proc_id,  
              temp, 
              temp_thrshld,
              RGB=True):
    ''' Function for carrying out 'num_plays' games  of 'alpha_net' self-play 
    of randomly generated upset-downset games. The play data is pickled for 
    each game played. (see the pickle_self_play() function)
    
    Parameters
    ----------
    alpha_net : UpDownNet
        The current best model.
    device : str
        the device to run the model on ('cuda' if available, else 'cpu')
    num_plays : int (nonnegative)
        the number of games to play during evaluation.
    search_iters : int (nonnegative), optional
        the number of iteratins of  MCTS to be performed for each turn. 
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline
    proc_id : int (nonnegative)
        the process index in multiprocess
    temp : float (nonnegative), optional
        controls the exploration in picking the next move.
    temp_thrshld : int (nonnegative), optional
        controls the number of moves in play until actions are chosen 
        deterministically via their visit count in MCTS. 
    RGB : bool, optional
        True if games played are to be red-green-blue. otherwise, 
        all green games will be played. the default is True.
        

    Returns
    -------
    None.

    '''
    actions = np.arange(gs.UNIV)
    
    for k in range(num_plays):
        #size = np.random.choice(np.arange(1,gs.UNIV+1))
        G = rud.RandomGame(gs.UNIV, RGB=RGB)
        first_move = np.random.choice([gs.UP, gs.DOWN])
        initial_state = gs.to_state(G, to_move=first_move)
        root = mcts.PUCTNode(initial_state)
        states = []
        policies = []
        move_count = 0
        
        # play until a terminal state is reached
        while not gs.is_terminal_state(root.state):
            mcts.MCTS(root, alpha_net, device, search_iters)
            if move_count <= temp_thrshld:
                t = temp
            else:
                t = 0
            policy = mcts.MCTS_policy(root, t)
            move = np.random.choice(actions, p=policy)
            states.append(root.state)
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
        
        # pickle self-play data
        utils.pickle_play_data('self_play',
                               train_data, 
                               proc_id, 
                               k, 
                               train_iter)

def multiprocess_self_play(train_iter,
                           num_processes=4,
                           total_plays=5000,
                           search_iters=800,
                           temp=1,
                           temp_thrshld=2):
    ''' Wrapper for performing multi-process self-play.
    
    Parameters
    ----------
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline
    num_processes : int (nonnegative), optional
        the number of processes in the multiprocess. The default is 4.
    total_plays : int (nonnegative), optional
        the total number of evaluation games to be play during all processes.
        The default is 5000.
    search_iters : int (nonnegative), optional
        the number of iteratins of  MCTS to be performed for each turn of 
        self-play. The default is 800.
    temp : float (nonnegative), optional
        controls the exploration in picking the next move. The default is 1
    temp_thrshld : int (nonnegative), optional
        controls the number of moves in play until actions are chosen 
        deterministically via their visit count in MCTS. The default is 2.
        
    Returns
    -------
    apprentice_wins : int (nonnegative)
        the total wins by 'apprentice_net' over the multiprocess evaluation.

    '''
    print('Preparing multi-process self-play...')
    
    # check if there are enough CPUs
    if num_processes > mp.cpu_count():
        num_processes = mp.cpu_count()
        print('The number of processes exceeds the number of CPUs.' +
              ' Setting num_processes to %d' % num_processes)
            
    # evenly split plays per process   
    num_plays = total_plays//num_processes
    
    # create directory for saving this iterations pickeled self-play data
    if not os.path.isdir('./self_play_data/iter_%d' % train_iter):
        if not os.path.isdir('./self_play_data'):
            os.mkdir('./self_play_data')
        os.mkdir('./self_play_data/iter_%d' % train_iter)
        
        
    #initialze alpha net,load paramaters and place in evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
      
    print('Initializing alpha model on device :', device, '...') 
    
    alpha_net = utils.get_model('alpha', device)
    alpha_net.eval()
    
    # self-play
    if num_processes > 1:
        
        print('%d self-play processes in progress @' % num_processes, 
              str(num_plays), 'games per process...')
        
        # start processes
        mp.set_start_method("spawn",force=True)
        processes = []       
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=self_play, 
                               args=(alpha_net,
                                     device,
                                     num_plays,
                                     search_iters,
                                     train_iter,
                                     i,
                                     temp,
                                     temp_thrshld))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            
        print('Finished %d games of multi-process self-play.\n' % total_plays)
    
    else:
        print('%d games of self-play in progress...' % total_plays)
        with torch.no_grad():
            self_play(alpha_net, 
                      device, 
                      num_plays,
                      search_iters,
                      train_iter,
                      0, 
                      temp,
                      temp_thrshld)
            
        print('Finished %d games of self-play.\n' % total_plays)
        
    del alpha_net; torch.cuda.empty_cache()
    
