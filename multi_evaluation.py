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

def evaluation(alpha_net, 
               apprentice_net, 
               device,
               num_plays,
               search_iters,
               train_iter,
               proc_id, 
               shared_results,
               RGB=True,
               temp = 0):
    '''Plays 'num_plays' games of randomly generated upset-downset games
    of size 'num'nodes' between 'alpha_net' and 'apprentice_net'. Returns 
    the total numbe of games won by 'apprentice_net'. The play data is pickled 
    for each game played. (see the pickle_evaluation_play() function)
    
    Parameters
    ----------
    alpha_net : UpDownNet
        The current best model.
    apprentice_net : UpDownNet
        the model currently training.
    device : str
        the device to run the model on ('cuda' if available, else 'cpu')
    num_plays : int (nonnegative)
        the number of games to play during evaluation.
    search_iters : int (nonnegative)
        the number of iteratins of  MCTS to be performed for each turn. 
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline
    proc_id : int (nonnegative)
        the process index in multiprocess
    shared_results: mp.manager.list()
        list shared amongst processes to store results of play.
    RGB : bool, optional
        True if games played are to be red-green-blue. otherwise, 
        all green games will be played. the default is True.
    temp : float (nonnegative), optional
        controls the exploration in picking the next move. The default is 0.

    Returns
    -------
    None.

    '''
    # track the models
    alpha = 0
    apprentice = 1
    
    # evaluation
    apprentice_wins = 0
    actions = np.arange(gs.UNIV) 
    
    for k in range(num_plays):
        
        #store states encountered
        states = []
        
        # uniformly randomly choose which model plays first
        # and which player moves first.
        net_to_start = np.random.choice([alpha, apprentice])     
        up_or_down = np.random.choice([gs.UP, gs.DOWN])
        
        # play a randomly generated game of upset-downset
        G = rud.RandomGame(gs.UNIV, RGB)
        cur_state = gs.to_state(G, to_move = up_or_down)
        states.append(cur_state)
        cur_net = net_to_start
        move_count = 0
        winner = None
        while not gs.is_terminal_state(cur_state):
            root = mcts.PUCTNode(cur_state)
            if cur_net == alpha:
                mcts.MCTS(
                    root, 
                    alpha_net, 
                    device, 
                    search_iters)
            else:
                mcts.MCTS(
                    root, 
                    apprentice_net, 
                    device, 
                    search_iters)
            policy = mcts.MCTS_policy(root, temp)
            move = np.random.choice(actions, p=policy)
            cur_state = root.edges[move].state
            cur_net = 1 - cur_net
            states.append(cur_state)
            move_count += 1
            
        # decide winner
        winner = 1-net_to_start if move_count %2 == 0 else net_to_start
        if winner == apprentice:
            apprentice_wins += 1
            
        # pickle evaluation data
        start = 'alpha' if net_to_start == alpha else 'apprentice'
        eval_data = [start, states]  
        utils.pickle_play_data('evaluation',
                               eval_data, 
                               proc_id, 
                               k, 
                               train_iter)
            
        
    # append results to shared list
    shared_results.append(apprentice_wins)

def multiprocess_evaluation(train_iter,
                        num_processes=4,
                        total_plays=400,
                        search_iters=800,
                        win_thrshld=0.55):
    ''' Wrapper for performing multi-process evaluation-play.
    
    Parameters
    ----------
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline.
    num_processes : int (nonnegative), optional
        the number of processes in the multiprocess. The default is 4
    total_plays : int (nonnegative)
        the total number of evaluation games to be play during all processes.
        The default is 400.
    search_iters : int (nonnegative)
        the number of iteratins of  MCTS to be performed for each turn 
        in evaluation. The default is 800.
    win_thrshld : float, optional
        between 0 and 1. The win percentage over all games played during 
        evaluation needed for'apprentice_net' to be considered better than 
        'alpha_net'. The default is 0.55.
        
    Returns
    -------
    apprentice_wins : int (nonnegative)
        the total wins by 'apprentice_net' over the multiprocess evaluation.

    '''
    print('Preparing multi-process evaluation...')
    
    # check if there are enough CPUs
    if num_processes > mp.cpu_count():
        num_processes = mp.cpu_count()
        print('The number of processes exceeds the number of CPUs.' +
              ' Setting num_processes to %d' % num_processes)
            
    # evenly split plays per process   
    num_plays = total_plays//num_processes
    
    # create directory for saving this iterations pickeled evaluation data
    if not os.path.isdir('./evaluation_data/iter_%d' % train_iter):
        if not os.path.isdir('./evaluation_data'):
            os.mkdir('./evaluation_data')
        os.mkdir('./evaluation_data/iter_%d' % train_iter)
        
        
    # initialze alpha net/apprentice_net,load paramaters and place 
    # in evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
      
    print('Initializing alpha and apprentice models on device :', device, '...') 
    
    alpha_net = utils.get_model('alpha', device)
    apprentice_net = utils.get_model('apprentice', device)
    alpha_net.eval()
    apprentice_net.eval()
    
    if num_processes > 1:  
        
        print('%d evaluation processes in progress @' % num_processes, 
              str(num_plays), 'games per process...')
        
        # a manager to collect results from all processes
        manager = mp.Manager()
        shared_results = manager.list()
    
        # evaluation processes
        mp.set_start_method("spawn",force=True)
        processes = []
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=evaluation, 
                               args=(alpha_net,
                                     apprentice_net,
                                     device,
                                     num_plays,
                                     search_iters,
                                     train_iter,
                                     i,
                                     shared_results))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            
        # collect the results        
        apprentice_wins = sum(shared_results)
            
        print('Finished %d games of multi-process evaluation.' % total_plays)
        
    else:        
        
        print('%d games of evaluation in progress... '% total_plays)
        
        results = []
        with torch.no_grad():
            evaluation(alpha_net, 
                       apprentice_net, 
                       device,
                       num_plays,
                       search_iters,
                       train_iter, 
                       0,
                       results,
                       RGB=True,
                       search_iters=800, 
                       temp = 0)
            
        # get the results        
        apprentice_wins = results[-1] 
        
        print('Finished %d games of evaluation.' % total_plays)
    
    if apprentice_wins/(num_processes*num_plays) > win_thrshld:
        
        print('The apprentice model beat the alpha', 
              apprentice_wins, 'games to ' + 
              str((num_plays*num_processes)-apprentice_wins) +
              '.')
        print('Updating the alpha models parameters...\n')
        
        # save model parameters  
        filename = str(train_iter) + '_alpha_net.pt'
        path = os.path.join('./model_data/alpha_data/', filename)
        torch.save(apprentice_net.state_dict(), path)
    
    else:
        print('The apprentice model won',
              apprentice_wins, 'games and alpha won ' +
              str((num_processes*num_plays)-apprentice_wins) + '.\n')
        
    del alpha_net; del apprentice_net; torch.cuda.empty_cache()
        
    return apprentice_wins