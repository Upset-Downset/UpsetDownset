"""
@author: Charles Petersen and Jamison Barsotti
"""

import utils
from gameState import GameState
import mcts
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
               prcs_id, 
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
    actions = np.arange(GameState.NUM_ACTIONS) 
    state_generator = GameState.state_generator(prcs_id, 
                                                train_iter,
                                                start=False,
                                                markov_exp=2)
    for k in range(num_plays):      
        #store states encountered
        states = []
    
        # uniformly randomly choose which model plays first
        net_to_start = np.random.choice([alpha, apprentice])     
        
        # play a randomly generated game of upset-downset
        game_state = next(state_generator)
        states.append(game_state.encoded_state)
        cur_net = net_to_start
        move_count = 0
        winner = None
        
        while not game_state.is_terminal_state():
            root = mcts.PUCTNode(game_state)
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
            game_state = root.edges[move].state
            states.append(game_state.encoded_state)
            cur_net = 1 - cur_net
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
                               prcs_id, 
                               k, 
                               train_iter)
    
    # append results to shared list
    shared_results.append(apprentice_wins)

def multi_evaluation(train_iter,
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
    
    # create directory for pickling
    if not os.path.isdir(f'./train_data/evaluation_data/iter_{train_iter}'):
        if not os.path.isdir('./train_data/evaluation_data'):
            os.mkdir('./train_data/evaluation_data')
        os.mkdir(f'./train_data/evaluation_data/iter_{train_iter}')
    
    # check if there are enough CPUs
    if num_processes > mp.cpu_count():
        num_processes = mp.cpu_count()
        print('The number of processes exceeds the number of CPUs.' \
              f'Setting num_processes to {num_processes}.')
            
    # evenly split plays per process   
    num_plays = total_plays//num_processes        
        
    # initialze alpha/apprentice nets and load paramaters 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'      
    print('Initializing alpha and apprentice models on' \
          f' device : {device}...')    
    alpha_net = utils.load_model('alpha', device)
    apprentice_net = utils.load_model('apprentice', device)
    alpha_net.eval()
    apprentice_net.eval()
    
    if num_processes > 1:  
        
        print(f'{num_processes} evaluation processes in progress @' \
              f'{num_plays} games per process...')
        
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
            
        print(f'Finished {total_plays} games of multi-process evaluation.')
        
    else:        
        
        print(f'{total_plays} games of evaluation in progress...')
        
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
                       temp = 0)
            
        # get the results        
        apprentice_wins = results[-1] 
        
        print(f'Finished {num_plays*num_processes} games of evaluation.') 
    
    #results
    print(f'The apprentice model won {apprentice_wins} games ' \
          f'and alpha won {(num_plays*num_processes)-apprentice_wins}.')
                  
    if apprentice_wins/(num_processes*num_plays) > win_thrshld:   
        print('Updating the alpha models parameters...')
        
        # save model parameters  
        utils.save_model(apprentice_net, 'alpha', train_iter) 
        
    del alpha_net; del apprentice_net; torch.cuda.empty_cache()
    print('')
    
    return apprentice_wins
