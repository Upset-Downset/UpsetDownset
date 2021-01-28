#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""
import os 
import datetime
import pickle

import model
import gameState as gs
import torch

def initialize_model_paramaters():  
    ''' Initialize model paramaters for the alpha and apprentice models
    and save them in the directories "./model_data/alpha_data" and 
    "./model_data/apprentice_data", respectively.
    
    Returns
    -------
    None.

    '''
    net = model.UpDownNet(
        input_shape=gs.STATE_SHAPE, 
        actions_n=gs.UNIV)
    
    # create directory for alpha model data
    if not os.path.isdir('./model_data/alpha_data'):
        if not os.path.isdir('./model_data'):
            os.mkdir('./model_data')
        os.mkdir('./model_data/alpha_data')
            
    # save alpha model data 
    torch.save(net.state_dict(), 
               './model_data/alpha_data/0_alpha_net.pt')
        
    # create directory for apprentice model data
    if not os.path.isdir('./model_data/apprentice_data'):
        os.mkdir('./model_data/apprentice_data')
            
    # save apprentice model data: should be same as alpha model at start!
    torch.save(net.state_dict(), 
               './model_data/apprentice_data/0_apprentice_net.pt')
    
    del net

def path_to_latest_model(alpha_or_apprentice):
    ''' Returns path to the most recent paramater data for the alpha model 
    or apprentice model.

    Parameters
    ----------
    alpha_or_apprentice : str
        the model data requested: 'alpha' or 'apprentice'.

    Returns
    -------
    most_recent : str
        path to the most recent paramater data for the alpha model or 
        apprentice model

    '''
    search_path = os.path.join('./model_data/' + alpha_or_apprentice +'_data')
    files = os.listdir(search_path)
    paths = [os.path.join(search_path, basename) for basename in files]
    most_recent = max(paths, key=os.path.getctime)
    
    return most_recent

def latest_iteration():
    '''Returns the index of the most recent iteration of the  main training 
    pipeline. (Determines most recent iteration of apprentice model data)
    
    Returns
    -------
    int : int (nonnegative)
        the index of the most recent iteration of main the training 
    pipeline.

    '''
    if os.path.isdir('./model_data/apprentice_data'):
        filename = path_to_latest_model('apprentice').split('/')[-1]
        last_iter = int(filename.split('_')[0]) 
    else:
        last_iter = 0
        
    return last_iter

def get_model(alpha_or_apprentice, device):
    ''' Returns the alpah model or apprentice model on 'device'.
    
    Parameters
    ----------
    alpha_or_apprentice : str
        the model requested : 'alpha' or 'apprentice'.
    device : str
        the device the model will be loaded on: either 'cpu' or 'cuda'. 

    Returns
    -------
    net : UpDownNet

    '''
    # get path to moset recent model data
    path_to_model_data = path_to_latest_model(alpha_or_apprentice)

    # initialize model and load paramaters
    net = model.UpDownNet(input_shape=gs.STATE_SHAPE, actions_n=gs.UNIV)
    # if GPU is available, put model on GPU. (device index 0 by default)
    if device == 'cuda':
        net.load_state_dict(
            torch.load(path_to_model_data, map_location='cuda:0'))
        net.to(device)
    # no GPU...
    else:
        net.load_state_dict(
            torch.load(path_to_model_data, map_location='cpu'))
        
    return net
    
def pickle_play_data(play_type, data, proc_id, play_idx, train_iter):
    ''' Pickles gameplay data.

    Parameters
    ----------
    play_type: str
        either 'self_play' of 'evaluation'
    data : list
        for 'self-play' data each game is stored as a list of triples,
        one triple for each state encountered (excluding the terminal state):
            - state encountered
            - mcts policy found for said state
            - game value determined from perspective of the current player at
            said state after self-play.
        for 'evaluation' data each game is stored as a pair:
            - model that moved first, 'alpha' or 'apprentice'
            - list of states encountered, including the terminal state.
    play_idx: int (nonnegative)
        the index of the game being played.
    proc_id : int (nonnegative)
        the process index in multiprocess
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline
        
    Returns
    -------
    None.
    
    '''
    path = os.path.join(
        './' + play_type + 
        '_data/iter_%d/self_play_iter%d_proc%i_game_%i_%s' % 
        (train_iter,
         train_iter,
         proc_id, 
         play_idx,
         datetime.datetime.today().strftime("%Y-%m-%d")))
    with open(path, 'wb') as write:
        pickle.dump(data, write)
        
def load_play_data(play_type, train_iter):
    '''loads all pickled 'play_type' play data for itertion 'train_iter'.
    
    Parameters
    ----------
    play_type: str
        either 'self-play' of 'evaluation'.
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline.

    Returns
    -------
    play_data: list
        for 'self-play' data each element in 'play_data' is a triple which 
        corresponds to a single state encountered during self-play:
            - state encountered
            - mcts policy found for said state
            - game value determined from perspective of the current player at
            said state after self-play.
        for 'evaluation' data each element in 'play_data' corresponds to a 
        single game of evaluation play. The data for each game is stored as
        pair of length three:
            - model that moved first, 'alpha' or 'apprentice'
            - player which moved first, 'Up' or 'Down'
            - list of states encountered, including the terminal state.

    '''
    play_data = []
    path = os.path.join('./' + play_type + '_data/iter_%d' % train_iter)
    for filename in os.listdir(path):
        full_filename = os.path.join(path, filename)
        file  = open(full_filename, 'rb')
        data = pickle.load(file)
        if play_type == 'self_play':
            play_data.extend(data)  
        else:
            play_data.append(data)
    return play_data
