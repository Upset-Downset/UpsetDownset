"""
@author: Charles Petersen and Jamison Barsotti
"""
import model
import torch
import os
import datetime
import pickle

def initialize_model_parameters():  
    ''' Initialize model parameters for the alpha and apprentice models
    and save them in the directories "./train_data/model_data/alpha_data" and 
    "/train_data/model_data/apprentice_data", respectively.
    
    Returns
    -------
    None.

    '''
    # initrialize model
    print('Initializing model parameters...\n')
    net = model.AlphaZeroNet()
    
    # create directory for alpha model data
    if not os.path.isdir('./train_data/model_data/alpha_data'):
        if not os.path.isdir('./train_data/model_data'):
            if not os.path.isdir('./train_data'):
                os.mkdir('./train_data')
            os.mkdir('./train_data/model_data')
        os.mkdir('./train_data/model_data/alpha_data')
    
    date =  datetime.datetime.today().strftime("%Y-%m-%d")
    
    # save alpha model data 
    torch.save(
        net.state_dict(), 
        f'./train_data/model_data/alpha_data/0_alpha_net_{date}.pt')
        
    # create directory for apprentice model data
    if not os.path.isdir('./train_data/model_data/apprentice_data'):
        os.mkdir('./train_data/model_data/apprentice_data')
            
    # save apprentice model data: should be same as alpha model at start!
    torch.save(
        net.state_dict(), 
        f'./train_data/model_data/apprentice_data/0_apprentice_net_{date}.pt')
    
    del net

def path_to_most_recent_file(search_path):
    ''' Returns path to the most recently updated file in the directory 
    ending 'search_path'.

    Parameters
    ----------
    serach_path : str
        the path to search for most recently updated data

    Returns
    -------
    most_recent : str
        path to the the most recently updated data in 'search_path'.'

    '''
    files = os.listdir(search_path)
    paths = [os.path.join(search_path, basename) for basename in files]
    most_recent = max(paths, key=os.path.getctime)
    
    return most_recent

def latest_training_iteration():
    '''Returns the index of the most recent iteration of the  main training 
    pipeline. (Determines most recent iteration of apprentice model data)
    
    Returns
    -------
    int : int (nonnegative)
        the index of the most recent iteration of main the training 
    pipeline.

    '''
    search_path = './train_data/model_data/apprentice_data'
    if os.path.isdir(search_path):
        filename = path_to_most_recent_file(search_path).split('/')[-1]
        last_iter = int(filename.split('_')[0]) 
    else:
        last_iter = 0
        
    return last_iter

def load_model(alpha_or_apprentice, device):
    ''' Returns the alpha model or apprentice model on 'device'.
    
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
    search_path = f'./train_data/model_data/{alpha_or_apprentice}_data'
    path_to_model_data = path_to_most_recent_file(search_path)

    # initialize model
    net = model.AlphaZeroNet()
    # if GPU is available, put model/params  on GPU. 
    # device index 0 by default.
    if device == 'cuda':
        net.load_state_dict(
            torch.load(path_to_model_data, map_location='cuda:0'))
        net.to(device)
    else:
        net.load_state_dict(
            torch.load(path_to_model_data, map_location='cpu'))
        
    return net

def save_model(net, alpha_or_apprentice, train_iter):
    '''Save the the parameters of 'net' as either alpha or apprentice.

    Parameters
    ----------
    net : UpDownNet
        apprnetice net
    alpha_or_apprentice : str
        paramaters to be saved as : 'alpha' or 'apprentice'.
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline

    Returns
    -------
    None.

    '''
    date =  datetime.datetime.today().strftime("%Y-%m-%d")
    filename = f'{train_iter}_{alpha_or_apprentice}_net_{date}.pt'
    path = os.path.join(
        f'./train_data/model_data/{alpha_or_apprentice}_data', filename)
    torch.save(net.state_dict(), path) 
    
def pickle_play_data(play_type, data, prcs_id, game_id, train_iter):
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
    game_id: int (nonnegative)
        the index of the game being played.
    prcs_id : int (nonnegative)
        the process index in multiprocess
    train_iter : int (nonnegative)
        the current iteration of the main training pipeline
        
    Returns
    -------
    None.
    
    '''
    
    date =  datetime.datetime.today().strftime("%Y-%m-%d")
    filename = f'{play_type}_iter{train_iter}_prcs{prcs_id}_game{game_id}_{date}'
    path = os.path.join(
        f'./train_data/{play_type}_data/iter_{train_iter}', filename)
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
    path = f'./train_data/{play_type}_data/iter_{train_iter}'
    for filename in os.listdir(path):
        full_filename = os.path.join(path, filename)
        file  = open(full_filename, 'rb')
        data = pickle.load(file)
        file.close()
        if play_type == 'self_play':
            play_data.extend(data)  
        else:
            play_data.append(data)
    return play_data