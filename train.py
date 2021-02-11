#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import gameState as gs 
import utils

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import random


def symmetries(train_data, num_symmetries):
    '''Returns 'num_samples' symmetries of each example in 'train_data'.
    (A reindexing of node labels in any upset-downset game provides a
     symmetry of the gameboard.)
    
    Parameters
    ----------
    train_data : list
        a list of triples: state, policy and value for training from self-play.
    dim : int (nonnegative), optional
        must be at least as large as the number of nodes in 'game'. 
        The default is UNIV.
    num_symmetries : int (positive),
        The number of symmetries to take (sampled w/ repitition).

    Returns
    -------
    sym_train_data : list
        'num_symmetries' symmetries of 'train_data'.

    ''' 
    sym_train_data = []
    
    for train_triple in train_data:
        
        state, policy, value =  train_triple
              
        for _ in range(num_symmetries):
            # get random permutation on dim # letters
            p = np.random.permutation(gs.UNIV)
            # re-index nodes by permuting columns and rows
            state_sym = state[:,:,p]
            state_sym = state_sym[:,p,:]
            # permute nodes in policy too!
            policy_sym = policy[p]
            sym_train_data.append((state_sym, policy_sym, value)) 
            
    return sym_train_data

def train(replay_buffer,
          train_iter,
          epochs,
          batch_size=256,
          num_symmetries=10,
          learning_rate=0.1,
          momentum=0.9):
    ''' Takes 'num_symmetries' of each training example from self_play 
    iteration 'train_iter' and perfoms a single training pass in batch sizes 
    of 'batch_size'.
    
    Parameters
    ----------
    train_iter : TYPE
        DESCRIPTION.
    batch_size : int (nonegatie), optional
        bathc sizes to be taken. The default is 256.
    num_symmetries : int (vepositi), optional
        The number of symmetries of each training example to take
        (sampled w/ repitition). The default is 10.
    learning_rate : float, optional
        the learning rate fro SGD optimization, between 0 and 1. 
        The default is 0.1.
    momentum : float, optional
        the momentum for SGD optimization, between 0 and 1. 
        The default is 0.9.

    Returns
    -------
    None.

    '''
    
    #initialze apprentice net and load paramaters 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    print(f'Loading apprentice model for training on device : {device}...')    
    apprentice_net = utils.load_model('apprentice', device)
    
    # initialize optimizer
    print('Initializing SGD optimizer...')   
    optimizer = optim.SGD(
        apprentice_net.parameters(),
        lr=learning_rate,
        momentum=momentum)
    
    # get self-play data
    print('Loading train data into the replay buffer..')   
    train_data = utils.load_play_data('self_play', train_iter)
    replay_buffer.extend(train_data)
    print(f'{len(train_data)} training examples added to the buffer.')
    print(f'Buffer size : {len(replay_buffer)}')
    
    # train the apprentice net 
    print(f'Performing {epochs} training steps on batches of' \
          f' size {batch_size} @ {num_symmetries} per training example...\n')
    
    epoch = 0
    total_loss = 0
    apprentice_net.train()
    for _ in range(epochs):
        # get training batch (uniformly at random w/ repoacement)
        batch = random.choices(replay_buffer, k=batch_size)
        # get symmetries
        batch = symmetries(batch, num_symmetries=num_symmetries)
        batch_states, batch_probs, batch_values = zip(*batch)
            
        # query the apprentice net
        optimizer.zero_grad()
        states = torch.FloatTensor(batch_states).to(device)
        probs = torch.FloatTensor(batch_probs).to(device)
        values = torch.FloatTensor(batch_values).to(device)
        out_logits, out_values = apprentice_net(states)
            
        # calculate loss
        loss_value = F.mse_loss(out_values.squeeze(-1), values)
        loss_policy = -F.log_softmax(out_logits, dim=1) * probs
        loss_policy = loss_policy.sum(dim=1).mean()
        loss = loss_policy + loss_value
        
        # step
        loss.backward()
        optimizer.step()
        
        del states; del probs; del values
        del out_logits; del out_values
        torch.cuda.empty_cache

        epoch +=1
        
        # get rolling avg loss
        total_loss += loss.item()     
        if epoch % 10 == 0:
            print(f'Average loss over {epoch} epochs : {total_loss/epoch}')
    
    # save model parameters  
    print('Saving apprentice model parameters...\n')
    
    utils.save_model(apprentice_net, 'apprentice', train_iter) 
    
    del apprentice_net; torch.cuda.empty_cache()
        
