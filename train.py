"""
@author: Charles Petersen and Jamison Barsotti
"""

from gameState import GameState
import numpy as np
import torch
import torch.nn.functional as F
import ray

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
            p = np.random.permutation(GameState.NUM_ACTIONS)
            # re-index nodes by permuting columns and rows
            state_sym = state[:,:,p]
            state_sym = state_sym[:,p,:]
            # permute nodes in policy too!
            policy_sym = policy[p]
            sym_train_data.append((state_sym, policy_sym, value)) 
            
    return sym_train_data

def train(agent,
          optimizer,
          scheduler, 
          batch_size,
          num_symmetries,
          num_epochs):
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
    train_idx = 1
    total_loss = 0
    for _ in range(num_epochs):
        batch = ray.get(scheduler.get_training_batch.remote(batch_size))
        batch_sym = symmetries(batch, num_symmetries)
        batch_states, batch_probs, batch_values = zip(*batch_sym)
            
        # query the agent
        optimizer.zero_grad()
        states = torch.FloatTensor(batch_states).to(agent.device)
        probs = torch.FloatTensor(batch_probs).to(agent.device)
        values = torch.FloatTensor(batch_values).to(agent.device)
        out_logits, out_values = agent.predict(states, training=True)
            
        # calculate loss
        loss_value = F.mse_loss(out_values.squeeze(-1), values)
        loss_policy = -F.log_softmax(out_logits, dim=1) * probs
        loss_policy = loss_policy.sum(dim=1).mean()
        loss = loss_policy + loss_value
        
        total_loss += loss.item()
        # step
        loss.backward()
        optimizer.step()
        if train_idx %10 == 0:
            #print(f'Avg. loss over {train_idx} epochs: {total_loss/train_idx}')
            pass
        del states; del probs; del values
        del out_logits; del out_values; 
        del loss_policy; del loss_value
        torch.cuda.empty_cache
        
        train_idx += 1
    
        
