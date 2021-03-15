"""
@author: Charles Petersen and Jamison Barsotti
"""
from config import *
from agent import Agent
from writeLock import save_with_lock
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import ray
import os

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
            p = np.random.permutation(MAX_NODES)
            # re-index nodes by permuting columns and rows
            state_sym = state[:,:,p]
            state_sym = state_sym[:,p,:]
            # permute nodes in policy too!
            policy_sym = policy[p]
            sym_train_data.append((state_sym, policy_sym, value)) 
            
    return sym_train_data

@ray.remote(num_gpus=0.1)
def train(scheduler, 
          num_symmetries=NUM_SYMMETRIES):
    
    #get apprentice
    apprentice = Agent(path='./model_data/apprentice.pt')

    # initialize SGD optimizer for training, if continuing, load parameters
    optimizer = optim.SGD(apprentice.model.parameters(),
                              lr=LEARN_RATE,
                              momentum=MOMENTUM)
    
    epoch = 0
    total_loss = 0
    while True:
        batch = ray.get(scheduler.get_training_batch.remote())
        batch_sym = symmetries(batch, num_symmetries)
        batch_states, batch_probs, batch_values = zip(*batch_sym)
            
        # query the agent
        optimizer.zero_grad()
        states = torch.FloatTensor(batch_states).to(apprentice.device)
        probs = torch.FloatTensor(batch_probs).to(apprentice.device)
        values = torch.FloatTensor(batch_values).to(apprentice.device)
        out_logits, out_values = apprentice.predict(states, training=True)
            
        # calculate loss
        loss_value = F.mse_loss(out_values.squeeze(-1), values)
        loss_policy = -F.log_softmax(out_logits, dim=1) * probs
        loss_policy = loss_policy.sum(dim=1).mean()
        loss = loss_policy + loss_value
        
        total_loss += loss.item()
        # step
        loss.backward()
        optimizer.step()
        
        epoch += 1
        
        if epoch %500 == 0:
            print(f'Avg. loss over {epoch} epochs: {total_loss/epoch}')
            ray.get(
                save_with_lock.remote(
                    apprentice, './model_data/apprentice.pt'
                    )
                )
         
        del states; del probs; del values
        del out_logits; del out_values; 
        del loss_policy; del loss_value
        torch.cuda.empty_cache       
        
            
            
    
        

            