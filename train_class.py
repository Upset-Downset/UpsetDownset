#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:35:14 2021

@author: charlie
"""
from config import *
from agent import Agent
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import ray


@ray.remote(num_gpus=0.075)
class Train(object):
    
    def __init__(self):
        self.agent = Agent(path='./model_data/apprentice.pt')
        self.optimizer =  optim.SGD(self.agent.model.parameters(),
                                    lr=LEARN_RATE,
                                    momentum=MOMENTUM,
                                    weight_decay=REGULAR)
        self.epoch = 0

    def get_symmetries(self, train_data, num_symmetries):
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
    
    def run(self,
            replay_buffer,
            evaluation_signal,
            num_symmetries=NUM_SYMMETRIES, 
            epochs_to_eval=EPOCHS_TO_EVAL): 
        
        total_loss = 0
        while True:
            batch = ray.get(replay_buffer.get_batch.remote())
            batch_sym = self.get_symmetries(batch, num_symmetries)
            batch_states, batch_probs, batch_values = zip(*batch_sym)
            
            # query the agent
            self.optimizer.zero_grad()
            states = torch.FloatTensor(batch_states).to(self.agent.device)
            probs = torch.FloatTensor(batch_probs).to(self.agent.device)
            values = torch.FloatTensor(batch_values).to(self.agent.device)
            out_logits, out_values = self.agent.predict(states, training=True)
            
            # calculate loss
            loss_value = F.mse_loss(out_values.squeeze(-1), values)
            loss_policy = -F.log_softmax(out_logits, dim=1) * probs
            loss_policy = loss_policy.sum(dim=1).mean()
            loss = loss_policy + loss_value
            total_loss += loss.item()
            
            # step
            loss.backward()
            self.optimizer.step()
        
            self.epoch += 1
            
            if self.epoch % 5000 == 0:
                print(f'Avg. loss over {self.epoch} epochs: {total_loss/self.epoch}')
        
            if self.epoch % epochs_to_eval == 0:
                torch.save(
                    self.agent.model.state_dict(), './model_data/apprentice.pt')
                evaluation_signal.send.remote()
                
            # cleanup
            del states; del probs; del values
            del out_logits; del out_values; 
            del loss_policy; del loss_value
            torch.cuda.empty_cache 