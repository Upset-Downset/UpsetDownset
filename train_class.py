"""
@author: Charles Petersen and Jamison Barsotti
"""
from config import *
from agent import Agent
from model import AlphaLoss
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import ray


@ray.remote(num_gpus=0.075)
class Train(object):
    ''' Abstract class for construction remote train actors.
    '''
    def __init__(self):
        ''' Instantiates a Train actor.

        Returns
        -------
        None.

        '''
        self.agent = Agent(path='./model_data/apprentice.pt')
        self.optimizer =  optim.SGD(self.agent.model.parameters(),
                                    lr=LEARN_RATE,
                                    momentum=MOMENTUM,
                                    weight_decay=REGULAR)
        self.loss = AlphaLoss()
        self.epoch = 0
        self.writer = SummaryWriter()

    def get_symmetries(self, train_data, num_symmetries):
        '''Returns 'num_symmetries' of each example in 'train_data'.
        (A reindexing of node labels in any upset-downset game provides a
        symmetry of the gameboard.)
    
        Parameters
        ----------
        train_data : list
            a list of triples: state, policy and value for training from self-play.

        num_symmetries : int (positive),
            The number of symmetries to take (sampled w/ replacement).
            
        Returns
        -------
        sym_train_data : list
             'num_symmetries' symmetries of 'train_data'.
        ''' 
        sym_train_data = [] 
        for train_triple in train_data:  
            # unpack the training example
            state, policy, value =  train_triple           
            for _ in range(num_symmetries):
                # get random permutation 
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
        '''Starts indefinite training loop. The train process is synchronized 
        with the self-play processes and evaluation processes via the 
        'replay_buffer' and 'evaluation_signal', respectively. 'replay_buffer'
        stores the self-play data and triggers the start of training while 
        'evaluaton_signal' triggers an evaluation.

        Parameters
        ----------
        replay_buffer : ReplayBuffer
            remote actor string and communicating self-play data between the
            self-play actors and the train actor. Also carries the signal 
            to start training.
        evaluation_signal : AsynSignal
            remote actor for synchronization between the train process and 
            evaluation processes. Triggers an evaluation.
        num_symmetries : int (positive), optional
            The number of symmetries of each training sample to take 
            (sampled w/ replacement). The default is NUM_SYMMETRIES.
        epochs_to_eval : int (positive), optional
            the number of training epochs between evaluations. 
            The default is EPOCHS_TO_EVAL.

        Returns
        -------
        None.

        '''
        # start trainning!
        while True:
            # grab a batch and get symmetries
            batch = ray.get(replay_buffer.get_batch.remote())
            batch_sym = self.get_symmetries(batch, num_symmetries)
            batch_states, batch_probs, batch_values = zip(*batch_sym)
            
            # predict
            self.optimizer.zero_grad()
            states = torch.FloatTensor(batch_states).to(self.agent.device)
            probs = torch.FloatTensor(batch_probs).to(self.agent.device)
            values = torch.FloatTensor(batch_values).to(self.agent.device)
            out_logits, out_values = self.agent.predict(states, training=True)
            
            # calculate loss
            loss = self.loss(out_values, values, out_logits, probs)
            self.writer.add_scalar("Loss/train", loss, self.epoch)
           
            # step
            loss.backward()
            self.optimizer.step()
            self.epoch += 1
        
            if self.epoch % epochs_to_eval == 0:
                self.agent.model.save_parameters(
                    path='./model_data/apprentice.pt'
                    )
                evaluation_signal.send.remote()
                
            # cleanup
            del states; del probs; del values
            del out_logits; del out_values; 
            del loss_policy; del loss_value
            torch.cuda.empty_cache 
