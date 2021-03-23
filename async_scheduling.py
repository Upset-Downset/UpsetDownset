"""
@author: charlie
"""
from config import *
import numpy as np
from collections import deque
import random
import asyncio
import ray

@ray.remote(num_cpus=0)
class AsyncSignal(object):
    '''Abstract class for construction of remote signal actors between async 
    self-play/ train processes and also train/evaluation processes.
    '''
    def __init__(self, clear=False):
        '''Instantiate an AsyncSignal Actor

        Parameters
        ----------
        clear : bool, optional
            True if the signal should be cleared (unset) once its been set and
            False otherwise. The default is False.

        Returns
        -------
        None.

        '''
        self.training_signal = asyncio.Event()
        self.clear = clear

    def send(self):
        ''' Set signal.
        
        Returns
        -------
        None.

        '''
        self.training_signal.set()
        if self.clear:
            self.training_signal.clear()

    async def wait(self, should_wait=True):
        ''' Wait for signal.

        Parameters
        ----------
        should_wait : bool, optional
            True if should wait until signal is set and False otherwise. 
            The default is True.

        Returns
        -------
        None.

        '''
        if should_wait:
            await self.training_signal.wait()

@ray.remote(num_cpus=0)            
class UpdateSignal(object):
    '''Abstract class for construction of remote update signal actors between  
    async self-play and evaluation processes.
    '''
    def __init__(self, size=ASYNC_SELF_PLAYS):
        '''Instanatiate an UpdateSignal.

        Parameters
        ----------
        size : int (positive), optional
            Dthe length of the update signal.I.e., the number of async 
            self-play processes The default is ASYNC_SELF_PLAYS.

        Returns
        -------
        None.

        '''
        self.signal = np.zeros(size, dtype=bool)
        
    def send_update(self):
        '''Send an update signal.

        Returns
        -------
        None.

        '''
        self.signal[:] = True
    
    def get_update(self, self_play_id):
        ''' Check if an update signal has been sent.

        Parameters
        ----------
        self_play_id : int (nonegative)
            unique identifier of a self-play process.

        Returns
        -------
        bool
            True if an update is needed and Fa;se otehrwise
        '''
        return self.signal[self_play_id]
        
    def clear_update(self, self_play_id):
        '''Clear the update signal after an update has occured.

        Parameters
        ----------
        self_play_id : int (nonnegative)
            unique identifier of a self-play process.

        Returns
        -------
        None.

        '''
        self.signal[process_id] = False
        
@ray.remote(num_cpus=0)
class ReplayBuffer(object):
    '''Abstract class for construction of replay buffers.
    '''
    def __init__(self, training_signal, size=MAX_REPLAY_BUFFER):
        '''Instantiate a ReplayBuffer.

        Parameters
        ----------
        training_signal : AsyncSignal
            signal to be sent once training should commence. Based on a 
            threshold for number of games added to the buffer. (See the add() 
            method.)
        size : int (positive), optional
            the maximum amount of positions to be stored in the buffer.
            The default is MAX_REPLAY_BUFFER.

        Returns
        -------
        None.

        '''
        self.buffer = deque(maxlen=size)
        self.training_signal = training_signal
        self.total_games = 0
   
    def get_total_games(self):
        '''Returns the total number of self-play games that have been added 
        to the buffer in its lifetime
        
        Returns
        -------
        int (nonnegative)
            the total number of games that have been added to the buffer in 
            its lifetime.

        '''
        return self.total_games
    
    def size(self):
        '''Returns the total number of training examples stored in the buffer.

        Returns
        -------
        int (nonegative)
            the total number of training examples stored in the buffer.
        '''
        return len(self.buffer)
    
    def add(self, train_data, games_to_train=GAMES_TO_TRAIN):
        ''' Adds all training examples for a single self-play game to the
        buffer. Send the signal to start training once the 'games_to_train'
        threshold has been reahced.
        
        Parameters
        ----------
        train_data : list
            list of training examples (state, policy, value) from a single 
            game of self-play.
        games_to_train : int (positive), optional
            the number of self-play games which need to be added to the buffer 
            before training commences. The default is GAMES_TO_TRAIN.

        Returns
        -------
        None.

        '''
        self.buffer.extend(train_data)
        self.total_games += 1
        if self.total_games == games_to_train:
            self.training_signal.send.remote()
    
    def get_batch(self, batch_size=BATCH_SIZE):
        '''Returns a uniformly randomly chosen batch of training examples 
        from the buffer (sample w/ replacement)

        Parameters
        ----------
        batch_size : int (positve), optional
            the batch size. The default is BATCH_SIZE.

        Returns
        -------
        list
            a list of training example of length 'batch_size'
        '''
        return random.choices(self.buffer, k=batch_size)