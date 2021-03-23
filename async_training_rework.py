#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:54:57 2021

@author: charlie
"""
from config import *
from agent import Agent
from async_scheduling import AsyncSignal, UpdateSignal, ReplayBuffer
from self_play_class import SelfPlay
from train_class import Train
from eval_play_class import EvalPlay
import ray
import os

if __name__ == '__main__':
    
    ray.init()
    
    # initialize agent
    Agent.initialize()
    
    # syncronization signals
    training_signal = AsyncSignal.remote(clear=False)   
    evaluation_signal = AsyncSignal.remote(clear=True)
    update_signal = UpdateSignal.remote()
    
    # replay buffer 
    replay_buffer = ReplayBuffer.remote(training_signal)
    
    # directories for pickled self-play and evaluation data    
    if not os.path.isdir('./self_play_data'):
        os.mkdir('./self_play_data')
    if not os.path.isdir('./evaluation_data'):
        os.mkdir('./evaluation_data')
    
    # get self-play handles and run async self-plays
    self_plays = [SelfPlay.remote() for _ in range(ASYNC_SELF_PLAYS)]    
    run_self_plays = [
        self_play.run.remote(
            replay_buffer,
            update_signal,
            self_plays.index(self_play),
            )
        for self_play in self_plays
        ]
    
    # get training handle and start training when signal arrives
    train = Train.remote()
    # wait for signal to begin training
    ray.get(training_signal.wait.remote())
    print('Lets go!')
    train.run.remote(replay_buffer, evaluation_signal)
    
    # run async evaluations each time signal arrives
    while True:      
        ray.get(evaluation_signal.wait.remote())
        evaluations = [Evaluation.remote() for _ in range(ASYNC_EVAL_PLAYS)]      
        eval_results = ray.get(
            [
                evaluation.run.remote(
                    evaluations.index(evaluation)
                    ) 
                for evaluation in evaaluations
                ]
            )      
        # tally the evaluation results
        apprentice_wins = sum(eval_results)
        print(f'the apprentice won {apprentice_wins} games...')
        update = sum(eval_results)/NUM_EVAL_PLAYS > WIN_RATIO
        # send updates if neccesary
        if update:
            print('updating...')
            # block until alpha parameters are saved
            ray.get(eval_plays[0].update_alpha_parameters.remote())
            print('new parameters saved')
            #send update signal to self-plays
            update_signal.send_update.remote()
            print('updates sent')
        #  free up resources by killing the evaluation actors
        for eval_play in eval_plays:
            ray.kill(eval_play)
        
        
