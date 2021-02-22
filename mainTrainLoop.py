#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""
import utils 
import selfPlay 
import train
import evalPlay 
import collections

if __name__ == '__main__':
    
    # (hyper) parameters
    TRAIN_ITERS = 5
    PROCESSES  = 4
    SELF_PLAYS = 10000
    SEARCHES = 400
    TEMP = 1
    TEMP_THRSHLD = 2
    REPLAY_BUFFER_SIZE = 250000
    TRAIN_PASSES = 2        
    BATCH_SIZE = 16
    SYMMETRIES = 8
    LEARN_RATE = 0.01
    MOMENTUM = 0.9
    EVAL_PLAYS = 200
    WIN_RATIO = 0.55
        
    # get the last training iteration. If 0 initialize model paramaters
    last_train_iter = utils.latest_training_iteration()    
    if last_train_iter == 0:
        utils.initialize_model_parameters()
    
    # set current training iteration
    cur_train_iter = last_train_iter + 1
    
    # initialize replay buffer
    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER_SIZE)
    
    # self-play, train, evaluation loop
    for train_iter in range(cur_train_iter, cur_train_iter + TRAIN_ITERS):      
        print(f'Starting self-play, training, evaluation'\
              f' loop at iteration : {train_iter}\n')         
               # self-play
        selfPlay.multi_self_play(train_iter,
                                 num_processes=PROCESSES,
                                 total_plays=SELF_PLAYS,
                                 search_iters=SEARCHES,
                                 temp=TEMP,
                                 temp_thrshld=TEMP_THRSHLD)      
        # train
        train.train(replay_buffer,
                    train_iter,
                    passes=TRAIN_PASSES  ,
                    batch_size=BATCH_SIZE,
                    num_symmetries=SYMMETRIES,
                    learning_rate=LEARN_RATE,
                    momentum=MOMENTUM)      
        # evaluation
        evalPlay.multi_evaluation(train_iter,
                                  num_processes=PROCESSES,
                                  total_plays=EVAL_PLAYS,
                                  search_iters=SEARCHES,
                                  win_thrshld=WIN_RATIO)
