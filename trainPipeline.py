#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import selfPlay
import train
import evalPlay
import utils


if __name__ == '__main__':
    
    # (hyper) parameters
    TRAIN_ITERS = 100
    PROCESSES  = 4
    SELF_PLAYS = 1000
    SEARCHES = 100
    TEMP = 1
    TEMP_THRSHLD = 2
    BATCH_SIZE = 128
    SYMMETRIES = 10
    LEARN_RATE = 0.01
    MOMENTUM = 0.9
    EVAL_PLAYS = 400
    WIN_RATIO = 0.55
        
    # get the last training iteration. If 0 initialize model paramaters
    last_train_iter = utils.latest_iteration()    
    if last_train_iter == 0:
        utils.initialize_model_paramaters()
    
    # set current training iteration
    cur_train_iter = last_train_iter + 1
    
    # self-play, train, evaluation loop
    for train_iter in range(cur_train_iter, cur_train_iter + TRAIN_ITERS):
        
        print('Starting self-play, training, evaluation loop at iteration :', 
          train_iter, '\n')
        
        # self-play
        selfPlay.multiprocess_self_play(train_iter,
                                        num_processes=PROCESSES,
                                        total_plays=SELF_PLAYS,
                                        search_iters=SEARCHES,
                                        temp=TEMP,
                                        temp_thrshld=TEMP_THRSHLD)
        
        # train
        train.train(train_iter,
                    batch_size=BATCH_SIZE,
                    num_symmetries=SYMMETRIES,
                    learning_rate=LEARN_RATE,
                    momentum=MOMENTUM)
        
        # evaluation
        evalPlay.multiprocess_evaluation(train_iter,
                                         num_processes=PROCESSES,
                                         total_plays=EVAL_PLAYS,
                                         search_iters=SEARCHES,
                                         win_thrshld=WIN_RATIO)
    
        

