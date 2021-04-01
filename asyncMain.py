"""
@author: Charles Petersen and Jamison Barsotti
"""
from config import *
from agent import Agent
from scheduling import AsyncSignal, UpdateSignal, ReplayBuffer
from selfPlay import SelfPlay
from train import Train
from evaluation import Evaluation
import ray
import gc

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
    
    # run async self-plays
    self_plays = [SelfPlay.remote() for _ in range(ASYNC_SELF_PLAYS)]    
    run_self_plays = [
        self_play.run.remote(
            replay_buffer,
            update_signal,
            self_plays.index(self_play),
            )
        for self_play in self_plays
        ]
    
    # start async training when signal arrives
    train = Train.remote()
    ray.get(training_signal.wait.remote())
    print('Training in progress...')
    train.run.remote(replay_buffer, evaluation_signal)
    
    # run async evaluations each time an evaluation signal arrives
    while True:      
        ray.get(evaluation_signal.wait.remote())
        print(f'Evaluation in progress...')
        # get the the current update id 
        update_id = ray.get(update_signal.get_update_id.remote())
        # run evaluations and get the results
        evaluations = [
            Evaluation.remote(update_id) for _ in range(ASYNC_EVALUATIONS)
            ]    
        results = ray.get(
            [evaluation.run.remote() for evaluation in evaluations]
            )      
        # tally the results 
        apprentice_wins = sum(results)
        print(f'the apprentice won {apprentice_wins} games...')
        update = (apprentice_wins/NUM_EVAL_PLAYS) > WIN_RATIO   
        # update if neccesary
        if update:
            # increment and get the the current update id
            update_id = ray.get(update_signal.set_update_id.remote())
            # block until new alpha parameters are saved, indexed by current 
            # update_id
            ray.get(
                evaluations[0].update_alpha_parameters.remote(update_id)
                )
            # send update signal to self-play actors
            update_signal.send_update.remote()      
        # free up resources by killing the evaluation actors
        for evaluation in evaluations:
            ray.kill(evaluation) 
        # manual garbage collection
        gc.collect()
            
        
        
