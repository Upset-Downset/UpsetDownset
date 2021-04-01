"""
@author: Charles Petersen and Jamison Barsotti
"""

from upDown import UpDown
from randomDag import uniform_random_dag
import numpy as np

class RandomGame(UpDown):
    ''' Subclass of UpDown for (almost) unifromly randomly generated games of 
    upset-downset. (See the randomDag module.)
    '''
    def __init__(self, 
                 num_nodes, 
                 markov_exp=2, 
                 extra_steps=0,  
                 RGB = False):
        ''' Initializes a game of Upset-Downset on an (almost) uniformly 
        randomly generated DAG with 'num_nodes' nodes. 
        
        Parameters
        ----------
        num_nodes : int (nonnegative)
            number of nodes in the game
        markov_exp: float, optional
            The exponent on 'n' determining the number of steps taken in 
            the markov chain. The default is 1.
        extra_steps: int (nonnegative), optional
            add some extra steps to the markov chain
        RGB : bool, optional
            determines the coloring. If 'True' the nodes will be colored 
            randomly. Otherwise, all nodes will be colored green.

        Returns
        -------
        None

        '''
        dag = uniform_random_dag(
            num_nodes, 
            exp=markov_exp, 
            extra_steps=extra_steps
            )
        colors = {i: np.random.choice([-1,0,1]) for i in range(num_nodes)} \
            if RGB else None
        UpDown.__init__(self, dag, coloring = colors)