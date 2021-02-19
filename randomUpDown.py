"""
@author: Charles Petersen and Jamison Barsotti
"""

from upDown import UpDown
from randomDag import uniform_random_dag
import numpy as np

class RandomGame(UpDown):
    ''' Subclass of UpDown for (almost) unifromly randomly generated games of 
    Upset-Downset.
    '''
    def __init__(self, 
                 n, 
                 markov_exp=2, 
                 extra_steps=0, 
                 start=None, 
                 RGB = False):
        ''' Initializes a game of Upset-Downset on an (almost) uniformly 
        randomly generated DAG with 'n' nodes. (See the randomDag module.)
        
        Parameters
        ----------
        n : int (nonnegative)
            number of nodes in the game
        exponent: float, optional
            The exponent on 'n' determining the number of steps taken in 
            the markov chain. The default is 1.
        extra_steps: int (nonnegative), optional
            add some extra steps to the markov chain
        start : dict, optional
            adjacency representation of the DAG from which to start the markov 
            process. (Adjacency lists keyed by node.) The default is the empty 
            DAG on 'n' nodes. 
        RGB : bool, optional
            determines the coloring. If 'True' the nodes will be colored 
            randomly. Otherwise, all nodes will be colored green.

        Returns
        -------
        None

        '''
        if start is not None:
            assert len(start) == n, 'starting DAG is too big.'
        dag = uniform_random_dag(n, 
                                 exp=markov_exp, 
                                 extra_steps=extra_steps, 
                                 X_0=start)
        colors = {i: np.random.choice([-1,0,1]) for i in range(n)} \
            if RGB else None
        UpDown.__init__(self, dag, coloring = colors)