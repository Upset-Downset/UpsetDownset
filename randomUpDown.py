#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import upDown as ud
import randomDag as rd

import numpy as np

class RandomGame(ud.UpDown):
    ''' Subclass of UpDown for (almost) unifromly randomly generated games of 
    Upset-Downset.
    '''
    def __init__(self, n, markov_exp=1, start=None, RGB = False):
        ''' Initializes a game of Upset-Downset on an (almost) uniformly 
        randomly generated DAG with 'n' nodes. (See the randomDag module.)
        
        Parameters
        ----------
        n : int (nonnegative)
            number of nodes in the game
        markov_exp: float, optional
            The exponent on n determining the number of steps taken in the markov 
        process
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
        dag = rd.uniform_random_dag(n, exp=markov_exp, X_0=start)
        colors = {i: np.random.choice([-1,0,1]) for i in range(n)} \
            if RGB else None
        ud.UpDown.__init__(self, dag, coloring = colors)