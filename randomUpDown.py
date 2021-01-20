#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import upDown as ud
import digraph
import random

def random_dag(n):
    ''' Retursn a randomly generated directed acyclic graph on 'n' nodes'.
    
    Parameters
    ----------
    n : int (nonnegative)
        number of nodes in the directed acyclic graph.

    Returns
    -------
    dict
        adjacecny representation of  directed acyclic graph.
        (Adjacecny lists keyed by node.)

    '''
    random_dag = {i:[] for i in range(n)}
    # randomy choose the (max possible) number of edges. (Between 0 and 
    # n(n-1)/2 since this is the maximum number of edges 
    # possible in a dag having n nodes.)
    num_edges = random.randint(0, n*(n-1)/2)
    # add edges.
    while num_edges > 0:
        # randomly choose two distinct nodes 
        i, j = random.sample(range(n), 2)
        # add edge i-->j.
        random_dag[i].append(j)
        # check for cycles
        if digraph.is_acyclic(random_dag):
            num_edges -= 1
        else:
            random_dag[i].pop()
    # take transitive reduction
    random_dag = digraph.transitive_reduction(random_dag)
    return random_dag

class RandomGame(ud.UpDown):
    ''' Subclass of UpDown for randomly generated games of upset-downset.
    '''
    def __init__(self, n, RGB = False):
        ''' Initializes a game of upset-downset on a randomly generated 
        directed acyclic graph with 'n' nodes. Optionally colored, all green by default and randomly 
        otherwise.
        
        Parameters
        ----------
        n : int (nonnegative)
            number of nodes in the game
        RGB : bool, optional
            determines the coloring. If 'True' the nodes will be colored 
            randomly. Otherwise, all nodes will be colored green.

        Returns
        -------
        None

        '''
        dag = random_dag(n)
        colors = {i: random.choice([-1,0,1]) for i in range(n)} \
            if RGB else None
        ud.UpDown.__init__(self, dag, coloring = colors, reduced = True)