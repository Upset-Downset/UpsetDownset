#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:40:51 2020

@author: charlie
"""

from upDown import *
import dagUtility as dag
import random

def random_poset_relations(n):
    '''
    Parameters
    ----------
    n : int (nonnegative)
        number of elements in the poset.

    Returns
    -------
    dict
        cover relations for poset. List of (upper) covers keyed by element.

    '''
    random_relations = {i:[] for i in range(n)}
    # Randomy choose the number of relations to include. (Between 0 and 
    # n(n-1)/2 since this is the maximum number of edges possible in a dag 
    # having n nodes.)
    r = random.randint(0, n*(n-1)/2)
    # Add relations.
    while r > 0:
        # Pick two distinct elements randomly
        i = random.randint(0, n-1)
        j = i
        while j == i:
            j = random.randint(0, n-1)
        # Add relation i<j.
        random_relations[i].append(j)
        # Check for anti-symmetry.
        if dag.is_acyclic(random_relations):
            r -= 1
        else:
            random_relations[i].pop()
    # Reduce to cover relations.
    random_covers = dag.transitive_reduction(random_relations)
    # determine number of components in Hasse diagram.
    components = dag.connected_components(random_relations)
    # If more than one component relabel elements so that each compnenet is 
    # labelled consecutively.
    if len(components) >1:
        random_covers_relabelled = {}
        relabel_index = 0
        for c in components:
            c_adj = dict(filter(lambda i: i[0] in c, random_covers.items()))
            c_adj_relabel = dag.integer_relabel(c_adj, relabel_index)['relabelled graph']
            relabel_index += len(c)
            random_covers_relabelled.update(c_adj_relabel)
        return random_covers_relabelled
    else: 
        return random_covers

class RandomGame(UpDown):
    ''' Subclass of Updaown for randomly generated games of upset-downset.
    '''
    def __init__(self, n, colored = False):
        ''' Initializes a game of upset-downset on a randomly generated poset 
        of cardinality 'n'. Optionally colored, all green by default and randomly 
        otherwise.
        
        Parameters
        ----------
        n : int (nonnegative)
            cardinality of the underlying poset.
        colored : bool, optional
            determines the coloring of the underlying poset. If 'True' the 
            elements will be colored randomly. Otherwise, all elements will be 
            colored green.

        Returns
        -------
        None

        '''
        if colored:
            coloring = {i: random.choice([-1,0,1]) for i in range(n)}
        else:
            coloring = {i:0 for i in range(n)}
        random_covers = random_poset_relations(n)
        UpDown.__init__(self, random_covers, coloring, covers = True)