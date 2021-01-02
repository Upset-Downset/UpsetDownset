#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import upDown as ud
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
        cover relations for randomly generated poset. List of (upper) covers 
        keyed by element.

    '''
    random_relations = {i:[] for i in range(n)}
    # Randomy choose the number of edges to include. (Between 0 and 
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
        # Check for cycles
        if dag.is_acyclic(random_relations):
            r -= 1
        else:
            random_relations[i].pop()
    # Reduce to cover relations.
    random_covers = dag.transitive_reduction(random_relations)
    return random_covers

class RandomGame(ud.UpDown):
    ''' Subclass of Updown for randomly generated games of upset-downset.
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
            coloring = None
        random_covers = random_poset_relations(n)
        ud.UpDown.__init__(self, random_covers, coloring = coloring, covers = True)