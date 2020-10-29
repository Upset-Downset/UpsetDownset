#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:46:57 2020

@author: charlie
"""

from upDown import *
import dagUtility as dag
import random

def total_orders(orders):
    ''' Returns poset cover relations and Hasse Diagram node positions
    for a disjoint sum of totally ordered posets.
    
    Parameters
    ----------
    orders : list
        positive intgers, each representing the cardinality of the corresponding 
        total order

    Returns
    -------
    dict
        of dicts: dict of poset relations keyed by 'relations', and dict of the
        corresponding Hasse diagram node positions keyed by 'positions'.     
            - 'relations' dict is keyed by elements of the poset (consecutive 
            nonnegative integers starting at 0) with corresponding value being 
            the list of (upper) covers for that element.
            - 'positions' dict is keyed by elements of the poset (consecutive 
            nonnegative integers starting at 0) with corresponding value being 
            the tuple containg the xy-position of that element in the Hasse 
            diagram of the poset.

    '''
    cover_relations = {}
    positions = {}
    nim_like = {'relations':cover_relations, 'positions': positions}
    heap_count = 0
    elem_count = 0
    for k in orders:  
        # Build relations for this heap, and add to cover relations
        heap_relations = {j:[j+1] for j in range(elem_count, elem_count+k-1)}
        heap_relations[elem_count+k-1] = []
        cover_relations.update(heap_relations)
        # Build positions of rthis heap, amnd add to positions.
        heap_positions = {j:(heap_count, j - elem_count) for j in range(elem_count, elem_count+k)}
        positions.update(heap_positions)
        elem_count += k
        heap_count += 1
    return nim_like

class NimLikeGame(UpDown):
    ''' Subclass of Updaown for NIM-like games of upset-downset.
    '''
    def __init__(self, heaps, colored = False):
        ''' Initializes a game of upset-downset on a disjoint union of totally 
        ordered posets. Optionally colored, all green by default and randomly 
        otherwise.
        
        Parameters
        ----------
        heaps : list (positive integers)
            each element of the list represents a NIM heap of the corresponding 
            size. 
        colored : bool, optional
            determines the coloring of the underlying poset. If 'True' the 
            elements will be colored randomly. Otherwise, all elements will be 
            colored green.

        Returns
        -------
        None

        '''
        n = sum(heaps)
        if colored:
            coloring = {i: random.choice([-1,0,1]) for i in range(n)}
        else:
            coloring = {i:0 for i in range(n)}
        orders = total_orders(heaps)
        covers = orders['relations']
        UpDown.__init__(self, covers, coloring, covers = True)
        self._positions = orders['positions']  # USE THIS TO PLOT....