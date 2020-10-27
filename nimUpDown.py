#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:46:57 2020

@author: charlie
"""

from upDown import *
import dagUtility as dag
import numpy as np
import random

def int_to_bin(n):
    '''
    Parameters
    ----------
    n : int

    Returns
    -------
    str
        binary representation of the integer n as a string.

    '''
    return bin(n).replace("0b", "")


def total_orders(orders):
    ''' Returns poset cover relations and Hasse diagram node positions
    for a disjoint union of totally ordered sets.
    
    Parameters
    ----------
    orders : list
        positive intgers, each representing the cardinality of the corresponding 
        total order.

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
    def __init__(self, heaps):
        ''' Initializes an 'all green' game of upset-downset on a disjoint 
        union of totally ordered posets. (Equivalent to a game of NIM!)
        
        Parameters
        ----------
        heaps : list (positive integers)
            each element of the list represents a NIM heap of the corresponding 
            size. 

        Returns
        -------
        None

        '''
        n = sum(heaps)
        coloring = {i:0 for i in range(n)}
        orders = total_orders(heaps)
        covers = orders['relations']
        UpDown.__init__(self, covers, coloring, covers = True)
        self._heaps = heaps
        self._positions = orders['positions']  # USE THIS TO PLOT....
        
    # def gameboard(self):
                    
    def nim_sum(self):
        ''' Returns the NIM sum of the game.
        -------
        _nim_sum : int (nonnegative)
            to compute the NIM sum convert the heap sizes to binarty and sum them 
            without carrying!

        '''
        # write each heap size as binary string each having a common length.
        bin_heaps = [int_to_bin(heap) for heap in self._heaps]
        n = max(len(bin_heap) for bin_heap in bin_heaps)
        bin_heaps = [bin_heap.zfill(n) for bin_heap in bin_heaps]
        bin_heaps = [list(bin_heap) for bin_heap in bin_heaps]
        # convert binary heap sizes to numpy array and compute binary sum of 
        # heaps without carrying!
        H = np.array(bin_heaps)
        H = H.astype(int)
        S = H.sum(0) 
        S = np.mod(S,2)
        S = S.astype(str)
        S = list(S)
        bin_nim_sum = ''.join(S)
        _nim_sum = int(bin_nim_sum,2)
        return _nim_sum