#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import upDown as ud

def integer_to_binary(n):
    ''' Returns the binary representation of af the integr 'n'.
    Parameters
    ----------
    n : int

    Returns
    -------
    str
        binary representation of the integer 'n'.

    '''
    return bin(n).replace("0b", "")


def nim_relations(heaps):
    ''' Returns poset cover relations for a disjoint union of totally ordered 
    sets (Nim heaps).
    
    Parameters
    ----------
    heaps : list
        positive intgers, each representing the cardinality of the corresponding 
        Nim heap.

    Returns
    -------
    dict
        keyed by elements of the poset (consecutive nonnegative integers 
        starting at 0) with corresponding value being the list of (upper) covers 
        for that element.

    '''
    cover_relations = {}
    elem_count = 0
    for k in heaps:  
        # Build relations for this heap, and add to cover relations
        heap_relations = {j:[j+1] for j in range(elem_count, elem_count+k-1)}
        heap_relations[elem_count+k-1] = []
        cover_relations.update(heap_relations)
        elem_count += k
    return cover_relations

def nim_coordinates(heaps):
    ''' Returns the coodinates of nodes in the Hasse diagram of a poset 
    which is disjoint union of totally ordered sets (NIM heaps).

    Parameters
    ----------
    heaps : list
        positive intgers, each representing the cardinality of the corresponding 
        NIM heap.

    Returns
    -------
    coordinates : dict
       xy-coordinates (tuple) of element in the Hasse diagram of the poset 
       keyed by element.

    '''
    coordinates = {}
    elem_count = 0
    heap_count = 0
    for k in heaps:
        heap_coords = {j:(heap_count, j - elem_count) for j in \
                       range(elem_count, elem_count+k)}
        coordinates.update(heap_coords)
        elem_count += k
        heap_count += 1
    return coordinates

class NimLikeGame(ud.UpDown):
    ''' Subclass of UpDown for NIM-like games of upset-downset.
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
        covers = nim_relations(heaps)
        coordinates = nim_coordinates(heaps)
        ud.UpDown.__init__(self, covers, coordinates = coordinates, covers = True)
        # Set heaps attribute
        self.heaps = heaps
        
                    
    def nim_sum(self):
        ''' Returns the NIM sum of the heaps in the game.
        -------
        nim_sum : int (nonnegative)
            to compute the NIM sum convert the heap sizes to binarty and sum them 
            without carrying!

        '''
        # write each heap size as binary string each having a common length.
        bin_heaps = [integer_to_binary(heap) for heap in self.heaps]
        n = max(len(bin_heap) for bin_heap in bin_heaps)
        bin_heaps = [bin_heap.zfill(n) for bin_heap in bin_heaps]
        # connvert binary strings to lists of bits
        bin_heaps = [list(bin_heap) for bin_heap in bin_heaps] 
        bin_heaps = [list(map(int,bin_heap)) for bin_heap in bin_heaps]
        # compute ethe nim sum
        _nim_sum = 0
        for i in range(n):
            digit = 0
            for bin_heap in bin_heaps:
                digit = (digit + bin_heap[-1-i]) % 2
            _nim_sum += digit*(2**i)
        return _nim_sum
    
    def outcome(self):
        ''' Returns the outcome of the game. Overloads outcome method from 
        UpDown class.
        -------
        str
            'Previous' if the game is a second player win, and 'Next' if the
            game is a first player win.
        '''
        ns = self.nim_sum()
        if ns == 0:
            return 'Previous'
        else:
            return 'Next'