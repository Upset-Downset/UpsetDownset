#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:58:21 2020

@author: charlie
"""
import dagUtility as dag

class Poset(object):
    def __init__(self, relations, coloring, cover = False): 
        ''' Class for red-blue-green colored posets. Not to be accessed directly, 
        but through the subclass heirarchy via creation of a Poset sublass on a 
        specific poset or family of posets.(**see randomUpDown)
    
        Parameters
        ----------
        relations : dict
            each key is an element in the poset and the corresponding value is 
            a list of elements greater than the key. The provided 'relations' may 
            contain some or all of the relational data of the poset with a 
            few caveats:
                - elements are to be labelled by consecutive nonnegatove integers 
                staring from 0.  
                - The dictionary must contain a key for each element in the poset.
                (The dict value of a maximal element should be an empty list.)
                - Reflexivity is assumed. Do not provide the reflexive relations.
                  (This is akin to a cycle in the corresponding dag.)
                - Anti-symmetry is checked via acylicity of the corresponding dag.
                - Transitivity is forced via reachability on the corresponding dag.
                  (In particular, the poset returned is the transitive closure of the 
                  provided relations.)
        coloring : dict,
             coloring map: color keyed by element, where 1 (resp. 0,-1) 
             represent blue (resp. green, red). 
        cover : bool, optional
            set to True if the given 'relations' are the (upper) covering relations, 
            and anti-symmetry need not be checked. The default is False.

        Returns
        -------
        None

        '''
        # If cover relations not provided, check anti-symmetry 
        if not cover:
            assert dag.isAcyclic(relations), 'The given relations are not anti-symmetric.'
            self._covers = dag.transitiveReduction(relations)
        # cover relations provided
        else:
            self._covers = relations
        self._colors = coloring

        
##############################################################################    
#################### GET HIDDEN ATTRIBUTES ####################################
##############################################################################
               
    def covers(self):
        ''' Returns the (upper) cover relations of the poset.

        Returns
        -------
        dict
            list of (upper) covers keyed by element.
            
        '''
        return self._covers
            
          
    def colors(self):
        ''' Returns the coloring map on the poset.

        Returns
        -------
        dict
            colors keyed by element, where 1 (resp. 0,-1) represent blue 
            (resp. green, red).

        '''
        return self._colors

##############################################################################    
#################### METHODS ON POSET#########################################
##############################################################################
    
    def elements(self):
        ''' Returns a list of all elements in the poset.  

        Returns
        -------
        list
            all elements in the poset

        '''
        return list(self._covers)
    
    def __len__(self):
        ''' returns the cardinality of the poset. Overloads the 'len' operator.

        Returns
        -------
        int
            number of elements in the poset.

        '''
        return len(self._covers)
    
    def colorSum(self):
        ''' Returns the sum of colors over all elemenets in the poset.

        Returns
        -------
        int
            the sum of colors of aall elemenets in the poset, 1where 1 (resp. 0,-1) 
            represent blue (resp. green, red).

        '''
        return sum(self._colors[x] for x in self.elements())
    
    def coverSum(self):
        ''' Returns the number of covering relations.
    
        Returns
        -------
        int
           the number of covering relations in the poset. I.e., the number of 
           edges in the Hasse diagram.

        '''
        return dag.numberOfEdges(self._covers)
        
    def height(self):
        ''' Returns the height of the poset.
        
        Returns
        -------
        int
        the length of a maximum chain in the poset. I.e., the 
        height of the Hasse diagram of the poset.

        '''
        return dag.longestPathLength(self._covers) 
    
    def maxs(self):
        ''' Returns all maixmal elements of the poset.
        
        Returns
        -------
       list
          all maximal elements of the poset.

        '''
        maxs = []
        for x in self.elements():
            if not self._covers[x]:
                maxs.append(x)
        return maxs
    
    def mins(self):
        ''' Returns all minimal elements of the poset.
        Returns
        -------
       list
          all minimal elements of the poset.

        '''
        neg = self.coloredDual()
        mins = []
        for x in neg.elements():
            if not neg.cover(x):
                mins.append(x)
        return mins
    
##############################################################################    
#################### METHODS ON ELEMENTS OF POSET#############################
##############################################################################    
    def cover(self, x):
        ''' Returns all (upper) covers of element 'x'.

        Parameters
        ----------
        x : int (nonnegative)
            element of poset.

        Returns
        -------
        list
            all (upper) covers of 'x'.

        '''
        return self._covers[x]
    
    def color(self, x):
        ''' Returns the color of element 'x'.
        Parameters
        ----------
        x : int (nonnegative)
            element of poset.

        Returns
        -------
        int
            color of 'x' where 1 (resp. 0,-1) represent blue (resp. green, red).

        '''
        return self._colors[x]
      
    def upset(self, x):
        ''' Returns the upset of 'x'.
        
        Parameters
        ----------
        x : int (nonnegative)
            element of the poset

        Returns
        -------
        list
            all elements of the poset which are greater than or equal to 'x'.

        '''
        upset = dag.descendants(self._covers, x)
        upset.append(x)
        return upset
    
    def downset(self, x):
        ''' Returns the downset of 'x'.
            
        Parameters
        ----------
        x : int
            element of the poset.

        Returns
        -------
        list
            all elements of the poset which are less than or equal
            to 'x'.

        '''
        downset = dag.ancestors(self._covers, x)
        downset.append(x)
        return downset
    
    def level(self, x):
        ''' Returns the level of 'x' in the Hasse diagram of the poset.

        Parameters
        ----------
        x : int
            element of underlying poset

        Returns
        -------
        int
            the length of a maximum chain ending in 'x' in the poset.
            I.e., the height of the element 'x' in the Hasse diagram of 
            the poset.

        '''
        downset = self.downset(x)
        downset_dag= dag.subgraph(self._covers, downset)
        return dag.longestPathLength(downset_dag)
    
##############################################################################
######### PLOT POSET #########################################################
##############################################################################
    def hasse(self):
        '''Plots the Hasse diagram of the poset.

        Returns
        -------
        None.

        '''
    
        # HARRY PLOTTER...?
        return None
    
##############################################################################   
########### NEW POSETS FROM OLD ##############################################
##############################################################################
    def coloredDual(self):
        ''' Returns the dual poset with opposite coloring.

        Returns
        -------
        Poset object
            the poset having relations dual to 'self' and with opposite coloring.

        '''
        dual = dag.reverse(self._covers)
        flipped_coloring = {x:-self.color(x) for x in self.elements()}
        return Poset(dual, flipped_coloring, cover = True)
    
    def subposet(self, subset):
        ''' Returns the poset on 'subset' with induced relations and coloring.

        Parameters
        ----------
        subset : list
            subset of elements of poset.

        Returns
        -------
        Poset object
            the poset on 'subset' having relations and coloring induced
            from 'self'.       

        '''
        sub_dag = dag.subgraph(self._covers, subset)
        sub_coloring = {x:self.color(x) for x in subset}
        return Poset(sub_dag, sub_coloring, cover = True )
    
    def __add__(self, other):
        ''' Returns the disjoint union of posets. Overloads the '+' operator.
        ** Relabels all elemnts with consecutive nonnegative integers starting 
        from 0.
        
        Parameters
        ----------
        other : Poset object

        Returns
        -------
        Poset object
            poset which contains all elements (relabelled), relations and coloring 
            from both 'self' and 'other' with the added relations that all elements 
            of 'self' and 'other' are incomparable.

        '''
        union = {}
        union_coloring = {}
        # Relabelling for self.
        self_relabel = {}
        self_new_label = 0
        for x in self.elements():
            self_relabel[x] = self_new_label
            self_new_label += 1
        # Relabelling for other.
        other_relabel = {}
        other_new_label = self_new_label
        for x in other.elements():
            other_relabel[x] = other_new_label
            other_new_label += 1
        # Construct coloring and cover relations for disjoint union
        for x in self.elements():
            union[self_relabel[x]] = []
            union_coloring[self_relabel[x]] = self.color(x)
            for y in self.cover(x):
                union[self_relabel[x]].append(self_relabel[y])
        for x in other.elements():
            union[other_relabel[x]] = []
            union_coloring[other_relabel[x]] = other.color(x)
            for y in other.cover(x):
                union[other_relabel[x]].append(other_relabel[y]) 
        return Poset(union, union_coloring, cover = True)
    
    def __or__(self, other):
        ''' Returns the ordinal sum of posets. (This is not a commutative operation).
        Overloads the '|' operator. **Relabels all elemnts with consecutive 
        nonnegative integers starting from 0.

        Parameters
        ----------
        other : Poset object

        Returns
        -------
        Poset object
            poset which contains all elements (relabelled), relations and coloring
            from both 'self' and 'other' with the added relations that every
            element of 'self' is greater than every element of 'other'

        '''
        ordinal = {}
        ordinal_coloring = {}
        # Relabelling for other.
        other_relabel = {}
        other_new_label = 0
        for x in other.elements():
            other_relabel[x] = other_new_label
            other_new_label += 1
        # Relabelling for self.
        self_relabel = {}
        self_new_label = other_new_label
        for x in self.elements():
            self_relabel[x] = self_new_label
            self_new_label += 1
        # Construct coloring and cover relations for disjoint union
        for x in self.elements():
            ordinal[self_relabel[x]] = []
            ordinal_coloring[self_relabel[x]] = self.color(x)
            for y in self.cover(x):
                ordinal[self_relabel[x]].append(self_relabel[y])
        for x in other.elements():
            ordinal[other_relabel[x]] = []
            ordinal_coloring[other_relabel[x]] = other.color(x)
            for y in other.cover(x):
                ordinal[other_relabel[x]].append(other_relabel[y]) 
            for z in self_relabel.values():
                ordinal[other_relabel[x]].append(z)
        ordinal = dag.transitiveReduction(ordinal)
        return Poset(ordinal, ordinal_coloring, cover = True)
        
        