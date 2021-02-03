#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:36:29 2021

@author: Charles Petersen and Jamison Barsotti

NOTE: This module is based on an algorithm proposed by
Patryk Kozieł and Małgorzata Sulkowska to generate a random DAG on n nodes, 
in a uniform fashion, with O(n^2) complexity (conjectured). In practive
we have found it an efficient source of generating DAGs for our purposes.
For more information, we refer the reader to their paper Uniform Random Posets.

Citation information:
    @misc{kozieł2018uniform,
      title={Uniform random posets}, 
      author={Patryk Kozieł and Małgorzata Sulkowska},
      year={2018},
      eprint={1810.05446},
      archivePrefix={arXiv},
      primaryClass={math.CO}
}
"""
import numpy as np
import digraph
import copy


def class_card(G):
    '''
    Computes the class cardinality of the DAG 'G'.

    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacencylists keyed 
        by node.)

    Returns
    -------
    int
        returns the class cardinality of 'G'
    '''
    #Get a list of the components for G
    components = digraph.connected_components(G)
    
    #For each component, compute the difference between the size
    # of the transitive closure (li) and transitive reduction (ri)
    # and keep a running sum.
    comp_sum = 0
    for i in components:
        sub_G = digraph.subgraph(G, i)
        li = digraph.number_of_edges(digraph.transitive_closure(sub_G))
        ri = digraph.number_of_edges(digraph.transitive_reduction(sub_G))
        comp_sum += li - ri
    
    return 2**comp_sum

def markov_step(X_t, cc, n):
    '''
    Computes one step in the markov chain generating a random DAG.

    Parameters
    ----------
    X_t : dict
        adjacency representation of the DAG 'X_t',
        the t-th step in the markov chain  
    n : int
        number of nodes in X_t
    cc : int
        the class cardinality of the DAG 'X_t'.
        [One should note that making the class_card computation
         is very complex, but in a markov chain, the previous iteration
         contains this calculation for 'X_t', therefore 
         it doesn't need to be recalculated]

    Returns
    -------
    dict
        returns the adjacency representation of the DAG 'X_t1',
        which is the t1-th step in the markov chain generating a
        random DAG.

    '''
    #Randomly sample two integers from 0,...,n-1. 
    i, j = np.random.randint(0,n), np.random.randint(0,n)
    
    #If i==j adding (i,j) will create a cyclic graph, so return X_t
    if i == j:
        return X_t, cc
    
    X_t_class_card = cc
    
    #Check to see if the edge is in X_t, if it is, probabilistically 
    # choose to move to the graph 'Z' created by removing it.
    # Else, probabilistically choose to move to the graph 
    # 'Y' created by adding it (as long as Y is acyclic).
    if i in X_t[j]:
        Z = copy.deepcopy(X_t)
        Z[j].remove(i)
        Z_class_card = class_card(Z)
        prob = min(1 , X_t_class_card / Z_class_card)
        choice = np.random.choice(['z','x'], p=[prob, 1-prob])
        if choice == 'z':
            X_t1 = Z
            cc = Z_class_card
        else:
            X_t1 = X_t
    else:
        #This adds the edge (i,j) if adding such an edge gives an acyclic graph.
        # Otherwise, it will return the same graph X_t.
        Y = digraph.add_edge(X_t,(i,j))    
        if Y == X_t:
            X_t1 = X_t 
        else:
            Y_class_card = class_card(Y)
            prob = min(1 , X_t_class_card / Y_class_card)
            choice = np.random.choice(['y','x'], p=[prob, 1-prob])
            if choice == 'y':
                X_t1 = Y
                cc = Y_class_card
            else:
                X_t1 = X_t
    return X_t1, cc

def markov_chain(G, steps, cc=None):
    '''
    Performs steps number of markov chain steps from the DAG 'G' and returns
    the final DAG 'H' in the chain. *DOES NOT MUTATE G*

    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
    steps : int
        number of steps in the markov chain. 
    cc: int
        the class cardinality of the DAG 'G'. If None, it will compute it

    Returns
    -------
    H : adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
        The final graph in the markov chain of length steps starting with the
        DAG  'G'.

    '''
    n = len(G.keys())
    H = copy.deepcopy(G)
    if cc == None:
        H_class_card = class_card(H)
    else:
        H_class_card = cc

    for _ in range(steps):
        H, H_class_card = markov_step(H, H_class_card, n)
    return H

def almost_uniform_random_graph(n, steps_exp=2, alpha=0, X_0=None, cc=None):
    '''
    Generate a random DAG (almost uniformly) on n nodes using the markov process described
    by Patryk Kozieł and Małgorzata Sulkowska.

    Parameters
    ----------
    n : int
        The number of nodes you want your graph to be
    steps_exp : int
        The exponent on n determining the number of steps taken in the markov process. 
        In the report by Kozieł and Sulkowska, they conjecture
        that it is sufficient for this to be 2 to obtain a random DAG almost uniformly
        selected. Thus, we make the default 2.
    alpha : int, optional
        Use this to fine tune the number of steps taken in the markov chain. The default is 0.
    X_0 : dict, optional
        The default is to start with the empty DAG on n nodes.
        If you would like to start elsewhere, then X_t is the
        adjacency representation of the directed graph 'X_t'. (Adjacency lists keyed 
        by node.)

    Returns
    -------
    G : dict
        An adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)

    '''
    if X_0 == None:
        X_0 = {i:[] for i in range(n)}
        X_0_class_card = 1
    else:
        if cc == None:
            X_0_class_card = class_card(X_0)
        else:
            X_0_class_card = cc
            
    G = markov_chain(X_0, int(n**steps_exp + alpha), cc=X_0_class_card)
    return G

    
