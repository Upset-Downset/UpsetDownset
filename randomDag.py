#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti

NOTE: This module is based on an algorithm proposed by
Patryk Kozieł and Małgorzata Sulkowska to generate random transitively 
reduced labelled DAGs on n nodes (partially orderd set of cardinailty n), in a
uniform fashion, via a Markov process with n^2 steps (conjectured). 
In practive we have found it to be an efficient source of generating 
DAGs for our purposes. For more information, we refer the reader to their
paper Uniform Random Posets. Citation information:
    
    @misc{kozieł2018uniform,
      title={Uniform random posets}, 
      author={Patryk Kozieł and Małgorzata Sulkowska},
      year={2018},
      eprint={1810.05446},
      archivePrefix={arXiv},
      primaryClass={math.CO}}
"""
import numpy as np
import digraph

def class_card(G):
    ''' Returns the class cardinality of the DAG 'G' under the equivalence 
    relation: X~Y if and only if  the trasnitive reductions of X and Y are 
    the same.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
    Returns
    -------
    int
        returns the class cardinality of 'G'.
    '''
    # get a list of the components for G
    components = digraph.connected_components(G)
    
    # for each component, compute the difference between the nuber of
    # edges in th transitive closure (li) and transitive reduction (ri)
    # and keep a running sum.
    comp_sum = 0
    for i in components:
        sub_G = digraph.subgraph(G, i)
        li = digraph.number_of_edges(digraph.transitive_closure(sub_G))
        ri = digraph.number_of_edges(digraph.transitive_reduction(sub_G))
        comp_sum += li - ri   
        
    return 2**comp_sum

def markov_step(X_t, cc, n):
    ''' Returns the DAG generated after one step of the Matrkov process 
    starting from the DAG 'X_t'.
    
    Parameters
    ----------
    X_t : dict
        adjacency representation of the DAG 'X_t': the t-th step 
        in the markov chain. (Adjacency lists keyed by node.)
    n : int (nonegative)
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
        which is the t+1-th step in the markov chain generating a
        random DAG.
    '''
    # Randomly sample two integers from 0,...,n-1. 
    i, j = np.random.choice(n, 2)
    if i == j:
        return X_t, cc

    # check to see if the edge (i,j) is in X_t, if it is, probabilistically 
    # choose to move to the graph 'Z' created by removing (i,j) from X_t.
    # else, probabilistically choose to move to the graph 
    # 'Y' created by adding it (as long as Y is acyclic).
    X_t_class_card = cc
    if j in X_t[i]:
        X_t[i].remove(j)
        Z_class_card = class_card(X_t)
        prob = min(1 , X_t_class_card / Z_class_card)
        choice = np.random.choice(['z','x'], p=[prob, 1-prob])
        if choice == 'z':
            X_t1 = X_t
            cc = Z_class_card
        else:
            X_t[i].append(j)
            X_t1 = X_t
    else:
        # add the edge (i,j) to X_t if adding such an edge gives an 
        # acyclic graph. otherwise, it will return the original graph X_t.
        X_t[i].append(j)     
        if not digraph.is_acyclic(X_t) :
            X_t[i].remove(j) 
            X_t1 = X_t
        else:  
            Y_class_card = class_card(X_t)
            prob = min(1 , X_t_class_card / Y_class_card)
            choice = np.random.choice(['y','x'], p=[prob, 1-prob])
            if choice == 'y':  
                X_t1 = X_t
                cc = Y_class_card
            else: 
                X_t[i].remove(j) 
                X_t1 = X_t
                
    return X_t1, cc

def markov_chain(G, steps, cc=None):
    ''' Returns the DAG resulting from 'steps' iterations of the markov
    process starting from the DAG 'G'.
    
    Parameters
    ----------
     : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
    steps : int (nonnegative)
        number of steps in the markov chain. 
    cc: int (nonegative)
        the class cardinality of the DAG 'G'. If None, it will compute it
        
    Returns
    -------
    H : adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.) The final graph in the markov chain of length 'steps' starting 
        with the DAG  'G'.
    '''
    n = len(G)
    if cc == None:
        G_class_card = class_card(G)
    else:
        G_class_card = cc

    for _ in range(steps):
        H, H_class_card = markov_step(G, G_class_card, n)
        
    return H, H_class_card

def uniform_random_dag(n, exp=2, alpha=0, X_0=None, cc=None):
    ''' Returns a random DAG (almost uniformly, as viewed in terms of 
    the equivalence classes of trasnistively reduced DAGS) on 'n' nodes using 
    the markov process described by Patryk Kozieł and Małgorzata Sulkowska.
    
    Parameters
    ----------
    n : int (nonegative)
        The number of nodes you want the returned DAG to have.
    exp : int
        The exponent on n determining the number of steps taken in the markov 
        process. In the report by Kozieł and Sulkowska, they conjecture
        that it is sufficient for this to be 2 to obtain a random DAG almost 
        uniformly selected. Thus, we make the default 2.
    alpha : int, optional
        Use this to fine tune the number of steps taken in the markov chain. 
        The default is 0.
    X_0 : dict, optional
        adjacency representation of the DAG from which to start the markov 
        process. (Adjacency lists keyed by node.) The default is the empty 
        DAG on 'n' nodes. 
        
    Returns
    -------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
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
            
    G = markov_chain(X_0, int(n**exp + alpha), cc=X_0_class_card)
    
    return G
