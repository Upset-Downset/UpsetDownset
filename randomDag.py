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

def class_cardinality(G):
    ''' Returns the class cardinality of the DAG 'G' under the equivalence 
    relation: X~Y if and only if the trasnitive reductions of X and Y are 
    the same.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
        
    Returns
    -------
    int
        the class cardinality of 'G'.
    '''
    # get a list of the components for G
    components = digraph.connected_components(G)
    
    # for each component, compute the difference between the nuber of
    # edges in th transitive closure (li) and transitive reduction (ri)
    # and keep a running sum.
    component_sum = 0
    for nodes in components:
        subG = digraph.subgraph(G, nodes)
        l_component = digraph.number_of_edges(
            digraph.transitive_closure(subG)
            )
        r_component = digraph.number_of_edges(
            digraph.transitive_reduction(subG)
            )
        component_sum += (l_component - r_component)
        
    return pow(2, component_sum)

def markov_step(Xt, class_card, num_nodes):
    ''' Returns the DAG generated after one step of the Matrkov process 
    starting from the DAG 'Xt'.
    
    Parameters
    ----------
    X_t : dict
        adjacency representation of the DAG 'Xt': the t-th step 
        in the markov chain. (Adjacency lists keyed by node.)
    num_nodes : int (nonegative)
        number of nodes in Xt
    class_card : int
        the class cardinality of the DAG 'Xt'.
        
    Returns
    -------
    dict
        returns the adjacency representation of the DAG 'Xt1',
        which is the t+1-th step in the markov chain generating a
        random DAG.
    '''
    # Randomly sample two integers from 0,...,n-1. 
    i, j = np.random.choice(num_nodes, 2)
    if i == j:
        return Xt, class_card

    # check to see if the edge (i,j) is in X_t, if it is, probabilistically 
    # choose to move to the graph 'Z' created by removing (i,j) from Xt.
    # else, probabilistically choose to move to the graph 
    # 'Y' created by adding it (as long as Y is acyclic).
    Xt_class_card = class_card
    if j in Xt[i]:
        Xt[i].remove(j)
        Z_class_card = class_cardinality(Xt)
        prob = min(1 , Xt_class_card / Z_class_card)
        choice = np.random.choice(['z','x'], p=[prob, 1-prob])
        if choice == 'z':
            class_card = Z_class_card
        else:
            Xt[i].append(j)
    else:
        # add the edge (i,j) to Xt if adding such an edge gives an 
        # acyclic graph. otherwise, it will return the original graph Xt.
        Xt[i].append(j)     
        if not digraph.is_acyclic(Xt) :
            Xt[i].remove(j) 
        else:  
            Y_class_card = class_cardinality(Xt)
            prob = min(1 , Xt_class_card / Y_class_card)
            choice = np.random.choice(['y','x'], p=[prob, 1-prob])
            if choice == 'y':  
                class_card = Y_class_card
            else: 
                Xt[i].remove(j) 
                
    Xt1 = Xt
                
    return Xt1, class_card

def markov_chain(G, num_steps, class_card=None):
    ''' Returns the DAG resulting from 'num_steps' iterations of the markov
    process starting from the DAG 'G'.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
    num_steps : int (nonnegative)
        number of steps in the markov chain. 
    class_card: int (nonegative)
        the class cardinality of the DAG 'G'. If None, it will compute it
        
    Returns
    -------
    H : adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.) The final graph in the markov chain of length 'num_steps' 
        starting with the DAG  'G'.
    '''
    num_nodes = len(G)
    if class_card is None:
        G_class_card = class_cardinality(G)
    else:
        G_class_card = class_card
    for _ in range(num_steps):
        H, H_class_card = markov_step(G, G_class_card, num_nodes)
        
    return H, H_class_card

def uniform_random_dag(num_nodes, 
                       exp=2, 
                       extra_steps=0, 
                       X0=None, 
                       class_card=None):
    ''' Returns a random DAG (almost uniformly, as viewed in terms of 
    the equivalence classes of trasnistively reduced DAGS) on 'num_nodes' using 
    the markov process described by Patryk Kozieł and Małgorzata Sulkowska.
    
    Parameters
    ----------
    num_nodes : int (nonegative)
        The number of nodes you want the returned DAG to have.
    exp : int
        The exponent on num_nodes determining the number of steps taken in the 
        markov process. In the report by Kozieł and Sulkowska, they conjecture
        that it is sufficient for this to be 2 to obtain a random DAG almost 
        uniformly selected. Thus, we make the default 2.
    extra_steps : int, optional
        Use this to fine tune the number of steps taken in the markov chain. 
        The default is 0.
    X_0 : dict, optional
        adjacency representation of the DAG from which to start the markov 
        process. (Adjacency lists keyed by node.) The default is the empty 
        DAG on 'num_nodes'. 
        
    Returns
    -------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
    '''
    if X0 is None:
        X0 = {i:[] for i in range(num_nodes)}
        X0_class_card = 1
    else:
        if class_card is None:
            X0_class_card = class_cardinality(X0)
        else:
            X0_class_card = class_card
            
    num_steps = int(
        pow(num_nodes,exp) + extra_steps
        )      
    G, _ = markov_chain(
        X0, 
        num_steps, 
        class_card=X0_class_card
        )
    
    return G
